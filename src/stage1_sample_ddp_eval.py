# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs distributed reconstructions with a pre-trained stage-1 model.
Inputs are loaded from an ImageFolder dataset, processed with center crops,
and the reconstructed images are saved as .png files alongside a packed .npz.
"""
import argparse
import math
import os
import sys
from typing import List
from eval import evaluate_reconstruction_distributed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import json
from sample_ddp import create_npz_from_sample_folder
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    """Replace OS separators to keep path components valid."""
    return component.replace(os.sep, "-")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    stage1_config, *_ = parse_configs(args.config)
    if stage1_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    model = instantiate_from_config(stage1_config).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(args.data_path, transform=transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::world_size]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = stage1_config.get("target", "stage1")
    ckpt_path = stage1_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    folder_name = "-".join(folder_components)
    possible_folder_name = os.environ.get('SAVE_FOLDER', None)
    if possible_folder_name:
        folder_name = possible_folder_name
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving reconstructed samples at {sample_folder_dir}")
    dist.barrier()
    loader = DataLoader(
        subset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    local_total = len(rank_indices)
    iterator = tqdm(loader, desc="Stage1 recon", total=math.ceil(local_total / args.per_proc_batch_size)) if rank == 0 else loader
    eval_metrics = ("rfid", "psnr", "ssim", "lpips")

    # Default values sourced from CLI args.
    global_step = 0  # Optionally replace with a checkpoint/trainer step.
    reference_npz_path = args.ref_npz_path
    batch_size = args.per_proc_batch_size  # Per-GPU batch size.

    eval_stats = evaluate_reconstruction_distributed(
        model=model,
        val_dataset=dataset,
        num_samples=requested,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        experiment_dir=sample_folder_dir,
        global_step=global_step,
        autocast_kwargs=autocast_kwargs,
        metric_batch_size=128,
        reference_npz_path=reference_npz_path,
        metrics_to_compute=eval_metrics,
    )

   
    if rank == 0:
        if eval_stats is not None:
            for k, v in eval_stats.items():
                print(f"eval/{k}: {v}")

        payload = {
            "metrics": eval_stats if eval_stats is not None else {},
        }

        out_json = os.path.join(sample_folder_dir, f"eval_metrics_step{global_step:07d}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Dumped metrics json to: {out_json}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images.")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store reconstructed samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4, help="Number of images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to reconstruct (defaults to full dataset).")
    parser.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision mode.")
    parser.add_argument("--ref_npz_path", type=str, default=None,
                        help="Optional fixed reference .npz path. If not set, reference NPZ is built automatically from current input images.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    args = parser.parse_args()
    main(args)
