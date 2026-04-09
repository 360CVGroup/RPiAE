---
license: apache-2.0
language:
- en
---

<p align="center">
  <img src="assets/logo.png" alt="RPiAE Logo" width="200">
</p>

<h1 align="center">RPiAE: A Representation-Pivoted Autoencoder Enhancing Both Image Generation and Editing</h1>

[![Project Page](https://img.shields.io/badge/Project%20Page-RPiAE-1f6feb?style=flat)](https://arthuring.github.io/RPiAE-page/)
[![GitHub](https://img.shields.io/badge/GitHub-360CVGroup%2FRPiAE-181717?style=flat&logo=github)](https://github.com/360CVGroup/RPiAE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-RPiAE-ff9900?style=flat)](https://huggingface.co/qihoo360/RPiAE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.19206-B31B1B?style=flat)](https://arxiv.org/abs/2603.19206)

This repository contains the PyTorch implementation of **RPiAE** and the corresponding latent diffusion training pipeline.

RPiAE follows a two-stage pipeline:
- **Stage 1 (image reconstruction):** train the RPiAE model to learn high-quality latent representations and reconstruction.
- **Stage 2 (class conditional image generation):** train a diffusion transformer in the pretrained RPiAE latent space for image generation.

## TODO

- [x] Release the RPiAE model
- [x] Release the training code for RPiAE
- [x] Release pretrained weights
- [ ] Release the training code for T2I

## Environment

### Dependency Setup
1. Create environment and install via `uv`:
   ```bash
   conda create -n rpiae python=3.10 -y
   conda activate rpiae
   pip install uv
   
   # Install PyTorch 2.8.0 with CUDA 12.9 # or your own cuda version
   uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 
   
   # Install other dependencies
   uv pip install -r requirements.txt
   ```

## Data & Model Preparation

### Download Pre-trained Models

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/qihoo360/RPiAE).

Download the pretrain weights to `./model_weights`:

```bash
hf download qihoo360/RPiAE \
  --repo-type model \
  --local-dir ./model_weights
```

### Prepare Dataset

1. Download ImageNet-1k.
2. Point scripts to the training split or validation split via `--data-path`.

## Config-based Initialization

All training and sampling entrypoints are driven by OmegaConf YAML files. A
single config describes the Stage 1 autoencoder, the Stage 2 diffusion model,
and the solver used during training or inference. A minimal example looks like:

```yaml
stage_1:
   target: stage1.RPiAE / stage1.RPiAE_VB
   params: { ... }
   ckpt: <path_to_ckpt>  

stage_2:
   target: stage2.models.lightningDiT.LightningDiT
   params: { ... }
   ckpt: <path_to_ckpt>  

transport:
   params:
      path_type: Linear
      prediction: velocity
      ...
sampler:
   mode: ODE
   params:
      num_steps: 50
      ...
guidance:
   method: cfg/autoguidance
   scale: 1.0
   ...
misc:
   latent_size: [64, 16, 16]
   num_classes: 1000
training:
   ...
eval:
   ...
```

- `stage_1` defines the **RPiAE training process** (reconstruction-oriented training).
- `stage_2` defines the **generation model** (LightningDiT) in the RPiAE latent space.
- `transport`, `sampler`, and `guidance` control ODE/SDE solving and guidance strategy.
- `misc` stores latent shape and shared constants.
- `training` and `eval` contain optimization and online evaluation settings.
- Stage 1 training configs additionally include a top-level `gan` block for discriminator and loss schedule.

### Provided Configs

#### Stage1

- Pretrained:
  - `configs/stage1/pretrained/DINOv2-B_decXL_RPiAE.yaml`
- Training:
  - `configs/stage1/training/DINOv2-B_decXL_RPiAE_stage1.yaml`
  - `configs/stage1/training/DINOv2-B_decXL_RPiAE_stage2.yaml`
  - `configs/stage1/training/DINOv2-B_decXL_RPiAE_stage3.yaml`

#### Stage2

- Training:
  - `configs/stage2/training/ImageNet256/LightingDiT-XL_f16d64rpiae-v2b_vitxl.yaml`
- Sampling:
  - `configs/stage2/sampling/ImageNet256/LightingDiT-XL_d64rpiae-v2b_vitxl.yaml`
  - `configs/stage2/sampling/ImageNet256/LightingDiT-XL_d64rpiae-v2b_vitxl_AG.yaml`

## RPiAE Training & image reconstruction evaluation
Use the provided shell scripts with the corresponding configs.

If you use wandb logging, also configure:

```bash
EXPERIMENT_NAME=
ENTITY=
PROJECT=
```

### Single-GPU Training

#### Train Stage 1

```bash
bash run_train_stage1_rpiae_s1.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage1.yaml
```

#### Train Stage 2

```bash
bash run_train_stage1_rpiae_s23.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage2.yaml
```

#### Train Stage 3

```bash
bash run_train_stage1_rpiae_s23.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage3.yaml
```

### Multi-GPU Training

For multi-GPU or multi-node training, please use the `*_mult*.sh` scripts instead.

Before launching, configure the distributed variables in the shell script:

```bash
RANK=
MASTER_ADDR=
GPUS_PER_NODE=
NNODES=
MASTER_PORT=
```


Then run the corresponding multi-GPU scripts.

#### Train Stage 1

```bash
bash run_train_stage1_mult_rpiae_s1.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage1.yaml
```

#### Train Stage 2

```bash
bash run_train_stage1_mult_rpiae_s23.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage2.yaml
```

#### Train Stage 3

```bash
bash run_train_stage1_mult_rpiae_s23.sh \
  configs/stage1/training/DINOv2-B_decXL_RPiAE_stage3.yaml
```

### Reconstruction Evaluation

```bash
bash run_sample_reconstruction_eval.sh \
  configs/stage1/pretrained/DINOv2-B_decXL_RPiAE.yaml
```

## Latent Diffusion Transformer Training & Class conditional Image Generation Evaluation

### Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/train_diffusion_rpiae.py \
  --config <training_config> \
  --data-path <imagenet_train_split> \
  --results-dir ckpts/diffusion \
  --compile \
  --precision fp32
```

For multi-GPU / multi-node training, use:

```bash
bash run_train_mult_diffusion.sh \
  configs/stage2/training/ImageNet256/LightingDiT-XL_f16d64rpiae-v2b_vitxl.yaml
```

For multi-node launch, set:

```bash
export RANK=<node_rank>
export MASTER_ADDR=<master_node_ip_or_hostname>
```

Although `bf16` is supported, we recommend using **fp32** for more stable training.

### Sampling

`src/sample.py` uses the same config schema to draw a small batch of images on a
single device and saves them to `sample.png`:

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42
```

### Distributed sampling for evaluation

`src/sample_ddp.py` parallelises sampling across GPUs, producing PNGs and an
FID-ready `.npz`:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision fp32/bf16 \
  --label-sampling equal
```

`--label-sampling {equal,random}`: `equal` uses exactly 50 images per class for FID-50k; `random` uniformly samples labels. We use `equal` by default. We recommend using fp32 when model FID is low.

Autoguidance and classifier-free guidance are controlled via the config’s `guidance` block.

## Evaluation

### ADM Suite FID setup

Use the ADM evaluation suite to score generated samples:

1. Clone the repo:

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```

## Acknowledgement

This code is built upon the following repositories:

* [SiT](https://github.com/willisma/sit)
* [DDT](https://github.com/MCG-NJU/DDT)
* [LightningDiT](https://github.com/hustvl/LightningDiT/)
* [MAE](https://github.com/facebookresearch/mae)
* [RAE](https://github.com/bytetriper/RAE)

## Citation
If you find this repository useful, please consider citing our paper:
```
@misc{RPiAE,
  title={RPiAE: A Representation-Pivoted Autoencoder Enhancing Both Image Generation and Editing},
  author={Yue Gong and Hongyu Li and Shanyuan Liu and Bo Cheng and Yuhang Ma and Liebucha Wu and Xiaoyu Wu and Manyuan Zhang and Dawei Leng and Yuhui Yin and Lijun Zhang},
  year={2026},
  eprint={2603.19206},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.19206},
}
```