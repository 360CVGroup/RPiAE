from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        if hasattr(self.encoder, "embeddings") and hasattr(self.encoder.embeddings, "mask_token"):
            self.encoder.embeddings.mask_token.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)
        
    # Helper used by training scripts: encoder.kd_last_layer()
    def kd_last_layer(self):
        # Refresh lazily in case encoder parameters are unfrozen later.
        if self._kd_last_layer is None:
            self._kd_last_layer = self.get_kd_last_layer()
        return self._kd_last_layer

    def get_kd_last_layer(self):
        """
        Return a last-layer parameter for adaptive KD weighting.
        """
        fm = self.encoder
        
        # 1) Last transformer block (HF: encoder.layer)
        last_blk = None
        if hasattr(fm, "encoder") and hasattr(fm.encoder, "layer") and len(fm.encoder.layer) > 0:
            last_blk = fm.encoder.layer[-1]

        if last_blk is None:
            # Fallback: use the last trainable parameter.
            for p in reversed(list(fm.parameters())):
                if p.requires_grad:
                    return p
            return None

        # 2) Prefer MLP projection.
        w = getattr(getattr(getattr(last_blk, "mlp", None), "fc2", None), "weight", None)
        if w is not None and w.requires_grad:
            return w

        # 3) Then attention output projection.
        w = getattr(
            getattr(getattr(getattr(last_blk, "attention", None), "output", None), "dense", None),
            "weight",
            None,
        )
        if w is not None and w.requires_grad:
            return w

        # 4) Then block norms.
        for name in ["norm2", "norm1"]:
            w = getattr(getattr(last_blk, name, None), "weight", None)
            if w is not None and w.requires_grad:
                return w

        # 5) Final layernorm.
        w = getattr(getattr(fm, "layernorm", None), "weight", None)
        if w is not None and w.requires_grad:
            return w

        # 6) Fallback: last trainable parameter.
        for p in reversed(list(fm.parameters())):
            if p.requires_grad:
                return p

        return None
    
    @torch.no_grad()
    def get_intermediate_layers(self, x: torch.Tensor, n_last_blocks: int = 4, return_class_token=True):
        out = self.encoder(x, output_hidden_states=True)
        hidden_states = out.hidden_states  # Tuple with embedding output and per-layer outputs.

        selected = hidden_states[-n_last_blocks:]
        results = []
        for h in selected:
            # h: [B, 1+4+N, C]
            cls_token = h[:, 0]
            patch_tokens = h[:, 5:]
            results.append((patch_tokens, cls_token))
        return results
