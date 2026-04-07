import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None, no_sum=False):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                if not no_sum:
                    return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
                else:
                    return torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
def chw_to_tokens(x):  # x: (B, C, H, W)
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)

def tokens_to_chw(tokens):  # tokens: (B, N, C)
    B, N, C = tokens.shape
    H = W = int(math.sqrt(N))
    assert H * W == N, f"N={N} not square"
    return tokens.transpose(1, 2).contiguous().view(B, C, H, W)


def normalize_training_stage(stage: str) -> str:
    if stage not in {"stage2", "stage3"}:
        raise ValueError(f"Unknown training_stage={stage}, expected one of: stage2, stage3")
    return stage

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...
class VariationalBridgeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_gelu: bool = True,
    ):
        super().__init__()
        assert in_dim % n_heads == 0, f"in_dim={in_dim} must be divisible by n_heads={n_heads}"
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, n_heads, dropout=attn_dropout, batch_first=True)
        self.skip_proj = nn.Linear(in_dim, in_dim)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(in_dim)
        hidden = int(in_dim * mlp_ratio)
        act = nn.GELU() if use_gelu else nn.SiLU()
        self.proj2 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )
        self.to_moments = nn.Linear(in_dim, 2 * latent_dim)

        # init near-identity
        nn.init.zeros_(self.proj2[-1].weight)
        nn.init.zeros_(self.proj2[-1].bias)

    def forward(self, e_tokens: torch.Tensor) -> torch.Tensor:
        x = e_tokens
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = self.skip_proj(x) + self.drop(attn_out)

        x2 = self.norm2(x)
        x = x + self.drop(self.proj2(x2))
        return self.to_moments(x)  # (B,N,2*latent_dim)


class VariationalBridgeDecoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, out_dim)
        blocks = []
        for _ in range(n_layers):
            blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=out_dim,
                    nhead=n_heads,
                    dim_feedforward=out_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, z_tokens: torch.Tensor):
        x = self.in_proj(z_tokens)
        x = self.net(x)
        return self.out_proj(x)  # (B,N,out_dim)

class RPiAE_VB(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        encoder_trainable: bool = True,
        use_teacher: bool = True,

        # ===== VB configs =====
        use_vb: bool = True,
        vb_latent_dim: int = 64,
        vb_drop_cls: bool = True,  
        training_stage: str = "stage3",
        vb_feat_loss_weight: float = 1.0,
        vb_kl_weight: float = 1e-3,
        vb_rec_weight: float = 0.05,    
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        self.eps = eps

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
            self.latent_mean = None
            self.latent_var = None

        # teacher
        self.encoder_trainable = encoder_trainable
        self.encoder.requires_grad_(encoder_trainable)
        if encoder_trainable and hasattr(self.encoder, "encoder"):
            emb = getattr(self.encoder.encoder, "embeddings", None)
            if emb is not None and hasattr(emb, "mask_token"):
                emb.mask_token.requires_grad_(False)

        self.use_teacher = use_teacher
        if self.use_teacher:
            self.teacher_encoder = deepcopy(self.encoder)
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
        else:
            self.teacher_encoder = None
        self.model = nn.Module()
        self.model.z_channels = self.latent_dim
        # ===== VB setup =====
        self.use_vb = use_vb
        self.vb_latent_dim = vb_latent_dim
        self.vb_drop_cls = vb_drop_cls
        self.training_stage = normalize_training_stage(training_stage)
        self.vb_feat_loss_weight = vb_feat_loss_weight
        self.vb_kl_weight = vb_kl_weight
        self.vb_rec_weight = vb_rec_weight

        if self.use_vb:
          
            n_heads = min(8, max(1, self.latent_dim // 64))
            self.vb_feature_encoder = VariationalBridgeEncoder(
                in_dim=self.latent_dim,
                latent_dim=self.vb_latent_dim,
                n_heads=n_heads,
                dropout=0.0,
            )
            self.vb_feature_decoder = VariationalBridgeDecoder(
                out_dim=self.latent_dim,
                latent_dim=self.vb_latent_dim,
                n_layers=6,
                n_heads=n_heads,
                dropout=0.0,
            )
            self.model.z_channels = self.vb_latent_dim
        else:
            self.vb_feature_encoder = None
            self.vb_feature_decoder = None

        # ===== Stage freezing policy =====
        def freeze_module(m: nn.Module, trainable: bool):
            for p in m.parameters():
                p.requires_grad = trainable

        if self.use_vb:
            if self.training_stage == "stage2":
                # Train only vb_feature_encoder and vb_feature_decoder.
                freeze_module(self.encoder, False)
                freeze_module(self.decoder, False)
                if self.teacher_encoder is not None:
                    freeze_module(self.teacher_encoder, False)
                freeze_module(self.vb_feature_encoder, True)
                freeze_module(self.vb_feature_decoder, True)

            elif self.training_stage == "stage3":
                # Train decoder (and encoder if enabled).
                freeze_module(self.vb_feature_encoder, False)
                freeze_module(self.vb_feature_decoder, False)
                # encoder_trainable controls encoder training.
                freeze_module(self.encoder, False)
                freeze_module(self.decoder, True)
            

            else:
                raise ValueError(f"Unknown training_stage={self.training_stage}")

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    def _maybe_norm_latent(self, z: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not self.do_normalization:
            return z
        latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
        latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
        if not inverse:
            return (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z * torch.sqrt(latent_var + self.eps) + latent_mean

    # --------- VB encode/decode on latent map (B,C,H,W) ----------
    def vb_encode(self, z_dec_map: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        z_dec_map: (B, latent_dim, H, W)  # decoder-space latent map
        return posterior over (B, vb_latent_dim, H, W)
        """
        z_tokens = chw_to_tokens(z_dec_map)                  # (B, N, latent_dim)
        moments_tokens = self.vb_feature_encoder(z_tokens)  # (B, N, 2*vb_latent_dim)
        moments_map = tokens_to_chw(moments_tokens)          # (B, 2*vb_latent_dim, H, W)
        return DiagonalGaussianDistribution(moments_map)

    def vb_decode(self, z_vb_map: torch.Tensor) -> torch.Tensor:
        """
        z_vb_map: (B, vb_latent_dim, H, W)
        return z_dec_map: (B, latent_dim, H, W)
        """
        z_tokens = chw_to_tokens(z_vb_map)                  # (B, N, vb_latent_dim)
        dec_tokens = self.vb_feature_decoder(z_tokens)      # (B, N, latent_dim)
        return tokens_to_chw(dec_tokens)                     # (B, latent_dim, H, W)

    def encode(self, x: torch.Tensor, return_detail: bool = False, return_posterior: bool = False):
        assert self.use_vb and self.reshape_to_2d, "This RPiAE_VB assumes use_vb=True and reshape_to_2d=True"

        # resize
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size),
                            mode="bicubic", align_corners=False)

        # encoder input normalize
        x_in = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

        # student tokens
        z_tokens = self.encoder(x_in)  # (B, N, latent_dim)

        # teacher tokens (PIV)
        if self.use_teacher and self.teacher_encoder is not None:
            with torch.no_grad():
                zt_tokens = self.teacher_encoder(x_in)
        else:
            zt_tokens = None

        # noise tokens (train only)
        z_tokens_noise = self.noising(z_tokens) if (self.training and self.noise_tau > 0) else z_tokens

        # tokens -> map
        b, n, c = z_tokens.shape
        hh = ww = int(math.sqrt(n))
        assert hh * ww == n, f"tokens N={n} not square"
        z_clean = z_tokens.transpose(1, 2).contiguous().view(b, c, hh, ww)      # (B, latent_dim, H, W)
        z_noise = z_tokens_noise.transpose(1, 2).contiguous().view(b, c, hh, ww)
        z_t = zt_tokens.transpose(1, 2).contiguous().view(b, c, hh, ww) if zt_tokens is not None else None

        # latent normalization (applies to decoder-space latent only!)
        z_clean = self._maybe_norm_latent(z_clean, inverse=False)
        z_noise = self._maybe_norm_latent(z_noise, inverse=False)
        if z_t is not None:
            z_t = self._maybe_norm_latent(z_t, inverse=False)

        # VB posterior built on decoder-space latent
        z_for_vb = z_noise if self.training else z_clean
        posterior = self.vb_encode(z_for_vb)  # posterior over (B, vb_latent_dim, H, W)

        out = posterior if return_posterior else posterior.mode()
        if return_detail:
            return out, z_clean, z_t
        return out

    def decode(self, z_vb, required_vb: bool = False) -> torch.Tensor:
        """
        z_vb:
        - DiagonalGaussianDistribution OR
        - Tensor (B, vb_latent_dim, H, W)
        return x_rec in the same domain as training images (typically [0,1])
        """
        assert self.use_vb and self.reshape_to_2d
        if isinstance(z_vb, DiagonalGaussianDistribution):
            z_vb_map = z_vb.mode()
        else:
            z_vb_map = z_vb

        # VB -> decoder latent map
        z_dec_map = self.vb_decode(z_vb_map)               # (B, latent_dim, H, W)

        # inverse latent normalization (decoder-space only)
        z_dec_map = self._maybe_norm_latent(z_dec_map, inverse=True)

        # map -> tokens for GeneralDecoder
        b, c, hh, ww = z_dec_map.shape
        z_tokens = z_dec_map.view(b, c, hh * ww).transpose(1, 2).contiguous()   # (B, N, latent_dim)

        out = self.decoder(z_tokens, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(out)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        if required_vb:
            return x_rec, z_dec_map
        return x_rec

    def forward(
        self,
        x: torch.Tensor,
        return_detail: bool = False,
        return_vb: bool = False,
        sample_vb: bool = False,
    ):
        """
        - stage3: x -> encode(z_noise) -> (optional vb bottleneck) -> decode -> x_rec
        - stage2: still returns x_rec (for rec_loss), and returns vb stats for vb_loss
        """
        posterior, z_clean, z_t = self.encode(x, return_detail=True, return_posterior=True)
        z_vb = posterior.sample() if (self.training and sample_vb) else posterior.mode()
        
        x_rec, x_vb_rec = self.decode(z_vb, required_vb=True)
        
        if return_vb:
            return x_rec, z_clean, posterior, x_vb_rec
        return x_rec

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        return self
