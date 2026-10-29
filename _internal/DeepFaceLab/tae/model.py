"""LIAE identity routing over a TAEQI2.1 image codec, implemented in PyTorch."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Internal feature refinement; never bypasses Inter with input latents."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.GroupNorm(1, channels), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GroupNorm(1, channels), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x) * 0.1


class Inter(nn.Module):
    """Two strided convolutions and a resolution-bound global identity bottleneck."""
    def __init__(self, latent_channels, inter_dims, hidden_dims, grid):
        super().__init__()
        self.grid, self.hidden_dims = grid, hidden_dims
        self.features = nn.Sequential(nn.Conv2d(latent_channels, hidden_dims, 3, 2, 1),
            nn.SiLU(), ResidualBlock(hidden_dims), nn.Conv2d(hidden_dims, hidden_dims, 3, 2, 1),
            nn.SiLU(), ResidualBlock(hidden_dims))
        self.compress = nn.Linear(hidden_dims * grid * grid, inter_dims)
        self.expand = nn.Linear(inter_dims, hidden_dims * grid * grid)

    def forward(self, latents):
        x = self.features(latents)
        if x.shape[-2:] != (self.grid, self.grid):
            raise ValueError("Input resolution differs from the saved Inter architecture")
        x = F.silu(self.compress(x.flatten(1)))
        return F.silu(self.expand(x)).view(-1, self.hidden_dims, self.grid, self.grid)


class MaskHead(nn.Module):
    def __init__(self, hidden_dims: int):
        super().__init__()
        width = max(8, hidden_dims // 4)
        self.reduce = nn.Conv2d(hidden_dims, width, 3, padding=1)
        self.refine = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        # Project at latent size before upsampling; avoid a full-resolution
        # hidden_dims-wide activation. Final logits are learned at pixel size.
        x = F.silu(self.reduce(features))
        x = F.interpolate(x, size=(max(1, size[0] // 4), max(1, size[1] // 4)),
                          mode="bilinear", align_corners=False)
        x = F.silu(self.refine(x))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.output(x).sigmoid()


class LIAEModel(nn.Module):
    """Source [AB,AB], destination [B,AB], and swap [AB(dst),AB(dst)].

    Train with source and destination reconstructions. The swap branch has no
    paired target. Only the frozen encoder is excluded from checkpoints; decoder state is always saved.
    Images and latents use NCHW internally; the public DFL adapter owns BGR/HWC.
    """

    def __init__(self, vae: nn.Module, inter_dims: int = 256, hidden_dims: int = 96, resolution: int = 256):
        super().__init__()
        if inter_dims < 4 or hidden_dims < 4:
            raise ValueError("Inter and hidden dimensions must be >=4")
        if not isinstance(vae, nn.Module) or not getattr(vae, "latent_channels", 0):
            raise TypeError("vae must be an nn.Module with latent_channels, encode and decode")
        self.vae = vae
        self.resolution = resolution
        if resolution < 64 or resolution % 64:
            raise ValueError("Inter resolution must be a positive multiple of 64")
        channels = int(vae.latent_channels)
        self.inter_AB = Inter(channels, inter_dims, hidden_dims, resolution // 64)
        self.inter_B = Inter(channels, inter_dims, hidden_dims, resolution // 64)
        self.fusion = nn.Sequential(nn.Conv2d(hidden_dims * 2, hidden_dims, 1), nn.SiLU(), ResidualBlock(hidden_dims))
        self.up1 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims, 3, padding=1), nn.SiLU(), ResidualBlock(hidden_dims))
        self.up2 = nn.Sequential(nn.Conv2d(hidden_dims, channels, 3, padding=1), nn.SiLU(), ResidualBlock(channels))
        self.to_latent = nn.Conv2d(channels, channels, 3, padding=1)
        self.mask_head = MaskHead(channels)
        self.architecture = {"name": "taeqi21-liae-v1", "inter_dims": inter_dims,
                             "hidden_dims": hidden_dims, "resolution": resolution,
                             "bottleneck_grid": resolution // 64, "latent_channels": channels,
                             "direct_input_skip": False}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.parameters()
                if parameter.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                if not key.startswith("vae.encoder.")}

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {key for key in self.state_dict() if not key.startswith("vae.encoder.")}
        if set(state) != expected:
            raise ValueError(f"LIAE state mismatch: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("vae.encoder.") for key in result.missing_keys):
            raise ValueError("LIAE checkpoint is incomplete")

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError("Input resolution differs from checkpoint")
        self.vae.assert_frozen()
        with torch.no_grad():
            return self.vae.encode(image).detach()

    def _features(self, latent: torch.Tensor, domain: str, *, ab: torch.Tensor | None = None) -> torch.Tensor:
        ab = self.inter_AB(latent) if ab is None else ab
        if domain in {"src", "swap"}:
            pair = torch.cat([ab, ab], dim=1)
        elif domain == "dst":
            pair = torch.cat([self.inter_B(latent), ab], dim=1)
        else:
            raise ValueError("domain must be src, dst, or swap")
        x = self.fusion(pair)
        x = self.up1(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.up2(F.interpolate(x, scale_factor=2, mode="nearest"))

    def _mask(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.mask_head(features, size)

    def _decode(self, features: torch.Tensor, size: tuple[int, int]) -> dict[str, torch.Tensor]:
        rgb = self.vae.decode(self.to_latent(features))
        if rgb.shape[-2:] != size:
            raise RuntimeError("VAE decoded image dimensions do not match the aligned face")
        return {"image": rgb, "mask": self._mask(features, size)}

    def reconstruct(self, image: torch.Tensor, domain: str = "src") -> dict[str, torch.Tensor]:
        if domain not in {"src", "dst"}:
            raise ValueError("Reconstruction domain must be src or dst")
        latent = self._encode(image)
        return self._decode(self._features(latent, domain), image.shape[-2:])

    def reconstruct_src(self, src: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.reconstruct(src, domain="src")
        return {"src": result["image"], "src_mask": result["mask"]}

    def reconstruct_dst(self, dst: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.reconstruct(dst, domain="dst")
        return {"dst": result["image"], "dst_mask": result["mask"]}

    def forward(self, src: torch.Tensor, dst: torch.Tensor, *, include_swap: bool = False) -> dict[str, torch.Tensor]:
        result = self.reconstruct_src(src)
        latent = self._encode(dst)
        ab = self.inter_AB(latent)
        destination = self._decode(self._features(latent, "dst", ab=ab), dst.shape[-2:])
        result.update(dst=destination["image"], dst_mask=destination["mask"])
        if include_swap:
            # Inference can share frozen encoder and deterministic Inter output.
            # Preserve independent autograd branches for callers training a
            # swap objective, including their original accumulation order.
            if torch.is_grad_enabled():
                latent, ab = self._encode(dst), None
            swapped = self._decode(self._features(latent, "swap", ab=ab), dst.shape[-2:])
            result.update(swap=swapped["image"], swap_mask=swapped["mask"])
        return result

    def predict(self, dst: torch.Tensor) -> dict[str, torch.Tensor]:
        """Swap RGB and two probability masks; caller selects no_grad/inference."""
        latent = self._encode(dst)
        ab = self.inter_AB(latent) if not torch.is_grad_enabled() else None
        swap = self._decode(self._features(latent, "swap", ab=ab), dst.shape[-2:])
        # Merger needs DST mask, but not a second expensive TAE RGB decode.
        dst_features = self._features(latent, "dst", ab=ab)
        return {"swap": swap["image"].clamp(0, 1), "swap_mask": swap["mask"],
                "dst_mask": self._mask(dst_features, dst.shape[-2:])}
