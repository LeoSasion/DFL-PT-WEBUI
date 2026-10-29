"""Pinned TAEQI2.1 RGBA codec with a permanently frozen encoder."""
from pathlib import Path
import os
import torch
from torch import nn
from safetensors.torch import load_file
from .checkpoint import file_sha256
from ._vendor import taesd
from me_backend.tensor_checks import finite_unit_interval

WEIGHTS_SHA256 = '6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e'
VENDOR_SHA256 = 'be3a8485c187d35d7fcf5c596fa800171f7b157c43f1a086e00e25134c7e6abd'
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'pretrained' / 'taeqi2_1.safetensors'
DECODER_MODES = ('frozen', 'tail', 'full')
TAIL_START = 12


class TAEQI21Codec(nn.Module):
    latent_channels = 64
    spatial_compression = 16

    def __init__(self, weights=None, device='cpu', decoder_mode='frozen'):
        super().__init__()
        path = Path(weights or os.environ.get('TAE_VAE_WEIGHTS') or DEFAULT_WEIGHTS).resolve()
        if file_sha256(taesd.__file__) != VENDOR_SHA256:
            raise ValueError('TAEQI2.1 vendored source SHA256 mismatch')
        if file_sha256(path) != WEIGHTS_SHA256:
            raise ValueError('TAEQI2.1 weight SHA256 mismatch')
        model = taesd.TAESD(encoder_path=None, decoder_path=None, latent_channels=64,
                            arch_variant='f16', image_channels=4)
        state = {}
        for key, value in load_file(str(path)).items():
            branch, layers, index, *suffix = key.split('.')
            if branch not in ('encoder', 'decoder') or layers != 'layers':
                raise ValueError('Unexpected TAEQI2.1 asset key')
            index = int(index) + (branch == 'decoder')
            state['.'.join([branch, str(index), *suffix])] = value
        model.load_state_dict(state, strict=True)
        self.encoder, self.decoder = model.encoder, model.decoder
        self.to(device=device, dtype=torch.float32)
        self.weight_sha256, self.weights_path = WEIGHTS_SHA256, str(path)
        self.source_sha256 = VENDOR_SHA256
        self.compute_mode = 'fp32-master-bf16-autocast' if torch.device(device).type == 'cuda' else 'fp32'
        self.set_decoder_mode(decoder_mode)
        self.eval()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def decoder_parameter_enabled(self, name):
        return self.decoder_mode == 'full' or (self.decoder_mode == 'tail' and int(name.split('.')[0]) >= TAIL_START)

    def set_decoder_mode(self, mode):
        if mode not in DECODER_MODES:
            raise ValueError('decoder_mode must be frozen, tail or full')
        self.decoder_mode = mode
        self.encoder.requires_grad_(False).eval()
        for p in self.encoder.parameters():
            p.grad = None
        for name, p in self.decoder.named_parameters():
            p.requires_grad_(self.decoder_parameter_enabled(name))
            p.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(bool(mode and getattr(self, 'decoder_mode', 'frozen') != 'frozen'))
        self.encoder.eval()
        for index, layer in enumerate(self.decoder):
            layer.train(bool(mode and (self.decoder_mode == 'full' or
                                      (self.decoder_mode == 'tail' and index >= TAIL_START))))
        return self

    def assert_frozen(self):
        if self.encoder.training or any(p.requires_grad or p.grad is not None for p in self.encoder.parameters()):
            raise RuntimeError('TAEQI2.1 encoder freeze contract violated')
        for name, p in self.decoder.named_parameters():
            if p.requires_grad != self.decoder_parameter_enabled(name) or (not p.requires_grad and p.grad is not None):
                raise RuntimeError('TAEQI2.1 decoder training scope violated')

    @torch.no_grad()
    def encode(self, rgb):
        self.assert_frozen()
        if rgb.ndim != 4 or rgb.shape[0] < 1 or rgb.shape[1] != 3 or any(s < 16 or s % 16 for s in rgb.shape[-2:]):
            raise ValueError('TAEQI2.1 encode requires RGB NCHW with sides divisible by 16')
        if rgb.device != self.device or not rgb.is_floating_point() or not bool(finite_unit_interval(rgb)):
            raise ValueError('RGB must be finite [0,1] floats on the codec device')
        return self.encoder(torch.cat((rgb, torch.ones_like(rgb[:, :1])), dim=1)).detach()

    def decode(self, latents, *, clamp=False):
        self.assert_frozen()
        if latents.ndim != 4 or latents.shape[0] < 1 or latents.shape[1] != 64 or min(latents.shape[-2:]) < 1:
            raise ValueError('TAEQI2.1 decode requires 64-channel NCHW latents')
        if latents.device != self.device or not latents.is_floating_point() or not torch.isfinite(latents).all():
            raise ValueError('Latents must be finite floats on the codec device')
        # Ordinary autograd crosses frozen decoder layers. Alpha is not a face mask.
        rgb = self.decoder(latents)[:, :3].float()
        if not torch.isfinite(rgb).all():
            raise FloatingPointError('Nonfinite TAEQI2.1 decoder output')
        return rgb.clamp(0, 1) if clamp else rgb

    def contract(self):
        return {'weights_sha256': self.weight_sha256, 'source_sha256': self.source_sha256,
                'asset': 'madebyollin/taeqi2_1', 'compute_mode': self.compute_mode,
                'encoder_frozen': True, 'latent_channels': 64, 'spatial_compression': 16,
                'encode': 'RGB [0,1] plus opaque alpha; native continuous latents',
                'decode_training': 'native tanh clamp; unclamped RGB output; independent face mask',
                'decoder_tail_start': TAIL_START}

