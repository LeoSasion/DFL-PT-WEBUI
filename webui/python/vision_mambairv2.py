"""Pinned official MambaIRv2 Large x4 with a validated portable scan adapter."""
import ast
import hashlib
from pathlib import Path
import types
import cv2
import numpy as np
from vision_resource_paths import ROOT, resource_group

FILES = {
    'mambairv2_arch.py': '89123f88ed2bcd343dcb64515453046eeec92321d4ebc727d99bbf7ad7f1051c',
    'mambairv2_classicSR_Large_x4.pth': '9f9090cda07ed4b8498c2db9863c8c263cd300f445017ec77b7fc8babbb8a78d',
}
_architecture = None


def verified_files():
    root = resource_group('mambairv2', ROOT / 'workspace/.vision-models/restoration/mambairv2')
    for name, digest in FILES.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
            raise ValueError('MambaIRv2 requires plain pinned resources: ' + name)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest is None or actual != digest:
            raise ValueError('MambaIRv2 resource SHA mismatch: ' + name)
    return root


def load_architecture(root):
    global _architecture
    if _architecture is not None:
        return _architecture
    # Keep the official source bytes unchanged. Redirect one optional CUDA API
    # import to the verified PyTorch recurrence; no architecture/weights changes.
    tree = ast.parse((root / 'mambairv2_arch.py').read_text(encoding='utf-8'))
    count = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'mamba_ssm.ops.selective_scan_interface':
            if [item.name for item in node.names] != ['selective_scan_fn', 'selective_scan_ref']:
                raise ValueError('Official selective scan import contract changed')
            node.module = 'mamba_scan'; count += 1
    if count != 1:
        raise ValueError('Official architecture needs one verified scan binding')
    module = types.ModuleType('dfl_verified_mambairv2')
    module.__file__ = str(root / 'mambairv2_arch.py')
    exec(compile(ast.fix_missing_locations(tree), module.__file__, 'exec'), module.__dict__)
    _architecture = module
    return module


class MambaRestoration:
    def __init__(self, device='cuda:0'):
        import torch
        from vision_restoration import compatibility_imports
        compatibility_imports()
        self.torch, self.device = torch, torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.root = verified_files()
        module = load_architecture(self.root)
        # Architecture/config exactly as the official Large x4 test YAML.
        self.network = module.MambaIRv2(upscale=4, in_chans=3, img_size=64, img_range=1.,
            embed_dim=174, d_state=16, depths=[6]*9, num_heads=[6]*9, window_size=16,
            inner_rank=64, num_tokens=128, convffn_kernel_size=5, mlp_ratio=2.,
            upsampler='pixelshuffle', resi_connection='1conv')
        weight = torch.load(self.root / 'mambairv2_classicSR_Large_x4.pth', map_location='cpu', weights_only=True)
        state = weight.get('params_ema', weight.get('params', weight))
        self.network.load_state_dict(state, strict=True)
        self.network.requires_grad_(False).eval().to(self.device)
        self.provenance = {'model': 'mambairv2', 'variant': 'official-classicSR-Large-x4', 'sha256': FILES['mambairv2_classicSR_Large_x4.pth'],
            'sourceSha256': FILES['mambairv2_arch.py'], 'precision': 'FP32', 'strict': True,
            'scan': 'pytorch-affine-prefix-FP32; full state across chunks', 'routingSeed': 10,
            'scope': 'classic bicubic SR weights evaluated on actual aligned/predicted faces',
            'newTraining': False, 'automaticDownloads': False, 'tiling': False}

    def restore(self, image, output_size=None):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('RGB uint8 Mamba input required')
        torch = self.torch
        data = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device) / 255.
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        # Official Gumbel routing is stochastic even in eval. Seed locally, keep
        # the original operation, restore other model RNG state after inference.
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(10)
            if devices:
                torch.cuda.manual_seed(10)
            result = self.network(data)[0].permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite Mamba restoration')
        result = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
        if output_size is not None and result.shape[1::-1] != tuple(output_size):
            result = cv2.resize(result, tuple(output_size), interpolation=cv2.INTER_AREA)
        return result
