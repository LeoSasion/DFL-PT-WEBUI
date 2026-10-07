"""Verified aligned/predicted-face restoration with the legacy input contract."""
import numpy as np

MODELS = ('mambairv2', 'realesrgan-x4plus')


def validate_enhancement_assets(model_id):
    if model_id not in MODELS:
        raise ValueError('Choose MambaIRv2 or Real-ESRGAN; old FaceEnhancer is retired')
    from .LandmarkCandidates import local_module, ROOT
    if model_id == 'mambairv2':
        module = local_module('vision_mambairv2')
        root = module.verified_files()
        import einops
        return {'model': model_id, 'assets': str(root)}
    module = local_module('vision_restoration')
    root = ROOT / 'workspace/.vision-models/restoration'
    weight = module.verified_asset(root, model_id)
    module.compatibility_imports()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    return {'model': model_id, 'weight': str(weight)}


class AlignedFaceEnhancer:
    def __init__(self, model_id='mambairv2', device='cpu'):
        validate_enhancement_assets(model_id)
        self.model_id = model_id
        from .LandmarkCandidates import local_module, ROOT
        if model_id == 'mambairv2':
            self.model = local_module('vision_mambairv2').MambaRestoration(device=device)
        else:
            module = local_module('vision_restoration')
            self.model = module.RestorationModel(model_id, ROOT / 'workspace/.vision-models/restoration', device=device, tile_size=None)
        self.provenance = {**self.model.provenance, 'inputDomain': 'BGR float32 [0,1]; explicit RGB uint8 adapter', 'alignedInput': True}

    def enhance(self, inp_img, is_tanh=False, preserve_size=True):
        data = np.asarray(inp_img)
        if data.ndim != 3 or data.shape[2] != 3 or data.dtype != np.float32 or not np.isfinite(data).all():
            raise ValueError('Finite HWC BGR float32 face required')
        lo, hi = (-1., 1.) if is_tanh else (0., 1.)
        if data.min() < lo - 1e-5 or data.max() > hi + 1e-5:
            raise ValueError('Face restoration input domain mismatch')
        unit = (data + 1.) / 2. if is_tanh else data
        rgb = np.rint(np.clip(unit[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
        h, w = data.shape[:2]
        size = (w, h) if preserve_size else (w * 4, h * 4)
        output = np.ascontiguousarray(self.model.restore(rgb, output_size=size)[:, :, ::-1], dtype=np.float32) / 255.
        if is_tanh:
            output = output * 2. - 1.
        if output.shape != (data.shape if preserve_size else (data.shape[0] * 4, data.shape[1] * 4, 3)) or not np.isfinite(output).all():
            raise RuntimeError('Restoration output violated shape/finite contract')
        return np.clip(output, lo, hi).astype(np.float32)
