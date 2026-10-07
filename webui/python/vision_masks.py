"""Frozen optional mask candidates. No training, acquisition or aligned-file writes.

RGB uint8 input; output is a declared binary/semantic mask, not interchangeable
XSeg supervision. Official source and weights are verified before import/load.
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vision_assets import canonical_digest, file_sha256, resolve_asset_path

MODELS = ('xseg-wf', 'bisenet-celebamaskhq', 'sam2.1-hiera-large')
WEIGHT_PINS = {
    'xseg-wf': '26e45677ef3136e0327f0fd51e452cbea81a703ee7fea9121b01ba58da65c385',
    'bisenet-celebamaskhq': '468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567',
    'sam2.1-hiera-large': '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
}
SOURCE_PINS = {
    'xseg-wf': 'e3ad8bc1a5de25cf7dd2d01499f7720d0b07665333eb2aa160e1f67c1da14def',
    'bisenet-celebamaskhq': 'e44578e1cf4df95d8eb6601ba52d64246b312691a366a0f17ed19e5016d42291',
    'sam2.1-hiera-large': '4c50a3a5a0e309404957d1e7fdd647c9556c4d02da1bba95dde7eae95c59f579',
}
LICENSE_PINS = {
    'xseg-wf': '0b383d5a63da644f628d99c33976ea6487ed89aaa59f0b3257992deac1171e6b',
    'bisenet-celebamaskhq': '69cbe123bddc6c22ef171e83226f2bfd1dbc158853074ff0aa398bfce4332aa4',
    'sam2.1-hiera-large': '1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6',
}
SOURCE_ROOTS = {'bisenet-celebamaskhq': 'bisenet-source', 'sam2.1-hiera-large': 'sam2-source'}
LABELS = ('background', 'skin', 'left-brow', 'right-brow', 'left-eye', 'right-eye',
          'eyeglasses', 'left-ear', 'right-ear', 'earring', 'nose', 'mouth',
          'upper-lip', 'lower-lip', 'neck', 'necklace', 'clothing', 'hair', 'hat')
FACE_PARTS = tuple(range(1, 9)) + (10, 11, 12, 13)


def verified_identity(root, model_id):
    root = Path(root).resolve()
    identity = json.loads((root / 'assets.json').read_text(encoding='utf-8'))
    if identity.get('schemaVersion') != 1:
        raise ValueError('Unsupported mask asset registry')
    entries = [entry for entry in identity['models'] if entry['id'] == model_id]
    if len(entries) != 1:
        raise ValueError('Mask asset identity must be unique')
    entry = entries[0]
    if model_id in SOURCE_ROOTS:
        source_root = resolve_asset_path(root, entry.get('sourceRoot'))
        if entry['sourceRoot'] != SOURCE_ROOTS[model_id] or source_root != (root / SOURCE_ROOTS[model_id]).resolve():
            raise ValueError('Mask sourceRoot differs from the pinned source directory')
        registered = {resolve_asset_path(root, item['path']) for item in entry['sourceFiles']}
        actual = {path.resolve() for pattern in ('*.py', '*.yaml') for path in source_root.rglob(pattern)
                  if '.git' not in path.parts}
        if registered != actual:
            raise ValueError('Mask executable/config files differ from the sourceFiles registry')
    if model_id in WEIGHT_PINS and (len(entry['weights']) != 1 or entry['weights'][0]['sha256'] != WEIGHT_PINS[model_id]):
        raise ValueError('Mask weight differs from the reviewed official pin')
    if model_id in SOURCE_PINS and canonical_digest(entry['sourceFiles']) != SOURCE_PINS[model_id]:
        raise ValueError('Mask source differs from the reviewed official revision')
    if model_id in LICENSE_PINS and (len(entry['licenseFiles']) != 1 or entry['licenseFiles'][0]['sha256'] != LICENSE_PINS[model_id]):
        raise ValueError('Mask license differs from the reviewed official source')
    for item in entry['weights'] + entry['sourceFiles'] + entry['licenseFiles']:
        target = resolve_asset_path(root, item['path'])
        if target.is_symlink() or not target.is_file() or file_sha256(target) != item['sha256']:
            raise ValueError('Mask official asset hash mismatch: ' + item['path'])
    return entry


def reject_foreign_sam2_modules(source_root, source_files):
    """An already imported package must belong to this verified source tree."""
    source_root = Path(source_root).resolve()
    registered = {Path(path).resolve() for path in source_files}
    for name, module in tuple(sys.modules.items()):
        if name != 'sam2' and not name.startswith('sam2.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.pyc':
                path = Path(importlib.util.source_from_cache(str(path))).resolve()
            if not path.is_relative_to(source_root) or path not in registered:
                raise ValueError('Already imported SAM2 module is outside the verified source tree')
        else:
            package_paths = list(getattr(module, '__path__', []))
            if not package_paths or any(not Path(path).resolve().is_relative_to(source_root) for path in package_paths):
                raise ValueError('Already imported SAM2 namespace is outside the verified source tree')


def validate_image(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        raise ValueError('RGB uint8 image with dimensions >=32 required')


def fixed_prompt(shape):
    """Locked alignment-relative prompt; no result-dependent prompt tuning."""
    height, width = shape[:2]
    return {'box': [0.16 * width, 0.12 * height, 0.84 * width, 0.90 * height],
            'point': [0.50 * width, 0.48 * height], 'pointLabel': 1,
            'definition': 'fixed aligned-image face-region box plus one positive centre point; not a human annotation'}


def mask_statistics(mask, valid, ground_truth=None):
    mask = np.asarray(mask)
    valid = np.asarray(valid, dtype=bool)
    if mask.shape != valid.shape or mask.ndim != 2 or not np.isfinite(mask).all() or not valid.any():
        raise ValueError('Finite mask and nonempty valid-pixel domain required')
    if not np.isin(mask, [0, 1, False, True]).all():
        raise ValueError('Binary mask required for coverage statistics')
    foreground = mask.astype(bool) & valid
    result = {'validPixels': int(valid.sum()), 'foregroundValidPixels': int(foreground.sum()),
              'foregroundFraction': float(foreground.sum() / valid.sum()), 'iou': None, 'dice': None}
    if ground_truth is not None:
        truth = np.asarray(ground_truth)
        if truth.shape != valid.shape or not np.isin(truth, [0, 1, False, True]).all():
            raise ValueError('Human truth must be a matching binary mask')
        truth = truth.astype(bool) & valid
        intersection = int((truth & foreground).sum())
        union = int((truth | foreground).sum())
        total = int(truth.sum() + foreground.sum())
        result.update(iou=intersection / union if union else 1.0,
                      dice=2 * intersection / total if total else 1.0)
    return result


class MaskCandidate:
    def __init__(self, model_id, assets_root, device='cuda'):
        if model_id not in MODELS:
            raise ValueError('Unsupported mask candidate')
        import torch
        self.torch, self.device, self.model_id = torch, torch.device(device), model_id
        root = Path(assets_root).resolve()
        from vision_resource_paths import ROOT, resource_group
        if root == (ROOT / 'workspace/.vision-models/masks').resolve():
            root = resource_group(model_id, root).resolve()
        self.identity = verified_identity(root, model_id)
        weight = resolve_asset_path(root, self.identity['weights'][0]['path'])
        if model_id == 'xseg-wf':
            project = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project / '_internal' / 'DeepFaceLab'))
            from core.leras import nn
            from core.leras.device import Devices
            from facelib import XSegNet
            Devices.initialize_main_env()
            nn.initialize(nn.DeviceConfig.CPU() if self.device.type == 'cpu' else nn.DeviceConfig.GPUIndexes([self.device.index or 0]))
            self.wrapper = XSegNet(name='XSeg', resolution=256, training=False, weights_file_root=weight.parent,
                                   run_on_cpu=self.device.type == 'cpu', raise_on_no_model_files=True)
            self.network = self.wrapper.model
        elif model_id == 'bisenet-celebamaskhq':
            source = resolve_asset_path(root, self.identity['sourceRoot'])
            # The upstream constructor downloads ImageNet weights even when a
            # complete parsing checkpoint will replace them. Disable ONLY that
            # initialization, then require every checkpoint key/shape strictly.
            prior = sys.modules.get('resnet')
            spec = importlib.util.spec_from_file_location('resnet', source / 'resnet.py')
            resnet = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(resnet)
            resnet.Resnet18.init_weight = lambda self: None
            sys.modules['resnet'] = resnet
            try:
                spec = importlib.util.spec_from_file_location('dfl_official_bisenet', source / 'model.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.network = module.BiSeNet(n_classes=19)
            finally:
                if prior is None:
                    sys.modules.pop('resnet', None)
                else:
                    sys.modules['resnet'] = prior
            self.network.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
        else:
            source = resolve_asset_path(root, self.identity['sourceRoot'])
            registered_sources = [resolve_asset_path(root, item['path']) for item in self.identity['sourceFiles']]
            reject_foreign_sam2_modules(source, registered_sources)
            sys.path.insert(0, str(source))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            reject_foreign_sam2_modules(source, registered_sources)
            self.network = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(weight), device=str(self.device),
                                     mode='eval', apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(self.network, max_hole_area=0, max_sprinkle_area=0)
        self.network.requires_grad_(False).eval().to(self.device)
        if any(parameter.requires_grad for parameter in self.network.parameters()) or self.network.training:
            raise RuntimeError('Mask candidate must be frozen and in eval mode')
        self.provenance = {'identity': self.identity, 'newTraining': False, 'frozen': True,
                           'automaticDownloads': False, 'precision': 'FP32', 'postprocess': False,
                           'parameterCount': sum(parameter.numel() for parameter in self.network.parameters()),
                           'requiresGradParameterCount': 0,
                           'humanGroundTruthAvailable': False, 'winner': None}

    def predict(self, image):
        validate_image(image)
        torch = self.torch
        height, width = image.shape[:2]
        with torch.inference_mode():
            if self.model_id == 'xseg-wf':
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                resized = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
                probability = self.wrapper.extract(resized)[:, :, 0]
                mask = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
                return {'mask': mask, 'semantics': 'generic XSeg WF learned binary foreground; NOT 19-class parsing',
                        'alignmentQualification': 'expects DFL WF; supplied FFHQ five-point crops are a functional cross-alignment probe, not WF quality acceptance',
                        'threshold': 0.5, 'prompt': None}
            if self.model_id == 'bisenet-celebamaskhq':
                resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device) / 255
                mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                labels = self.network((tensor - mean) / std)[0].argmax(1)[0].cpu().numpy().astype(np.uint8)
                labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
                return {'mask': np.isin(labels, FACE_PARTS), 'labels': labels, 'labelNames': LABELS,
                        'semantics': 'union labels 1..8,10..13; excludes background/earring/neck/necklace/clothes/hair/hat; NOT XSeg truth',
                        'selectedClasses': FACE_PARTS, 'prompt': None}
            prompt = fixed_prompt(image.shape)
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(point_coords=np.array([prompt['point']], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32), box=np.array(prompt['box'], dtype=np.float32), multimask_output=False)
            return {'mask': masks[0].astype(bool), 'semantics': 'prompted binary object region; no face-part classes; NOT XSeg-compatible semantics',
                    'prompt': prompt, 'predictedIouConfidence': float(scores[0]), 'confidenceIsMeasuredIou': False,
                    'postprocess': False, 'maskLogitThreshold': 0.0}
