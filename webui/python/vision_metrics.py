"""Frozen optional perceptual metrics; raw distances, never quality rankings.

LPIPS Alex/v0.1 is the existing reference. LPIPS VGG/v0.1 and official DISTS
share verified ImageNet VGG16 convolution parameters. No implicit downloads.
"""
import importlib.util
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vision_restoration import OfflineLpips, sha256_file

METRIC_ASSETS = {
    "vgg-backbone": ("vgg16-397923af.pth", "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"),
    "vgg-calibration": ("vgg-v0.1.pth", "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"),
    "dists-source": ("DISTS_pt.py", "69b63ebd4e1aece0f79a3b922ba803b5f6fec127babee94869e9eceaa42b4612"),
    "dists-calibration": ("weights.pt", "f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218"),
}
METRIC_IDS = ("lpips-alex-v0.1", "lpips-vgg-v0.1", "dists")


def metric_asset(root, key):
    filename, expected = METRIC_ASSETS[key]
    root = Path(root).resolve()
    target = root / filename
    if target.is_symlink() or not target.is_file() or target.resolve().parent != root:
        raise ValueError(f"Missing/plain-file official metric asset: {filename}")
    if sha256_file(target) != expected:
        raise ValueError(f"Official metric SHA-256 mismatch: {filename}")
    return target


def prepare_metric_pair(reference, output, valid_mask=None):
    """Same-scale RGB tensors; invalid source pixels are identical in each pair."""
    if (reference.dtype != np.uint8 or output.dtype != np.uint8 or reference.shape != output.shape
            or reference.ndim != 3 or reference.shape[2] != 3 or min(reference.shape[:2]) < 32):
        raise ValueError("Equal-size RGB uint8 images, at least 32 pixels per side, required")
    prediction = output.copy()
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != reference.shape[:2] or not valid.any():
            raise ValueError("Nonempty matching valid-pixel mask required")
        prediction[~valid] = reference[~valid]
    return reference, prediction


class PerceptualMetricSuite:
    def __init__(self, assets_root, device="cuda"):
        import torch
        import torchvision.models as models
        import lpips
        self.torch, self.device = torch, torch.device(device)
        self.alex = OfflineLpips(assets_root, device=device)
        weight = metric_asset(assets_root, "vgg-backbone")
        state = torch.load(weight, map_location="cpu", weights_only=True)
        # Build architecture only, then strict-load all published parameters.
        backbone = models.vgg16(weights=None)
        backbone.load_state_dict(state, strict=True)
        self.shared_vgg_features = backbone.features
        self.shared_vgg_features.requires_grad_(False).eval().to(self.device)
        del backbone, state

        self.vgg = lpips.LPIPS(net="vgg", pretrained=False, pnet_rand=True, verbose=False)
        # Both candidates reference these same convolution objects, not merely
        # different random networks with similar shapes. Pooling differs by the
        # official DISTS definition and is retained unchanged.
        for _, slice_module in self.vgg.net.named_children():
            for index in list(slice_module._modules):
                slice_module._modules[index] = self.shared_vgg_features[int(index)]
        calibration = torch.load(metric_asset(assets_root, "vgg-calibration"), map_location="cpu", weights_only=True)
        full_vgg = self.vgg.state_dict()
        for key in full_vgg:
            if key.startswith("lin"):
                calibration_key = re.sub(r"^lins\.(\d+)\.", lambda match: f"lin{match[1]}.", key)
                full_vgg[key] = calibration[calibration_key]
            elif not (key.startswith("net.slice") or key.startswith("scaling_layer.")):
                raise ValueError(f"Unexpected/unloaded LPIPS VGG parameter: {key}")
        self.vgg.load_state_dict(full_vgg, strict=True)
        self.vgg.requires_grad_(False).eval().to(self.device)

        source = metric_asset(assets_root, "dists-source")
        spec = importlib.util.spec_from_file_location("dfl_verified_official_dists", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Replace the factory binding in this verified private module only.
        # Official pooling, feature layers and DISTS forward remain untouched.
        module.models = SimpleNamespace(vgg16=lambda **_options: SimpleNamespace(features=self.shared_vgg_features))
        self.dists = module.DISTS(load_weights=False)
        calibration = torch.load(metric_asset(assets_root, "dists-calibration"), map_location="cpu", weights_only=True)
        if set(calibration) != {"alpha", "beta"}:
            raise ValueError("Official DISTS alpha/beta calibration required")
        full_dists = self.dists.state_dict()
        full_dists.update(calibration)
        self.dists.load_state_dict(full_dists, strict=True)
        self.dists.requires_grad_(False).eval().to(self.device)
        self.provenance = {"metrics": list(METRIC_IDS), "sharedVggBackboneSha256": METRIC_ASSETS["vgg-backbone"][1],
                           "lpipsVggCalibrationSha256": METRIC_ASSETS["vgg-calibration"][1],
                           "distsSourceSha256": METRIC_ASSETS["dists-source"][1],
                           "distsCalibrationSha256": METRIC_ASSETS["dists-calibration"][1],
                           "strict": True, "sharedConvolutionObjects": True, "frozen": True,
                           "automaticDownloads": False, "training": False,
                           "distsInputPolicy": "same-scale RGB [0,1], no official CLI 256 resize",
                           "lpipsInputPolicy": "same-scale RGB [-1,1], v0.1 scaling",
                           "humanPreferenceCorrelation": None, "metricQualityRanking": None}

    def evaluate(self, reference, output, valid_mask=None):
        reference, prediction = prepare_metric_pair(reference, output, valid_mask)
        torch = self.torch
        def tensor(image):
            return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device) / 255
        with torch.inference_mode():
            left, right = tensor(reference), tensor(prediction)
            results = {"lpips-alex-v0.1": self.alex(reference, prediction),
                       "lpips-vgg-v0.1": float(self.vgg(left * 2 - 1, right * 2 - 1).item()),
                       "dists": float(self.dists(left, right, require_grad=False).item())}
        if not all(math.isfinite(value) for value in results.values()):
            raise ValueError("Perceptual metric returned a nonfinite value")
        return results
