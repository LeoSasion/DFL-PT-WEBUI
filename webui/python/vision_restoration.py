"""Optional frozen restoration inference and quality comparison. Never trains/downloads.

Official assets live in the ignored model cache; H3CE code is not imported.
Input/output arrays use RGB uint8. Generated output has no DFL aligned metadata.
"""
import hashlib
import importlib.util
import math
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np

ASSETS = {
    "swinir-psnr": ("003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth", "450be6eac63a59959b55a83df6743444de9f018547f81d16f699b3680d366ad7"),
    "realesrgan-x4plus": ("RealESRGAN_x4plus.pth", "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"),
    "gfpgan-v1.4": ("GFPGANv1.4.pth", "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"),
    "swinir-source": ("network_swinir.py", "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913"),
    "lpips-calibration": ("alex-v0.1.pth", "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"),
    "lpips-backbone": ("alexnet-owt-7be5be79.pth", "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"),
}
MODELS = ("swinir-psnr", "realesrgan-x4plus", "gfpgan-v1.4")
FFHQ_TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936],
                          [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float64)


def sha256_file(target):
    digest = hashlib.sha256()
    with Path(target).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_asset(root, key):
    filename, digest = ASSETS[key]
    root = Path(root).resolve()
    from vision_resource_paths import ROOT, resource_group
    if root == (ROOT / 'workspace/.vision-models/restoration').resolve():
        group = 'swinir-psnr' if key == 'swinir-source' else key
        root = resource_group(group, root).resolve()
    target = root / filename
    if target.is_symlink() or not target.is_file() or target.resolve().parent != root:
        raise ValueError(f"Missing/plain-file official model asset: {filename}")
    if sha256_file(target) != digest:
        raise ValueError(f"Official model SHA-256 mismatch: {filename}")
    return target


def compatibility_imports():
    """BasicSR 1.4.2's old import name resolves to the same official operation."""
    import torchvision.transforms.functional as official
    legacy = "torchvision.transforms.functional_tensor"
    if legacy not in sys.modules:
        shim = types.ModuleType(legacy)
        shim.rgb_to_grayscale = official.rgb_to_grayscale
        sys.modules[legacy] = shim


def five_point_alignment(source_points):
    points = np.asarray(source_points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Five finite source points are required")
    x, y = points - points.mean(axis=0), FFHQ_TEMPLATE - FFHQ_TEMPLATE.mean(axis=0)
    variance = float(np.mean(np.sum(x * x, axis=1)))
    if variance < 1.0:
        raise ValueError("Degenerate source landmark geometry")
    u, singular, vt = np.linalg.svd(x.T @ y / 5.0)
    correction = np.eye(2)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    linear = scale * rotation
    translation = FFHQ_TEMPLATE.mean(axis=0) - linear @ points.mean(axis=0)
    return np.column_stack((linear, translation))


def aligned_reference(image, source_points):
    matrix = five_point_alignment(source_points)
    reference = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)
    valid = cv2.warpAffine(np.ones(image.shape[:2], dtype=np.uint8), matrix, (512, 512),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(valid, np.ones((11, 11), dtype=np.uint8))
    if float(valid.mean()) < 0.5:
        raise ValueError("Insufficient valid source pixels after alignment")
    return reference, valid.astype(bool), matrix


def synthetic_down4(reference):
    if reference.shape != (512, 512, 3) or reference.dtype != np.uint8:
        raise ValueError("Locked 512 RGB uint8 reference required")
    return cv2.resize(reference, (128, 128), interpolation=cv2.INTER_AREA)


class RestorationModel:
    def __init__(self, model_id, assets_root, device="cuda", tile_size=128, halo=16):
        if model_id not in MODELS:
            raise ValueError("Unsupported restoration model")
        import torch
        self.torch, self.model_id, self.device = torch, model_id, torch.device(device)
        self.tile_size, self.halo = None if tile_size is None else int(tile_size), int(halo)
        if ((self.tile_size is not None and (self.tile_size < 32 or self.tile_size % 8))
                or self.halo < 0 or self.halo % 8):
            raise ValueError("SR tile/halo must be aligned to an 8-pixel window")
        if self.tile_size is None:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        weight = verified_asset(assets_root, model_id)
        if model_id == "swinir-psnr":
            source = verified_asset(assets_root, "swinir-source")
            spec = importlib.util.spec_from_file_location("dfl_verified_official_swinir", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.network = module.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1.0, depths=[6] * 9, embed_dim=240, num_heads=[8] * 9,
                mlp_ratio=2, upsampler="nearest+conv", resi_connection="3conv")
        else:
            compatibility_imports()
            if model_id == "realesrgan-x4plus":
                from basicsr.archs.rrdbnet_arch import RRDBNet
                self.network = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
                # Direct aligned network: no FaceRestoreHelper/detector/parser downloads.
                self.network = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                    decoder_load_path=None, fix_decoder=False, num_mlp=8, input_is_latent=True,
                    different_w=True, narrow=1, sft_half=True)
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self.network.load_state_dict(state, strict=True)
        self.network.requires_grad_(False).eval().float().to(self.device)
        self.provenance = {"model": model_id, "sha256": ASSETS[model_id][1], "strict": True,
                           "precision": "FP32", "newTraining": False, "automaticDownloads": False,
                           "tiling": self.tile_size is not None,
                           "srTile": self.tile_size, "srHalo": self.halo if self.tile_size is not None else 0,
                           "cleanSrPolicy": "native-x4-then-area-downsample-to-512",
                           "gfpganPolicy": "already-aligned-512; RGB [-1,1]; deterministic-noise"}

    def restore(self, image, output_size=(512, 512)):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB uint8 image required")
        torch = self.torch
        with torch.inference_mode():
            if self.model_id == "gfpgan-v1.4":
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
                tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device) / 127.5 - 1.0
                result = (self.network(tensor, return_rgb=False, randomize_noise=False)[0] + 1.0) / 2.0
                output = result[0].permute(1, 2, 0).cpu().numpy()
            else:
                height, width = image.shape[:2]
                pad_y, pad_x = (-height) % 8, (-width) % 8
                padded = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
                tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device) / 255.0
                if self.tile_size is None:
                    output = self.network(tensor)[0].permute(1, 2, 0).cpu().numpy()
                else:
                    output = np.zeros((padded.shape[0] * 4, padded.shape[1] * 4, 3), dtype=np.float32)
                    # Disjoint output cores retain halo context, avoiding averaged overlapping predictions.
                    for y in range(0, padded.shape[0], self.tile_size):
                        for x in range(0, padded.shape[1], self.tile_size):
                            end_y, end_x = min(y + self.tile_size, padded.shape[0]), min(x + self.tile_size, padded.shape[1])
                            top, left = max(0, y - self.halo), max(0, x - self.halo)
                            bottom, right = min(end_y + self.halo, padded.shape[0]), min(end_x + self.halo, padded.shape[1])
                            result = self.network(tensor[:, :, top:bottom, left:right])[0].permute(1, 2, 0).cpu().numpy()
                            output[y*4:end_y*4, x*4:end_x*4] = result[(y-top)*4:(end_y-top)*4, (x-left)*4:(end_x-left)*4]
                output = output[:height*4, :width*4]
            output = np.uint8(np.rint(np.clip(output, 0, 1) * 255))
            if (not isinstance(output_size, (tuple, list)) or len(output_size) != 2
                    or any(not isinstance(value, int) or value < 1 for value in output_size)):
                raise ValueError("Positive output width/height required")
            return cv2.resize(output, tuple(output_size), interpolation=cv2.INTER_AREA) if output.shape[:2] != tuple(output_size)[::-1] else output


class OfflineLpips:
    def __init__(self, assets_root, device="cuda"):
        import torch
        import lpips
        self.torch, self.device = torch, torch.device(device)
        backbone = torch.load(verified_asset(assets_root, "lpips-backbone"), map_location="cpu", weights_only=True)
        calibration = torch.load(verified_asset(assets_root, "lpips-calibration"), map_location="cpu", weights_only=True)
        self.network = lpips.LPIPS(net="alex", pretrained=False, pnet_rand=True, verbose=False)
        full = self.network.state_dict()
        for key in list(full):
            if key.startswith("net.slice"):
                match = re.fullmatch(r"net\.slice\d+\.(\d+)\.(weight|bias)", key)
                if not match:
                    raise ValueError(f"Unexpected AlexNet feature parameter: {key}")
                full[key] = backbone[f"features.{match[1]}.{match[2]}"]
            elif key.startswith("lin"):
                calibration_key = re.sub(r"^lins\.(\d+)\.", lambda m: f"lin{m[1]}.", key)
                full[key] = calibration[calibration_key]
            elif not key.startswith("scaling_layer."):
                raise ValueError(f"Unexpected/random LPIPS parameter: {key}")
        self.network.load_state_dict(full, strict=True)
        self.network.requires_grad_(False).eval().to(self.device)

    def __call__(self, reference, output):
        torch = self.torch
        def tensor(image):
            return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device) / 127.5 - 1.0
        with torch.inference_mode():
            return float(self.network(tensor(reference), tensor(output)).item())


def quality_metrics(reference, output, valid_mask=None, perceptual=None):
    if reference.shape != output.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Metrics require equal-scale RGB arrays")
    valid = np.ones(reference.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != reference.shape[:2] or not valid.any():
        raise ValueError("Nonempty valid-pixel mask required")
    ref, prediction = reference.astype(np.float64) / 255, output.astype(np.float64) / 255
    mse = float(np.mean((ref[valid] - prediction[valid]) ** 2))
    mean_ref, mean_out = cv2.GaussianBlur(ref, (11, 11), 1.5), cv2.GaussianBlur(prediction, (11, 11), 1.5)
    var_ref = np.maximum(cv2.GaussianBlur(ref*ref, (11, 11), 1.5) - mean_ref**2, 0)
    var_out = np.maximum(cv2.GaussianBlur(prediction*prediction, (11, 11), 1.5) - mean_out**2, 0)
    covariance = cv2.GaussianBlur(ref*prediction, (11, 11), 1.5) - mean_ref*mean_out
    structural = ((2*mean_ref*mean_out + 0.01**2)*(2*covariance + 0.03**2)
                  / ((mean_ref**2+mean_out**2+0.01**2)*(var_ref+var_out+0.03**2)))
    interior = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), dtype=np.uint8)).astype(bool)
    interior[:5] = interior[-5:] = False
    interior[:, :5] = interior[:, -5:] = False
    if not interior.any():
        raise ValueError("Insufficient valid SSIM windows")
    masked_output = output.copy()
    masked_output[~valid] = reference[~valid]
    distance = perceptual(reference, masked_output) if perceptual is not None else None
    return {"psnrDb": 10*math.log10(1/mse) if mse > 0 else None, "exactMatch": mse == 0,
            "ssim": float(np.mean(structural[interior])), "lpips": distance,
            "validPixels": int(valid.sum()), "ssimWindowPixels": int(interior.sum())}


def metric_quality_score(metrics):
    if metrics.get("lpips") is None:
        return None
    psnr = 1.0 if metrics["exactMatch"] else min(max(metrics["psnrDb"] / 40.0, 0), 1)
    return 100 * (0.25*psnr + 0.35*np.clip(metrics["ssim"], 0, 1) + 0.40*np.clip(1-metrics["lpips"], 0, 1))
