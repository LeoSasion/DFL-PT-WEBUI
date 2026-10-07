"""Offline 2-D landmark candidates. No detector calls, metadata writes or training.

External research code/weights stay in ignored local evaluation assets. All
coordinates retain their native 68/98 definition; no 98-to-68 conversion occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from vision_assets import canonical_digest, resolve_asset_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / "workspace/.vision-models/landmarks"
MODEL_IDS = ("tufa98", "tufa68", "orformer98", "regression98", "fan68")
LANDMARK_CONTRACTS = {
    "TUFA": {
        "repository": "https://github.com/Jiahao-UTS/TUFA", "revision": "58b3833ebbd481aeaa72a8500d5cc781e4ab8112",
        "source_digest": "af832506c36a09e91fd92b6250b1f6505629abad30367d55cd5295f85e2c06f6",
        "required_sources": ("Prompt.py", "utils/Detector.py", "Prompt/shape_68.npz", "Prompt/shape_98.npz"),
        "weights": [{"file": "weights/model.pth", "sha256": "3fe7071bef814af320c6e6ec7d724b4b1bb253677500e1a48a5e2a66aaf10899", "bytes": 143969647}],
        "license_path": "source/LICENSE", "license_sha256": "189b1af95d661151e054cea10c91b3d754e4de4d3fecfb074c1fb29476f7167b",
    },
    "ORFormer": {
        "repository": "https://github.com/ben0919/ORFormer", "revision": "7e77569783b677f00a71f0caa45d8663d6113167",
        "source_digest": "9434f35f44cbada609dc7f5522c01ab78cca90f5393360501d561fa3a6b4c9e9",
        "required_sources": ("Config/default.py", "Model/StackedHGNet.py", "Model/VQVAE.py", "Model/simple_vit.py", "utils/get_transforms.py"),
        "weights": [{"file": "weights/hgnet.pth", "sha256": "77b1646c825abbe516b7f3cb6484c404c16cc9910187358f5681b06f77dee585", "bytes": 69789933},
                    {"file": "weights/orformer.pth", "sha256": "6229cea05411f0c1fe48e1cd1c1cdb1053a03b66a900d7baaebb0d2bffb59bfd", "bytes": 19149782}],
        "license_path": "source/README.md", "license_sha256": "f19b3ae8abf3a834422cf33af8a252c876f0492741bda0ac92aabb2db9bc4b09",
    },
    "regression": {
        "repository": "https://github.com/ca-joe-yang/regression-without-softarg", "revision": "b391d21aae976fdeecbc22285aae8702faecd00e",
        "source_digest": "0ad94244d526a365438eadf7cf85591cd45696af65e038a13acfaa0ca260a9db",
        "required_sources": ("models/__init__.py", "models/hgnet.py", "data/dataset.py"),
        "weights": [{"file": "weights/wflw.pkl", "sha256": "066d150329f57bdba780ae8915a7ec7c158436c8affaba6dfcd98603cab84b1e", "bytes": 178212388}],
        "license_path": "source/README.md", "license_sha256": "7cc960925add2229340f218b9617ab15ebb6d6489c43062ffc57663ae0d8c9d6",
    },
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_box(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("box_xyxy must contain four finite coordinates with positive width/height")
    return box


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (2, 3):
        raise ValueError("Expected Nx2 points and 2x3 affine matrix")
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError("Coordinates and matrix must be finite")
    return points @ matrix[:, :2].T + matrix[:, 2]


def square_affine(box, expansion=1.0, output_center=128.0):
    box = validate_box(box)
    center = (box[:2] + box[2:]) / 2
    side = np.max(box[2:] - box[:2]) * expansion
    scale = 256.0 / side
    return np.array([[scale, 0, output_center - center[0] * scale],
                     [0, scale, output_center - center[1] * scale]], dtype=np.float64)


def regression_denormalize(points):
    """Official dataset point inverse: align_corners=False, [-.5,255.5].

    Crop-matrix centering is a separate convention from landmark normalization.
    """
    return ((points + 1) * 256 - 1) / 2


def point_definition(count):
    if count == 98:
        regions = {"image_left_eye": list(range(60, 68)), "image_right_eye": list(range(68, 76)), "inner_mouth": list(range(88, 96))}
        name = "WFLW98"
    elif count == 68:
        regions = {"image_left_eye": list(range(36, 42)), "image_right_eye": list(range(42, 48)), "inner_mouth": list(range(60, 68))}
        name = "IBUG68"
    else:
        raise ValueError("No implicit topology conversion is available")
    return {"name": name, "count": count, "dimensions": 2, "index_base": 0, "regions": regions,
            "left_right_convention": "image-space, not subject anatomical left/right", "visibility": "not-estimated", "three_dimensional": False}


@contextmanager
def isolated_source(path):
    """Keep unrelated upstream packages named utils/Config/models apart."""
    roots = {"Backbone", "Prompt", "Config", "Model", "utils", "models", "configs", "data"}
    saved = {name: value for name, value in sys.modules.items() if name.split(".")[0] in roots}
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path.remove(str(path))


def source_module(path):
    spec = importlib.util.spec_from_file_location("_vision_geometry_" + file_sha256(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_identity(directory, expected_model=None):
    directory = Path(directory)
    identity = json.loads(resolve_asset_path(directory, "identity.json").read_text(encoding="utf-8"))
    family = expected_model or identity.get("model")
    if family not in LANDMARK_CONTRACTS or identity.get("model") != family:
        raise ValueError("Landmark manifest model differs from fixed adapter")
    contract = LANDMARK_CONTRACTS[family]
    if any(identity.get(field) != contract[field] for field in ("repository", "revision")):
        raise ValueError("Landmark source differs from fixed repository/revision")
    if identity.get("weights") != contract["weights"]:
        raise ValueError("Landmark weights differ from fixed model identity")
    if not set(contract["required_sources"]) <= set(identity.get("source_files", {})):
        raise ValueError("Required landmark execution sources are not registered")
    if canonical_digest({"source_files": identity["source_files"], "license": identity.get("license")}) != contract["source_digest"]:
        raise ValueError("Official source identity fingerprint mismatch")
    for item in identity["weights"]:
        weight = resolve_asset_path(directory, item["file"])
        if not weight.is_file() or weight.stat().st_size != item["bytes"] or file_sha256(weight) != item["sha256"]:
            raise ValueError("Weight identity mismatch")
    for relative, expected in identity["source_files"].items():
        p = resolve_asset_path(resolve_asset_path(directory, "source"), relative)
        if not p.is_file() or file_sha256(p) != expected:
            raise ValueError("Official source identity mismatch")
    if identity.get("source_license_record") != contract["license_path"] or file_sha256(resolve_asset_path(directory, contract["license_path"])) != contract["license_sha256"]:
        raise ValueError("Official license record differs from fixed identity")
    return {key: identity[key] for key in ("model", "repository", "revision", "license", "weights", "official_weight_url", "distribution")}


class LandmarkPredictor:
    def __init__(self, model_id, assets_root=None, device="cuda"):
        if model_id not in MODEL_IDS:
            raise ValueError("Unknown landmark model")
        self.model_id = model_id
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.count = 68 if model_id.endswith("68") else 98
        self.reference = None
        if model_id == "fan68":
            with isolated_source(ROOT / "_internal/DeepFaceLab"):
                from facelib import FANExtractor
                self.model = FANExtractor(landmarks_3D=False, place_model_on_cpu=self.device.type == "cpu")
            # The existing compatibility adapter selects this project-local asset.
            weight = ROOT / "_internal/DeepFaceLab/DeepFaceLab-master/facelib/2DFAN.npy"
            if not weight.exists():
                weight = ROOT / "_internal/DeepFaceLab/facelib/2DFAN.npy"
            self.identity = {"model": "FAN68", "repository": "project-existing-FAN-PyTorch-port", "revision": None,
                             "license": "existing project asset; see release records", "weights": [{"file": "2DFAN.npy", "sha256": file_sha256(weight)}],
                             "distribution": "existing compatibility baseline"}
            self.preprocessing = "existing FAN crop, no second pass or multi-sample"
            return
        family = "TUFA" if model_id.startswith("tufa") else "ORFormer" if model_id == "orformer98" else "regression"
        from vision_resource_paths import resource_group
        directory = resource_group('tufa', DEFAULT_ASSETS / family) if assets_root is None and family == 'TUFA' else resolve_asset_path(assets_root or DEFAULT_ASSETS, family)
        self.directory = directory
        self.identity = verified_identity(directory, expected_model=family)
        source = directory / "source"
        with isolated_source(source):
            if family == "TUFA":
                from Prompt import Face_prompt
                self.model = Face_prompt("base")
                self._load(self.model, directory / "weights/model.pth")
                self.prompt = torch.from_numpy(np.load(source / f"Prompt/shape_{self.count}.npz")["offset"] / 256).float().to(self.device)[None]
                self.identity["prompt"] = {"file": f"Prompt/shape_{self.count}.npz", "sha256": file_sha256(source / f"Prompt/shape_{self.count}.npz")}
                self.geometry = source_module(source / "utils/Detector.py")
                self.preprocessing = "official Camera crop 1.15; RGB ImageNet normalization; prompt native topology"
            elif family == "ORFormer":
                from Config import cfg
                from Model.StackedHGNet import IntergrationStackedHGNet
                from Model.VQVAE import VQVAE
                from Model.simple_vit import ORFormer
                self.model = IntergrationStackedHGNet(classes_num=[98, 15, 98], edge_info=cfg.WFLW.EDGE_INFO, nstack=4)
                self._load(self.model, directory / "weights/hgnet.pth")
                self.reference = VQVAE(h_dim=128, res_h_dim=32, output_dim=15, n_res_layers=2, n_embeddings=2048,
                                       embedding_dim=256, code_dim=256, beta=.25,
                                       vit=ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256))
                self._load(self.reference, directory / "weights/orformer.pth")
                self.geometry = source_module(source / "utils/get_transforms.py")
                class NumpyAliases:
                    def __getattr__(self, name):
                        return float if name == "float" else getattr(np, name)
                self.geometry.np = NumpyAliases()
                self.preprocessing = "official crop expansion 1.20; RGB ImageNet normalization; 64px heatmap reference"
            else:
                from models import StackedHGNetV1
                # Read only class metadata; avoid importing training/data dependencies.
                edges = [(False, tuple(range(33))), (True, tuple(range(33, 42))), (True, tuple(range(42, 51))),
                         (False, tuple(range(51, 55))), (False, tuple(range(55, 60))), (True, tuple(range(60, 68))),
                         (True, tuple(range(68, 76))), (True, tuple(range(76, 88))), (True, tuple(range(88, 96)))]
                self.model = StackedHGNetV1(image_size=(256, 256), num_classes=[98, 9, 98], edge_info=edges, activation="none")
                self._load(self.model, directory / "weights/wflw.pkl", checkpoint=True)
                self.preprocessing = "official test BGR [-1,1]; official GetCropMatrix geometry, scale from shared bbox max side / 200 (no WFLW GT metadata)"

    def _load(self, model, path, checkpoint=False):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint:
            state = {key.removeprefix("module."): value for key, value in state["net"].items() if key != "n_averaged"}
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False).to(self.device)

    def _tensor(self, rgb, imagenet=True):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(self.device) / 255
        if imagenet:
            x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]
        else:
            x = x * 2 - 1
        return x[None]

    def predict(self, image, box_xyxy):
        if isinstance(image, (str, Path)):
            from PIL import Image
            rgb = np.array(Image.open(image).convert("RGB"))
        else:
            rgb = np.asarray(image)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Input must be RGB uint8 HxWx3")
        box = validate_box(box_xyxy)
        if self.model_id == "fan68":
            original = self.model.extract(rgb, [box], is_bgr=False, multi_sample=False)[0]
            if original is None:
                raise RuntimeError("Existing FAN compatibility inference failed")
            center = (box[:2] + box[2:]) / 2
            scale = (box[2] - box[0] + box[3] - box[1]) / 195
            # Actual legacy crop rounds its endpoints before cv2.resize. Record
            # that pixel-center crop affine, not its analytic heatmap decoder.
            ul = self.model.transform([1, 1], center, scale, 256).astype(np.int32)
            br = self.model.transform([256, 256], center, scale, 256).astype(np.int32)
            factor = 256 / (br - ul)
            trans = np.array([[factor[0], 0, (-ul[0] + .5)*factor[0] - .5],
                              [0, factor[1], (-ul[1] + .5)*factor[1] - .5]])
            points = transform_points(original, trans)
        else:
            if self.model_id.startswith("tufa"):
                crop, trans = self.geometry.crop_img(rgb[:, :, ::-1].copy(), box, torch.from_numpy)
                crop = crop[0].numpy()
            elif self.model_id == "orformer98":
                trans = self.geometry.get_transforms(np.r_[box[:2], box[2:] - box[:2]], 1.20, 0., 256)
                crop = cv2.warpAffine(rgb, trans, (256, 256), flags=cv2.INTER_LINEAR)
            else:
                trans = square_affine(box, output_center=127.5)
                crop = cv2.warpPerspective(rgb, np.vstack([trans, [0, 0, 1]]), (256, 256), flags=cv2.INTER_LINEAR)
            with torch.inference_mode():
                tensor_crop = crop[:, :, ::-1] if self.model_id == "regression98" else crop
                x = self._tensor(tensor_crop, imagenet=self.model_id != "regression98")
                if self.model_id.startswith("tufa"):
                    points = (self.model(x, self.prompt)[0, -1] * 256).cpu().numpy()
                elif self.model_id == "orformer98":
                    small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
                    heatmaps = self.reference(self._tensor(small))[1]
                    _, output = self.model(x, reference_heatmaps=heatmaps)
                    points = ((output[0] + 1) / 2 * 63 * 4).cpu().numpy()
                else:
                    output, _, _, _ = self.model(x)
                    points = regression_denormalize(output[9][0]).cpu().numpy()
            original = transform_points(points, cv2.invertAffineTransform(trans))
        if np.asarray(original).shape != (self.count, 2) or not np.isfinite(original).all():
            raise ValueError("Model returned invalid native landmark points")
        inverse = cv2.invertAffineTransform(np.asarray(trans, dtype=np.float64))
        return {"schema_version": 1, "model_id": self.model_id, "points_original": np.asarray(original).tolist(),
                "points_model": np.asarray(points).tolist(), "point_definition": point_definition(self.count),
                "box_xyxy": box.tolist(), "image_size_wh": [rgb.shape[1], rgb.shape[0]],
                "source_to_model_affine": np.asarray(trans).tolist(), "model_to_source_affine": inverse.tolist(),
                "asset_identity": self.identity, "preprocessing": self.preprocessing, "flip_tta": False,
                "precision": "float32; TF32 disabled; eval/inference_mode", "compatible_68": self.count == 68,
                "training_mask_approved": False, "visibility": "not-estimated"}


def predict_landmarks(image, box_xyxy, model_id="tufa98", assets_root=None, device="cuda"):
    return LandmarkPredictor(model_id, assets_root, device).predict(image, box_xyxy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict_landmarks(args.image, args.box, args.model, args.assets, args.device)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
