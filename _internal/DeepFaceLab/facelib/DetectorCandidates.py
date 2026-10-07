"""Opt-in hash-verified WebUI detector adapters for the DFL rects API."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

CANDIDATE_IDS = ("yolo11m-face", "yolo12l-face", "yolo26s-face")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEBUI_PYTHON = PROJECT_ROOT / "webui/python"


def _local_module(name):
    location = str(WEBUI_PYTHON)
    if location not in sys.path:
        sys.path.insert(0, location)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != WEBUI_PYTHON / f"{name}.py":
        raise ValueError(f"Unexpected detector adapter module: {name}")
    return module


def validate_candidate_assets(model_id, assets_root=None):
    """Fail before extraction modifies output if the optional runtime is absent."""
    if model_id not in CANDIDATE_IDS:
        raise ValueError("Unknown optional face detector")
    assets = _local_module("vision_assets")
    try:
        path, entry = assets.verified_asset(model_id, assets_root=assets_root)
        catalog = json.loads((PROJECT_ROOT / "tools/vision-model-candidates.json").read_text(encoding="utf-8"))
        expected = [asset for asset in catalog["detectorAssets"] if asset.get("id") == model_id]
        if catalog.get("schemaVersion") != 1 or len(expected) != 1:
            raise ValueError("Unsupported fixed detector catalog")
        for field in ("sha256", "sizeBytes", "runtime", "task", "scale", "architecture", "filename"):
            if entry.get(field) != expected[0].get(field):
                raise ValueError(f"Local detector record differs from fixed catalog: {field}")
        version = importlib.metadata.version("ultralytics")
        if entry["runtime"] != "ultralytics==" + version:
            raise ValueError("Installed detector runtime differs from the asset record")
        importlib.metadata.version("torchvision")
        # Metadata alone does not catch a broken Torch/torchvision native ABI.
        importlib.import_module("torchvision")
    except Exception as error:
        raise ValueError(f"Optional detector {model_id} unavailable: {error}. "
                         "Prepare its verified assets and project-local optional dependencies before retrying") from error
    return path, entry


class CandidateDetector:
    def __init__(self, model_id, *, device="cpu", assets_root=None):
        validate_candidate_assets(model_id, assets_root)
        source = _local_module("vision_detectors")
        options = {"device": device}
        if assets_root is not None:
            options["assets_root"] = assets_root
        # The shared adapter prohibits implicit downloads and checks task/scale.
        self.detector = source.FaceDetector(model_id, **options)
        self.identity = self.detector.identity

    def extract(self, image, is_bgr=True, is_remove_intersects=False):
        # Preserve the rects API's largest-face-first limit and optional overlap
        # suppression; the shared prediction record retains confidence scores.
        rects = self.detector.extract(image, is_bgr=is_bgr)
        rects.sort(key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]), reverse=True)
        if is_remove_intersects:
            for index in range(len(rects) - 1, 0, -1):
                left, top, right, bottom = rects[index]
                l0, t0, r0, b0 = rects[index - 1]
                if min(right, r0) >= max(left, l0) and min(bottom, b0) >= max(top, t0):
                    rects.pop(index)
        return rects
