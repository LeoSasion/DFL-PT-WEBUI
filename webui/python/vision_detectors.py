"""Hash-verified PyTorch detection candidates; no automatic default promotion."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from vision_assets import DEFAULT_ASSETS_ROOT, PROJECT_ROOT, canonical_digest, verified_asset

DETECTOR_IDS = ("yolo11m-face", "yolo12l-face", "yolo26s-face")


class FaceDetector:
    def __init__(self, model_id, *, assets_root=None, device="cpu", image_size=960,
                 confidence=0.25, nms_iou=0.5):
        if model_id not in DETECTOR_IDS:
            raise ValueError("Unknown face detector candidate")
        if not (isinstance(image_size, int) and 128 <= image_size <= 4096 and image_size % 32 == 0):
            raise ValueError("Detector image size must be a multiple of 32 between 128 and 4096")
        if not 0 < confidence < 1 or not 0 < nms_iou < 1:
            raise ValueError("Invalid detection thresholds")
        path, entry = verified_asset(model_id, assets_root=assets_root)
        catalog = json.loads((PROJECT_ROOT / "tools/vision-model-candidates.json").read_text(encoding="utf-8"))
        expected = [item for item in catalog["detectorAssets"] if item["id"] == model_id]
        if catalog.get("schemaVersion") != 1 or len(expected) != 1 or any(
            entry.get(key) != expected[0].get(key)
            for key in ("sha256", "sizeBytes", "architecture", "task", "family", "scale", "runtime", "filename")
        ):
            raise ValueError("Detector record differs from the reviewed official model contract")
        version = importlib.metadata.version("ultralytics")
        if entry["runtime"] != "ultralytics==" + version:
            raise ValueError("Candidate runtime version differs from its asset record")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        config_root = Path(assets_root).resolve() / "runtime" if assets_root is not None else PROJECT_ROOT / '.runtime/vision/yolo'
        config_root.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_root)
        from ultralytics import YOLO
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        import ultralytics.utils as utilities
        if utilities.AUTOINSTALL or utilities.ONLINE:
            raise ValueError("Restart the runtime with offline model loading enabled")
        self.model = YOLO(str(path), task=entry["task"])
        names = self.model.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        if self.model.task != entry["task"] or names != {0: "face"}:
            raise ValueError("The acquired model does not implement one face detection class")
        scale = self.model.model.yaml.get("scale")
        if scale != entry["scale"]:
            raise ValueError("The acquired model scale differs from the registry")
        self.model.model.float().eval().requires_grad_(False)
        self.device, self.entry = device, entry
        self.settings = {"imageSize": image_size, "confidence": float(confidence),
                         "nmsIou": float(nms_iou), "precision": "fp32", "TF32": False, "augment": False,
                         "preprocessing": "ultralytics-rgb-to-bgr-letterbox-v1"}
        self.identity = {"id": model_id, "sha256": entry["sha256"], "runtime": entry["runtime"],
                         "architecture": entry["architecture"], "task": entry["task"], "source": entry["source"],
                         "settings": self.settings, "settingsDigest": canonical_digest(self.settings)}

    def predict(self, image_rgb):
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Face detection requires a nonempty RGB uint8 HWC image")
        import torch
        height, width = image.shape[:2]
        with torch.inference_mode():
            results = self.model.predict(source=np.ascontiguousarray(image[:, :, ::-1]),
                imgsz=self.settings["imageSize"], conf=self.settings["confidence"],
                iou=self.settings["nmsIou"], device=self.device, quantize=None, augment=False,
                classes=[0], verbose=False, save=False, stream=False)
        if len(results) != 1 or tuple(results[0].orig_shape) != (height, width):
            raise ValueError("Detector changed source geometry or result count")
        rows = results[0].boxes.data.detach().cpu().numpy()
        if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all()
                or np.any(rows[:, 5] != 0) or np.any((rows[:, 4] < 0) | (rows[:, 4] > 1))):
            raise ValueError("Invalid face detector output")
        if len(rows) and (np.any(rows[:, 2] <= rows[:, 0]) or np.any(rows[:, 3] <= rows[:, 1])
                or np.any(rows[:, :4] < 0) or np.any(rows[:, [0, 2]] > width)
                or np.any(rows[:, [1, 3]] > height)):
            raise ValueError("Face boxes are outside the source canvas")
        # Stable ordering gives a deterministic review contract, without assigning identity.
        detections = [{"box_xyxy": row[:4].astype(float).tolist(), "confidence": float(row[4])}
                      for row in rows]
        detections.sort(key=lambda item: (-item["confidence"], *item["box_xyxy"]))
        return {"schemaVersion": 1, "canvas": [height, width], "detections": detections,
                "asset_identity": self.identity, "identityVerified": False}

    def extract(self, image_bgr, is_bgr=True, **_):
        """Existing DFL rects extractor adapter; scores remain in predict records."""
        image = image_bgr[:, :, ::-1] if is_bgr else image_bgr
        return [tuple(item["box_xyxy"]) for item in self.predict(image)["detections"]]
