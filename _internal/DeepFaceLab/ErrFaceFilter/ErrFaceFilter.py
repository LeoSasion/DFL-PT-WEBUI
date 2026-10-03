"""Landmark error classifier from the original Guozili DFL utility.

CL.json is a lossless export of the bundled CL.model (XGBoost 1.6.2).
Its 160 numeric decision trees are evaluated with NumPy, so this tool has
no TensorFlow or XGBoost runtime dependency. Predictions are review hints.
"""

import argparse
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cv2ex import cv2_imread, cv2_imwrite
from DFLIMG import DFLIMG


class LandmarkErrorClassifier:
    def __init__(self, model_path=None):
        payload = json.loads(Path(model_path or Path(__file__).with_name("CL.json")).read_text(encoding="utf-8"))
        learner = payload["learner"]
        booster = learner["gradient_booster"]
        if learner["objective"]["name"] != "binary:logistic" or booster["name"] != "gbtree":
            raise ValueError("Unsupported landmark classifier model")
        params = learner["learner_model_param"]
        if int(params["num_feature"]) != 136 or int(params["num_class"]) != 0:
            raise ValueError("The landmark classifier requires 68 two-dimensional points")
        base_score = float(params["base_score"])
        self.base_margin = np.float32(math.log(base_score / (1.0 - base_score)))
        model = booster["model"]
        if any(int(item) != 0 for item in model["tree_info"]):
            raise ValueError("Unsupported landmark classifier tree groups")
        self.trees = []
        for tree in model["trees"]:
            if any(int(item) != 0 for item in tree["split_type"]):
                raise ValueError("Categorical classifier trees are unsupported")
            self.trees.append({
                "left": np.asarray(tree["left_children"], dtype=np.int32),
                "right": np.asarray(tree["right_children"], dtype=np.int32),
                "feature": np.asarray(tree["split_indices"], dtype=np.int32),
                "condition": np.asarray(tree["split_conditions"], dtype=np.float32),
                "default_left": np.asarray(tree["default_left"], dtype=bool),
            })

    def predict_proba(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 136:
            raise ValueError("Classifier input must have shape (N, 136)")
        margins = np.full(len(features), self.base_margin, dtype=np.float32)
        for tree in self.trees:
            nodes = np.zeros(len(features), dtype=np.int32)
            active = np.ones(len(features), dtype=bool)
            while active.any():
                indices = np.flatnonzero(active)
                node = nodes[indices]
                leaf = tree["left"][node] == -1
                if leaf.any():
                    leaves = indices[leaf]
                    margins[leaves] += tree["condition"][nodes[leaves]]
                    active[leaves] = False
                indices = indices[~leaf]
                if len(indices):
                    node = nodes[indices]
                    value = features[indices, tree["feature"][node]]
                    go_left = np.where(np.isnan(value), tree["default_left"][node], value < tree["condition"][node])
                    nodes[indices] = np.where(go_left, tree["left"][node], tree["right"][node])
        return 1.0 / (1.0 + np.exp(-margins))

    def predict(self, features):
        return (self.predict_proba(features) > 0.5).astype(np.int64)


def audit_directory(directory, classifier=None):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    classifier = classifier or LandmarkErrorClassifier()
    records, features = [], []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        item = {"name": path.name}
        try:
            image = DFLIMG.load(path)
            landmarks = np.asarray(image.get_landmarks(), dtype=np.float64) if image and image.has_data() else None
            if landmarks is None or landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
                raise ValueError("Missing or invalid 68-point DFL landmarks")
            size = image.get_shape()[0]
            if size <= 0:
                raise ValueError("Invalid aligned image size")
            features.append((landmarks * (512.0 / size)).reshape(136).astype(np.int64))
            item["_index"] = len(features) - 1
        except (ValueError, TypeError, KeyError, cv2.error) as error:
            item["error"] = str(error)
        records.append(item)
    scores = classifier.predict_proba(features) if features else []
    for item in records:
        if "_index" in item:
            score = float(scores[item.pop("_index")])
            item.update(probability=score, candidate=score > 0.5)
    return {"schemaVersion": 1, "directory": str(directory), "samples": records,
            "candidateCount": sum(bool(item.get("candidate")) for item in records),
            "invalidCount": sum("error" in item for item in records)}


def draw_directory(directory):
    directory = Path(directory).resolve()
    output = directory / "Landmarks"
    output.mkdir(exist_ok=True)
    names = []
    chains = [(0, 17, False), (17, 22, False), (22, 27, False), (27, 31, False),
              (31, 36, False), (36, 42, True), (42, 48, True), (48, 60, True), (60, 68, True)]
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        image = DFLIMG.load(path)
        if image is None or not image.has_data():
            continue
        landmarks = np.asarray(image.get_landmarks(), dtype=np.float64)
        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            continue
        canvas = np.zeros(image.get_shape(), dtype=np.uint8)
        for start, end, closed in chains:
            points = landmarks[start:end].round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], closed, (255, 255, 255), max(1, canvas.shape[0] // 100))
        cv2_imwrite(output / path.name, canvas)
        names.append(path.name)
    return {"output": str(output), "count": len(names)}


def quarantine(directory, names):
    directory = Path(directory).resolve()
    names = list(dict.fromkeys(names))
    if any(Path(name).name != name or not (directory / name).is_file() for name in names):
        raise ValueError("Invalid quarantine selection")
    output = directory / ("errFace-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    output.mkdir()
    manifest = {"schemaVersion": 1, "source": str(directory), "names": names,
                "createdAt": datetime.now(timezone.utc).isoformat()}
    (output / "restore.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    moved = []
    try:
        for name in names:
            shutil.move(str(directory / name), str(output / name))
            moved.append(name)
    except Exception:
        for name in reversed(moved):
            shutil.move(str(output / name), str(directory / name))
        raise
    return {"archive": str(output), "count": len(names)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Landmark error review (Guozili classifier)")
    parser.add_argument("mode", nargs="?", choices=("1", "2", "3", "audit", "draw", "clean"), default="audit")
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--quarantine", action="store_true", help="Move selected candidates to a recoverable folder")
    args = parser.parse_args(argv)
    directory = args.directory or input("DFL aligned directory: ").strip().strip('"')
    if args.mode in ("2", "draw"):
        result = draw_directory(directory)
    elif args.mode in ("3", "clean"):
        directory = Path(directory).resolve()
        review = directory / "Landmarks"
        if not review.is_dir():
            raise FileNotFoundError("Create/review the Landmarks directory before cleaning")
        names = [item.name for item in directory.iterdir() if item.is_file()
                 and item.suffix.lower() in (".jpg", ".jpeg") and not (review / item.name).exists()]
        result = {"selected": names}
        if args.quarantine:
            result.update(quarantine(directory, names))
    else:
        result = audit_directory(directory)
        if args.quarantine:
            result.update(quarantine(directory, [item["name"] for item in result["samples"] if item.get("candidate")]))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
