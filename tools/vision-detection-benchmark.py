"""Detection candidates + a shared landmark scorer on a locked local sample set.

H3CE reference boxes were detector predictions, not bounding-box ground truth.
The optional downstream contour score is a different, explicitly named quality
measure. No detector AP or recall is invented from baseline predictions.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from vision_assets import DEFAULT_ASSETS_ROOT, StageCache, atomic_json, canonical_digest, file_sha256, prepare_evaluation_output
from vision_detectors import DETECTOR_IDS, FaceDetector


def landmark_evaluation():
    spec = importlib.util.spec_from_file_location("landmark_evaluation", ROOT / "tools" / "vision-landmark-benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(manifest, output, models, *, assets=DEFAULT_ASSETS_ROOT, device="cuda:0", image_size=960,
        downstream_landmarks=True):
    output = Path(output)
    samples = manifest["samples"]
    if not models or len(set(models)) != len(models):
        raise ValueError("Select unique candidate model IDs")
    if not samples or len(samples) > 24 or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Lock 1..24 uniquely identified samples before comparison")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", sample["id"]) for sample in samples):
        raise ValueError("Sample IDs cannot contain path separators")
    output = prepare_evaluation_output(output)
    contract = {"manifestDigest": canonical_digest(manifest), "models": list(models),
                "device": device, "imageSize": image_size, "selectionPolicy": "quality-only",
                "performanceInRanking": False, "referenceKind": manifest.get("reference_kind", "unspecified"),
                "scorer": "tufa98-shared-downstream-contours" if downstream_landmarks else None,
                "sourceReadOnly": True}
    atomic_json(output / "protocol.json", contract)
    cache = StageCache(output / "cache")
    evaluation = landmark_evaluation()
    scorer = None
    if downstream_landmarks:
        from vision_landmarks import LandmarkPredictor
        scorer = LandmarkPredictor("tufa98", assets_root=Path(assets) / "landmarks", device=device)
    records, failures, summaries = [], [], []
    for model_id in models:
        try:
            detector = FaceDetector(model_id, assets_root=assets, device=device, image_size=image_size)
        except Exception as error:
            failures.append({"model_id": model_id, "phase": "load", "error": str(error), "qualityScore": None})
            continue
        for sample in samples:
            try:
                if file_sha256(sample["image"]) != sample["sha256"]:
                    raise ValueError("Source bytes changed after the sample was locked")
                rgb = np.array(Image.open(sample["image"]).convert("RGB"))
                dependencies = {"sourceSha256": sample["sha256"], "asset": detector.identity,
                                "adapterSha256": file_sha256(ROOT / "webui/python/vision_detectors.py"),
                                "device": device, "decode": "pillow-rgb-no-exif-v1"}
                result = cache.read("detection", dependencies)
                cache_hit = result is not None
                if result is None:
                    result = detector.predict(rgb)
                    cache.write("detection", dependencies, result)
                record = {"sample_id": sample["id"], "model_id": model_id,
                          "sourceSha256": sample["sha256"], "prediction": result,
                          "cacheHit": cache_hit, "detectionAccuracyScore": None,
                          "detectionAccuracyStatus": "requires-annotated-box-ground-truth",
                          "quality": {"image_quality_error_head256_px": None}}
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                for item in result["detections"]:
                    draw.rectangle(item["box_xyxy"], outline="#00e8ff", width=3)
                if scorer and result["detections"] and sample.get("reference"):
                    # H3CE's preparation selected the largest face. This remains
                    # a declared crop policy, never a multi-person identity rule.
                    box = max(result["detections"], key=lambda item:
                              (item["box_xyxy"][2] - item["box_xyxy"][0]) *
                              (item["box_xyxy"][3] - item["box_xyxy"][1]))["box_xyxy"]
                    prediction = scorer.predict(rgb, box)
                    record["downstreamLandmarks"] = prediction
                    # Fixed common reference normalization; candidate boxes
                    # cannot change their own scoring denominator.
                    record["quality"] = evaluation.score_prediction(prediction, sample["reference"], sample["box_xyxy"])
                    record["quality"]["scope"] = "detector-plus-fixed-TUFA98; not standalone detection AP"
                    for point in prediction["points_original"]:
                        x, y = point
                        draw.ellipse((x-2, y-2, x+2, y+2), fill="#ffb347")
                atomic_json(output / f'{sample["id"]}-{model_id}.json', record)
                image.save(output / f'{sample["id"]}-{model_id}.png')
                records.append(record)
            except Exception as error:
                failures.append({"model_id": model_id, "sample_id": sample["id"],
                                 "phase": "inference-or-score", "error": str(error), "qualityScore": None})
        matches = [record for record in records if record["model_id"] == model_id]
        scores = [record["quality"]["image_quality_error_head256_px"] for record in matches
                  if record["quality"]["image_quality_error_head256_px"] is not None]
        summaries.append({"model_id": model_id, "completedImages": len(matches),
                          "imagesWithDetections": sum(bool(record["prediction"]["detections"]) for record in matches),
                          "totalDetections": sum(len(record["prediction"]["detections"]) for record in matches),
                          "scoredImages": len(scores),
                          "downstreamContourErrorHead256": float(np.mean(scores)) if scores else None,
                          "detectorAP": None, "detectorRecall": None,
                          "accuracyStatus": "No human bounding-box annotations supplied",
                          "assetIdentity": detector.identity})
    final = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
             "protocol": contract, "summaries": summaries, "failures": failures,
             "pairedUncertainty": evaluation.paired_uncertainty(records, baseline="yolo11m-face"),
             "records": records, "automaticDefaultChange": False}
    atomic_json(output / "results.json", final)
    print(json.dumps({"summaries": summaries, "failures": failures,
                      "pairedUncertainty": final["pairedUncertainty"]}, ensure_ascii=False), flush=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--h3-hardset", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DETECTOR_IDS, default=list(DETECTOR_IDS))
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--no-downstream-landmarks", action="store_true")
    args = parser.parse_args()
    if args.h3_hardset and args.reference:
        manifest = landmark_evaluation().load_hardset(args.h3_hardset, args.reference)
    elif args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        parser.error("Supply --manifest, or both --h3-hardset and --reference")
    result = run(manifest, args.output, args.models, assets=args.assets, device=args.device,
        image_size=args.image_size, downstream_landmarks=not args.no_downstream_landmarks)
    if result["failures"] or len(result["records"]) != len(args.models) * len(manifest["samples"]):
        sys.exit(1)
