"""Paired landmark quality evaluation; assistant references are never expert NME.

Input manifest: {samples:[{id,image,sha256,box_xyxy,reference:[
 {region:left_eye|right_eye|inner_mouth,points:[[x,y],...],closed:true,
 excluded:false,uncertainty_px:3}]}]}. Images are only read. Outputs must remain
local when the samples or references are private. No speed-based ranking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui/python"))
from vision_landmarks import DEFAULT_ASSETS, MODEL_IDS, LandmarkPredictor, file_sha256, point_definition, validate_box
from vision_assets import prepare_evaluation_output


def curve_segments(points, closed):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Reference curve must contain at least two finite XY points")
    if closed:
        points = np.vstack([points, points[0]])
    a, b = points[:-1], points[1:]
    keep = np.linalg.norm(b - a, axis=1) > 1e-12
    if not keep.any():
        raise ValueError("Curve has no nonzero segments")
    return a[keep], b[keep]


def sample_curve(points, closed, count=256):
    a, b = curve_segments(points, closed)
    lengths = np.linalg.norm(b - a, axis=1)
    ends = np.cumsum(lengths)
    # Midpoint arc sampling avoids duplicating the first/last endpoint.
    distance = (np.arange(count) + .5) / count * ends[-1]
    index = np.minimum(np.searchsorted(ends, distance), len(a)-1)
    starts = np.r_[0, ends[:-1]]
    fraction = (distance - starts[index]) / lengths[index]
    return a[index] + fraction[:, None] * (b[index] - a[index])


def point_segment_distance(points, curve, closed):
    a, b = curve_segments(curve, closed)
    d = b - a
    projected = np.clip(np.sum((points[:, None] - a) * d, axis=2) / np.sum(d*d, axis=1), 0, 1)
    closest = a + projected[:, :, None] * d
    return np.min(np.linalg.norm(points[:, None] - closest, axis=2), axis=1)


def symmetric_curve_distance(a, a_closed, b, b_closed, samples=256):
    return float((point_segment_distance(sample_curve(a, a_closed, samples), b, b_closed).mean()
                  + point_segment_distance(sample_curve(b, b_closed, samples), a, a_closed).mean()) / 2)


def score_prediction(prediction, reference, box):
    points = np.asarray(prediction["points_original"], dtype=np.float64)
    definition = prediction["point_definition"]
    if points.shape != (definition["count"], 2):
        raise ValueError("Prediction point count does not match its native definition")
    region_map = {"left_eye": "image_left_eye", "right_eye": "image_right_eye", "inner_mouth": "inner_mouth"}
    scale = 256 / (1.35 * np.max(validate_box(box)[2:] - validate_box(box)[:2]))
    scores = []
    for ref in reference:
        if ref.get("excluded"):
            continue
        if ref["region"] not in region_map:
            raise ValueError("Unsupported reference region")
        curve = points[definition["regions"][region_map[ref["region"]]]]
        error = symmetric_curve_distance(curve, True, ref["points"], bool(ref.get("closed", True)))
        scores.append({"region": ref["region"], "source_px": error, "head256_px": error * scale,
                       "reference_uncertainty_source_px": ref.get("uncertainty_px")})
    groups = []
    eyes = [score["head256_px"] for score in scores if "eye" in score["region"]]
    mouth = [score["head256_px"] for score in scores if score["region"] == "inner_mouth"]
    if eyes:
        groups.append(float(np.mean(eyes)))
    if mouth:
        groups.append(float(np.mean(mouth)))
    return {"regions": scores, "eyes_head256_px": float(np.mean(eyes)) if eyes else None,
            "mouth_head256_px": float(np.mean(mouth)) if mouth else None,
            "image_quality_error_head256_px": float(np.mean(groups)) if groups else None,
            "metric": "assistant-reference symmetric curve distance; lower is closer; not expert NME"}


def load_hardset(hardset, reference):
    """Adapt read-only evidence records, never import H3CE application code."""
    protocol = json.loads((hardset / "protocol.json").read_text(encoding="utf-8-sig"))
    refs = json.loads(reference.read_text(encoding="utf-8-sig"))["entries"]
    grouped = {}
    for ref in refs:
        grouped.setdefault(ref["case"], []).append(ref)
    selected = {item["sha256"] for item in protocol["selected"]}
    samples = []
    for name in grouped:
        record = json.loads((hardset / name / "predictions.json").read_text(encoding="utf-8-sig"))
        if record["sha256"] not in selected:
            raise ValueError("Reference case outside locked selected set")
        samples.append({"id": name, "image": record["source"], "sha256": record["sha256"], "box_xyxy": record["bbox"], "reference": grouped[name]})
    if len(samples) != len(selected):
        raise ValueError("Incomplete locked sample set")
    return {"samples": samples, "reference_kind": "assistant_visual_manual_exploratory_not_expert",
            "reference_sha256": file_sha256(reference), "selection_protocol_sha256": file_sha256(hardset / "protocol.json"),
            "prior_models_seen": True, "claimed_blind": False}


def overlay(rgb, points, definition, references):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for region in definition["regions"].values():
        curve = [tuple(points[i]) for i in region]
        draw.line(curve + curve[:1], fill=(255, 180, 40), width=2)
    for ref in references:
        if ref.get("excluded"):
            continue
        curve = [tuple(point) for point in ref["points"]]
        if ref.get("closed", True):
            curve += curve[:1]
        draw.line(curve, fill=(40, 230, 235), width=2)
    return image


def paired_uncertainty(records, baseline="tufa98", repeats=20000, seed=20261006):
    values = {}
    for record in records:
        score = record["quality"]["image_quality_error_head256_px"]
        if score is not None:
            values.setdefault(record["model_id"], {})[record["sample_id"]] = score
    base = values.get(baseline, {})
    result = []
    for model_id, scores in values.items():
        if model_id == baseline:
            continue
        cases = sorted(set(base) & set(scores))
        if not cases:
            continue
        delta = np.array([scores[case] - base[case] for case in cases])
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(cases), size=(repeats, len(cases)))
        interval = np.quantile(delta[index].mean(axis=1), [.025, .975]).tolist()
        result.append({"model_id": model_id, "baseline": baseline, "paired_images": len(cases),
                       "mean_error_difference_head256_px": float(delta.mean()), "conditional_bootstrap_95pct": interval,
                       "repeats": repeats, "seed": seed,
                       "interpretation": "Conditional on locked nonexpert references and this sample selection; does not include reference bias or establish population superiority."})
    return result


def run(manifest, output, models, assets, device):
    output = prepare_evaluation_output(output)
    samples = manifest["samples"]
    if not samples:
        raise ValueError("No samples selected")
    for sample in samples:
        if not sample.get("sha256") or file_sha256(sample["image"]) != sample["sha256"]:
            raise ValueError("Source image checksum mismatch")
        validate_box(sample["box_xyxy"])
    records = []
    failures = []
    for model_id in models:
        try:
            predictor = LandmarkPredictor(model_id, assets, device)
        except Exception as error:
            failures.append({"model_id": model_id, "stage": "load", "error_type": type(error).__name__, "message": str(error)})
            continue
        for index, sample in enumerate(samples):
            try:
                rgb = np.array(Image.open(sample["image"]).convert("RGB"))
                prediction = predictor.predict(rgb, sample["box_xyxy"])
                score = score_prediction(prediction, sample.get("reference", []), sample["box_xyxy"])
                record = {"sample_id": sample["id"], "source_sha256": sample["sha256"], "model_id": model_id, "prediction": prediction, "quality": score}
                (output / f"{index+1:02d}-{model_id}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                overlay(rgb, prediction["points_original"], prediction["point_definition"], sample.get("reference", [])).save(output / f"{index+1:02d}-{model_id}.png")
                records.append(record)
                print(f"{model_id} {index+1}/{len(samples)} complete", flush=True)
            except Exception as error:
                failures.append({"model_id": model_id, "sample_id": sample["id"], "stage": "predict", "error_type": type(error).__name__, "message": str(error)})
        del predictor
    summary = []
    for model_id in models:
        matched = [record for record in records if record["model_id"] == model_id]
        finite = [record["quality"]["image_quality_error_head256_px"] for record in matched if record["quality"]["image_quality_error_head256_px"] is not None]
        eyes = [r["quality"]["eyes_head256_px"] for r in matched if r["quality"]["eyes_head256_px"] is not None]
        mouths = [r["quality"]["mouth_head256_px"] for r in matched if r["quality"]["mouth_head256_px"] is not None]
        summary.append({"model_id": model_id, "completed_images": len(matched), "scored_images": len(finite),
                        "scored_regions": sum(len(r["quality"]["regions"]) for r in matched),
                        "quality_error_head256_px": float(np.mean(finite)) if finite else None,
                        "eyes_error_head256_px": float(np.mean(eyes)) if eyes else None,
                        "mouth_error_head256_px": float(np.mean(mouths)) if mouths else None})
    result = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "summary": summary,
              "paired_uncertainty": paired_uncertainty(records),
              "failures": failures, "records": records, "sample_count": len(samples), "expert_nme": None,
              "reference_kind": manifest.get("reference_kind", "unspecified-reference-not-certified"),
              "reference_sha256": manifest.get("reference_sha256"), "selection_protocol_sha256": manifest.get("selection_protocol_sha256"),
              "scope": "shared locked source RGB and xyxy boxes; each model native official preprocessing; no training/no detector/no media writes",
              "limitations": ["Nonexpert references; earlier predictions were seen; not a blinded study.", "Fixed twelve-image hardset does not establish global model superiority.", "Native 68 and 98 contours differ in representation density.", "Regression candidate uses bbox-derived official crop geometry rather than WFLW GT-derived metadata.", "No visibility or 3D predictions. No automatic default model selection."],
              "implementation_sha256": {"adapter": file_sha256(ROOT / "webui/python/vision_landmarks.py"), "benchmark": file_sha256(__file__)}}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--h3-hardset", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.h3_hardset and not args.reference:
        parser.error("--reference is required with --h3-hardset")
    manifest = load_hardset(args.h3_hardset, args.reference) if args.h3_hardset else json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    result = run(manifest, args.output, args.models, args.assets, args.device)
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
