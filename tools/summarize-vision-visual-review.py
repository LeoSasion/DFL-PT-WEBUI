"""Validate locked blinded Astra reviews, then reveal model identities.

Indices summarize an ordinal visual rubric. They are never accuracy percentages,
expert ground truth, AP, IoU, or proof that a default should change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui/python"))
from vision_assets import atomic_json, file_sha256, resolve_asset_path

REGIONS = ("image_left_eye", "image_right_eye", "inner_mouth")
MASK_FIELDS = ("faceCoreCoverageScore", "skinBoundaryFitScore", "externalLeakScore", "occlusionHandlingScore")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def score(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 4:
        raise ValueError("Visual scores must be integers 0..4 or null")
    return value


def mean(values):
    known = [x for x in values if x is not None]
    return float(np.mean(known)) if known else None


def landmark_case(regions, visibility):
    values = {}
    for region in REGIONS:
        entry = regions[region]
        value = score(entry["score"])
        if not entry.get("reason") or entry.get("confidence") not in ("high", "medium", "low"):
            raise ValueError("Each region requires a reason and confidence")
        scorable = visibility[region]["scorable"]
        if type(scorable) is not bool or (scorable and value is None) or (not scorable and value is not None):
            raise ValueError("Candidate scores must follow the source-only visibility mask")
        values[region] = value
    eyes = mean([values[r] for r in REGIONS[:2]])
    mouth = values["inner_mouth"]
    return {"eyes": eyes, "mouth": mouth, "overall": mean([eyes, mouth])}


def validate_review(kind, manifest, review, directory):
    assessor = review["assessor"]
    expected_assessor = {"model": "gpt-6-astra", "reasoningEffort": "low", "identitiesHidden": True,
                         "previousScoresHidden": True, "visualReviewNotGroundTruth": True}
    if any(assessor.get(k) != v for k, v in expected_assessor.items()):
        raise ValueError("Assessor identity or blinding declaration mismatch")
    expected = {c["case"]: c for c in manifest["cases"]}
    actual = {c["case"]: c for c in review["cases"]}
    if any(type(c["case"]) is not int for c in review["cases"]) or len(actual) != len(review["cases"]) or set(actual) != set(expected):
        raise ValueError("Incomplete or duplicate reviewed cases")
    for case_id, case in actual.items():
        planned = expected[case_id]
        sheets = [planned[k] for k in ("reference", "candidates") if k in planned] if kind == "landmarks" else [planned["sheet"]]
        inspected = case["inspectedFiles"]
        if {Path(p).name for p in inspected} != {p["file"] for p in sheets}:
            raise ValueError("Every planned visual sheet must have been inspected")
        for item in sheets:
            path = resolve_asset_path(directory, item["file"])
            if file_sha256(path) != item["sha256"]:
                raise ValueError("Review evidence changed")
        tokens = [c["token"] for c in case["candidates"]]
        if len(set(tokens)) != len(tokens) or set(tokens) != set(planned["tokens"]):
            raise ValueError("Incomplete or duplicate candidate ratings")
        if kind == "landmarks":
            for region in REGIONS:
                entry = case["reference"][region]
                score(entry["score"])
                if not entry.get("reason") or entry.get("confidence") not in ("high", "medium", "low"):
                    raise ValueError("Each reference region requires a reason and confidence")
                if not case["sourceVisibility"][region].get("reason"):
                    raise ValueError("Source visibility requires an evidence reason")
        for candidate in case["candidates"]:
            if kind == "landmarks":
                landmark_case(candidate["regions"], case["sourceVisibility"])
            else:
                fields = ("faceCoverageScore", "boxPlacementScore") if kind == "detection" else MASK_FIELDS
                for field in fields:
                    score(candidate[field])
                if not candidate.get("reason") or candidate.get("confidence") not in ("high", "medium", "low"):
                    raise ValueError("Every candidate requires a reason and confidence")
    return actual


def paired_intervals(rows, baseline="tufa98"):
    scores = {}
    for row in rows:
        scores.setdefault(row["model_id"], {})[row["case"]] = row["overall"]
    baseline_scores = scores.get(baseline, {})
    output = []
    for model, values in scores.items():
        if model == baseline:
            continue
        shared = sorted(c for c in baseline_scores if baseline_scores[c] is not None and values.get(c) is not None)
        if not shared:
            continue
        delta = np.array([25 * (values[c] - baseline_scores[c]) for c in shared])
        rng = np.random.default_rng(20261006)
        boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
        output.append({"model_id": model, "baseline": baseline, "pairedCases": len(shared),
                       "visualIndexDifference": float(delta.mean()), "conditional95Interval": np.quantile(boot, [.025, .975]).tolist(),
                       "scope": "fixed-case and fixed-LLM-ratings resampling only; excludes assessor bias and population uncertainty"})
    return output


def summarize(pack):
    protocol = read(pack / "protocol.json")
    sealed = read(pack / "sealed-key.json")
    if protocol["assessorModel"] != "gpt-6-astra" or protocol["reasoningEffort"] != "low":
        raise ValueError("Unexpected requested assessor")
    for item in sealed["inputRecords"] + sealed["sources"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("Original source or inference evidence changed")
    output = {"schemaVersion": 1, "generatedAt": datetime.now(timezone.utc).isoformat(),
              "assessorModel": "gpt-6-astra", "reasoningEffort": "low", "protocolSha256": file_sha256(pack / "protocol.json"),
              "sealedKeySha256": file_sha256(pack / "sealed-key.json"), "reviewInputs": [], "categories": {},
              "isAccuracyPercentage": False, "expertGroundTruth": False, "automaticDefaultChange": False,
              "limitations": protocol["limitations"]}
    for kind in ("landmarks", "detection", "masks"):
        directory = pack / kind
        path = directory / "review.json"
        review = read(path)
        cases = validate_review(kind, read(directory / "manifest.json"), review, directory)
        output["reviewInputs"].append({"kind": kind, "sha256": file_sha256(path)})
        rows = []
        for case_id, case in sorted(cases.items()):
            for candidate in case["candidates"]:
                keys = [k for k in sealed["mapping"] if k["kind"] == kind and k["case"] == case_id and k["token"] == candidate["token"]]
                if len(keys) != 1:
                    raise ValueError("Unblinding key is ambiguous")
                row = {"case": case_id, "token": candidate["token"], "model_id": keys[0]["model_id"], "sample_id": keys[0]["sample_id"]}
                if kind == "landmarks":
                    row.update(landmark_case(candidate["regions"], case["sourceVisibility"]))
                elif kind == "detection":
                    row.update({f: score(candidate[f]) for f in ("faceCoverageScore", "boxPlacementScore")})
                    row["overall"] = mean([row["faceCoverageScore"], row["boxPlacementScore"]])
                    for field in ("clearFalsePositiveCount", "clearMissedFaceCount", "duplicateFaceCount"):
                        value = candidate[field]
                        if value is not None and (type(value) is not int or value < 0):
                            raise ValueError("Visual counts must be nonnegative integers or null")
                        row[field] = value
                else:
                    row.update({f: score(candidate[f]) for f in MASK_FIELDS})
                rows.append(row)
        models = []
        fields = ("eyes", "mouth", "overall") if kind == "landmarks" else (("faceCoverageScore", "boxPlacementScore", "overall") if kind == "detection" else MASK_FIELDS)
        for model_id in sorted({r["model_id"] for r in rows}):
            selected = [r for r in rows if r["model_id"] == model_id]
            summary = {"model_id": model_id, "reviewedCases": len(selected)}
            for field in fields:
                value = mean([r[field] for r in selected])
                summary[field + "Index"] = None if value is None else value * 25
                summary[field + "ScoredCases"] = sum(r[field] is not None for r in selected)
            if kind == "detection":
                for field in ("clearFalsePositiveCount", "clearMissedFaceCount", "duplicateFaceCount"):
                    summary[field] = sum(r[field] for r in selected if r[field] is not None)
                    summary[field + "KnownCases"] = sum(r[field] is not None for r in selected)
            models.append(summary)
        category = {"cases": len(cases), "candidateRatings": len(rows), "models": models, "unblindedRows": rows}
        if kind == "landmarks":
            category["pairedUncertainty"] = paired_intervals(rows)
            category["reference"] = [{"case": case_id, "regions": case["reference"], "visibility": case["sourceVisibility"]} for case_id, case in sorted(cases.items())]
        if kind == "masks":
            category["pooledWinner"] = None
            category["qualification"] = "different semantics and FFHQ/WF alignment; dimension-specific visual indices only"
        output["categories"][kind] = category
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    args = parser.parse_args()
    target = args.pack / "summary.json"
    if target.exists():
        raise ValueError("Locked visual summary already exists; use a fresh review version")
    result = summarize(args.pack.resolve())
    atomic_json(target, result)
    print(json.dumps({"output": str(target.resolve()), "categories": {k: v["candidateRatings"] for k, v in result["categories"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
