"""Compute three frozen metrics on the same locked restoration pairs; no ranking."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from vision_metrics import METRIC_IDS, PerceptualMetricSuite
from vision_restoration import ASSETS, sha256_file

RESTORERS = ("swinir-psnr", "realesrgan-x4plus", "gfpgan-v1.4")


def write_json(target, value):
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def rgb(target):
    image = cv2.imread(str(target), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable locked image: {target.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def evaluation_status():
    return {"humanPreferenceLabels": None, "humanPreferenceCorrelation": None,
            "metricQualityRanking": None, "restorationQualityRanking": None,
            "metricWinner": None, "reason": "no independent human preference annotations; raw distances only"}


def validate_pair_manifest(protocol, restoration):
    cases = [case["case"] for case in protocol["cases"]]
    model_ids = [model["model"] for model in restoration["models"]]
    if (protocol["caseCount"] != 6 or len(cases) != 6 or len(set(cases)) != 6
            or len(model_ids) != 3 or set(model_ids) != set(RESTORERS)
            or set(protocol["models"]) != set(RESTORERS)):
        raise ValueError("Locked six unique sources and three expected restorers required")
    for model in restoration["models"]:
        model_cases = [case["case"] for case in model["cases"]]
        if len(model_cases) != 6 or set(model_cases) != set(cases):
            raise ValueError("Every restorer must contain each locked case exactly once")


def benchmark(pairs, assets, output, device="cuda"):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    protocol_file, restoration_file = pairs / "protocol.json", pairs / "results.json"
    protocol, restoration = json.loads(protocol_file.read_text(encoding="utf-8")), json.loads(restoration_file.read_text(encoding="utf-8"))
    if restoration["protocolSha256"] != sha256_file(protocol_file):
        raise ValueError("Restoration protocol changed; comparison refused")
    validate_pair_manifest(protocol, restoration)
    suite = PerceptualMetricSuite(assets, device=device)
    report = {"schemaVersion": 1, "purpose": "frozen perceptual metric candidate coverage, no training or quality ranking",
              "restorationProtocolSha256": sha256_file(protocol_file), "restorationResultsSha256": sha256_file(restoration_file),
              "metrics": list(METRIC_IDS), "caseCount": 6, "expectedPairCount": 36, "expectedDistanceCount": 108,
              "pairCount": 0, "scoredDistanceCount": 0, "status": "running",
              "inputs": "same512 RGB uint8; identical source-valid mask for all metrics; invalid output pixels replaced by source reference",
              "scope": "original-relative source contains real defects; down4 synthetic reconstruction and clean preservation; no pristine HQ",
              "runtime": {"torch": torch.__version__, "precision": "FP32", "TF32": False, "automaticDownloads": False},
              "metricProvenance": {**suite.provenance, "lpipsAlexBackboneSha256": ASSETS["lpips-backbone"][1],
                                   "lpipsAlexCalibrationSha256": ASSETS["lpips-calibration"][1]},
              "evaluation": evaluation_status(), "pairs": [], "summaries": [],
              "cautions": ["raw distances have different calibrations; numbers are not comparable across metric families",
                           "lower distance within one metric does not prove anatomical/identity fidelity",
                           "LPIPS and DISTS learned parameters remain frozen; no gradient or optimization",
                           "official DISTS CLI resize256 is explicitly disabled to keep the same512 inputs as LPIPS"]}
    for model in restoration["models"]:
        model_pairs = []
        for result in model["cases"]:
            case = next(item for item in protocol["cases"] if item["case"] == result["case"])
            directory = pairs / case["case"]
            for filename, expected in [("reference.png", case["referenceSha256"]), ("valid-mask.png", case["validMaskSha256"]),
                                       (f"{model['model']}.png", result["outputSha256"]),
                                       (f"{model['model']}-clean.png", result["cleanOutputSha256"])]:
                if sha256_file(directory / filename) != expected:
                    raise ValueError("Locked restoration pixels changed; metric comparison refused")
            reference = rgb(directory / "reference.png")
            if reference.shape != (512, 512, 3):
                raise ValueError("Locked metric references must remain 512x512 RGB")
            valid = cv2.imread(str(directory / "valid-mask.png"), cv2.IMREAD_GRAYSCALE) >= 128
            for kind, filename, old_metric in [("synthetic-down4", f"{model['model']}.png", result["full"]["lpips"]),
                                               ("clean-preservation", f"{model['model']}-clean.png", result["cleanPreservation"]["lpips"])]:
                distances = suite.evaluate(reference, rgb(directory / filename), valid)
                delta = abs(distances["lpips-alex-v0.1"] - old_metric)
                if delta > 2e-6:
                    raise ValueError("Alex baseline does not reproduce locked restoration evaluation")
                entry = {"restorationModel": model["model"], "case": case["case"], "pairKind": kind,
                         "validPixels": int(valid.sum()), "referenceSha256": case["referenceSha256"],
                         "predictionSha256": sha256_file(directory / filename), "distances": distances,
                         "alexBaselineAbsoluteDelta": delta}
                report["pairs"].append(entry)
                report["pairCount"] = len(report["pairs"])
                report["scoredDistanceCount"] = len(report["pairs"]) * len(METRIC_IDS)
                model_pairs.append(entry)
                print(f"{model['model']} {case['case']} {kind}: " + json.dumps(distances), flush=True)
        for kind in ["synthetic-down4", "clean-preservation"]:
            selected = [entry for entry in model_pairs if entry["pairKind"] == kind]
            report["summaries"].append({"restorationModel": model["model"], "pairKind": kind, "pairCount": len(selected),
                                        "meanDistances": {metric: float(np.mean([entry["distances"][metric] for entry in selected])) for metric in METRIC_IDS},
                                        "ranking": None, "correlationWithHumanPreference": None})
        write_json(output, report)
    report["status"] = "complete"
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "workspace" / ".vision-evaluation" / "restoration" / "six-source-v1")
    parser.add_argument("--assets", type=Path, default=ROOT / "workspace" / ".vision-models" / "metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / ".vision-evaluation" / "metrics" / "three-metrics-six-sources-v1.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / "workspace" / ".vision-evaluation").resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark(args.pairs.resolve(), args.assets.resolve(), output, args.device)


if __name__ == "__main__":
    main()
