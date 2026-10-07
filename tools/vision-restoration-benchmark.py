"""Six-source locked restoration comparison; private outputs, frozen inference only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webui" / "python"))
from vision_restoration import (ASSETS, MODELS, OfflineLpips, RestorationModel,
                                aligned_reference, metric_quality_score, quality_metrics,
                                sha256_file, synthetic_down4)


def save_json(target, value):
    target = Path(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {Path(path).name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise ValueError("Image output failed")


def region_box(points, minimum_size=64, margin=12):
    points = np.asarray(points, dtype=np.float64)
    lower, upper = points.min(axis=0) - margin, points.max(axis=0) + margin
    middle, extent = (lower + upper) / 2, np.maximum(upper - lower, minimum_size)
    lower = np.clip(np.floor(middle - extent / 2), 0, 512).astype(int)
    upper = np.clip(np.ceil(middle + extent / 2), 0, 512).astype(int)
    return [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]


def prepare(reference_root, output):
    source_protocol = reference_root / "runs" / "h3-landmark-hardset-20260914-v1" / "protocol.json"
    hardset = json.loads(source_protocol.read_text(encoding="utf-8"))
    manual_path = reference_root / "runs" / "h3-landmark-precision-20260914-v1" / "manual_reference_v1.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    cases = []
    for ordinal, selected in enumerate(hardset["selected"][:6], start=1):
        source = Path(selected["path"])
        if sha256_file(source) != selected["sha256"]:
            raise ValueError("Original source SHA-256 changed; locked comparison refused")
        case_name = f"{ordinal:02d}_{source.stem}"
        prediction_file = source_protocol.parent / case_name / "predictions.json"
        prediction = json.loads(prediction_file.read_text(encoding="utf-8"))
        if prediction["sha256"] != selected["sha256"]:
            raise ValueError("Saved landmarks do not belong to the locked source")
        points = np.asarray(prediction["predictions"]["TUFA98"]["source_xy"], dtype=np.float64)
        if points.shape != (98, 2) or not np.isfinite(points).all():
            raise ValueError("Saved TUFA98 coordinates are invalid")
        five = points[[96, 97, 54, 76, 82]]
        image = read_rgb(source)
        reference, valid, matrix = aligned_reference(image, five)
        directory = output / case_name
        directory.mkdir(parents=True, exist_ok=True)
        write_rgb(directory / "reference.png", reference)
        write_rgb(directory / "lq-down4.png", synthetic_down4(reference))
        write_rgb(directory / "bicubic.png", cv2.resize(synthetic_down4(reference), (512, 512), interpolation=cv2.INTER_CUBIC))
        cv2.imwrite(str(directory / "valid-mask.png"), valid.astype(np.uint8) * 255)
        regions = []
        for annotation in manual["entries"]:
            if annotation["case"] != case_name:
                continue
            if annotation.get("excluded"):
                regions.append({"name": annotation["region"], "available": False,
                                "reason": "excluded-in-nonexpert-source-reference"})
                continue
            original_points = np.asarray(annotation["points"], dtype=np.float64)
            transformed = original_points @ matrix[:, :2].T + matrix[:, 2]
            regions.append({"name": annotation["region"], "available": True, "box": region_box(transformed),
                            "definition": "expanded-source-contour-content-window; not landmark ground truth",
                            "referenceQualification": "assistant visual exploratory, nonexpert; original exclusions retained"})
        cases.append({"case": case_name, "sourceFilename": source.name, "sourceSha256": selected["sha256"],
                      "savedPredictionSha256": sha256_file(prediction_file), "sourceSize": [image.shape[1], image.shape[0]],
                      "fivePointSource": "saved TUFA98 indices [96,97,54,76,82], predicted not human truth",
                      "fiveSourcePoints": five.tolist(), "sourceTo512": matrix.tolist(),
                      "alignmentResidualPx": float(np.mean(np.linalg.norm(five @ matrix[:, :2].T + matrix[:, 2]
                                                                    - __import__('vision_restoration').FFHQ_TEMPLATE, axis=1))),
                      "referenceSha256": sha256_file(directory / "reference.png"),
                      "lqSha256": sha256_file(directory / "lq-down4.png"),
                      "validMaskSha256": sha256_file(directory / "valid-mask.png"),
                      "validPixelFraction": float(valid.mean()), "regions": regions})
    protocol = {"schemaVersion": 1, "caseCount": len(cases), "selection": "first six cases of previously frozen H3CE 12-case hardset; locked before restoration inference",
                "originalProtocolSha256": sha256_file(source_protocol), "regionReferenceSha256": sha256_file(manual_path),
                "referenceKind": "unaltered aligned source-relative reference, containing existing real defects; NOT pristine HQ or AI target",
                "realDegradationTruth": False, "degradation": "512 reference ->128 INTER_AREA down4 only; no synthetic noise/JPEG/AI targets",
                "alignment": "512 GFPGAN/facexlib FFHQ-style five-point template; deterministic similarity least squares; reflected border excluded by warped/eroded source-valid mask",
                "models": list(MODELS), "cleanPreservation": "unaltered512 -> restoration network -> same512; SR native x4 then area downsample",
                "scoreFormula": "metricScore=100*(.25*clip(PSNR/40)+.35*clip(SSIM)+.40*clip(1-LPIPS)); case=.50 full + .30 mean available eye/mouth windows + .20 clean preservation, omit unavailable ROI term; mean6; no performance term",
                "limitations": ["original-relative reconstruction only; existing original degradation is not pristine truth",
                                "predicted alignment and nonexpert region windows are not anatomy/identity ground truth",
                                "scores do not establish real low-quality repair or identity preservation",
                                "SR clean preservation uses native4x plus area resize and tile128 halo16; boundary artifacts remain part of tested behavior"],
                "cases": cases}
    save_json(output / "protocol.json", protocol)
    return protocol


def run_models(protocol, output, assets, device):
    import torch
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    perceptual = OfflineLpips(assets, device=device)
    report = {"schemaVersion": 1, "purpose": "finite original-relative quality comparison, no training and no performance ranking",
              "protocolSha256": sha256_file(output / "protocol.json"), "torchVersion": torch.__version__,
              "lpips": {"backboneSha256": ASSETS["lpips-backbone"][1], "calibrationSha256": ASSETS["lpips-calibration"][1],
                        "strict": True, "pretrainedAutoDownload": False}, "models": [], "limitations": protocol["limitations"]}
    for model_id in MODELS:
        print(f"Loading frozen model: {model_id}", flush=True)
        model = RestorationModel(model_id, assets, device=device)
        model_results = []
        for case in protocol["cases"]:
            directory = output / case["case"]
            for filename, key in [("reference.png", "referenceSha256"), ("lq-down4.png", "lqSha256"), ("valid-mask.png", "validMaskSha256")]:
                if sha256_file(directory / filename) != case[key]:
                    raise ValueError("Locked benchmark inputs changed")
            reference, lq = read_rgb(directory / "reference.png"), read_rgb(directory / "lq-down4.png")
            valid = cv2.imread(str(directory / "valid-mask.png"), cv2.IMREAD_GRAYSCALE) >= 128
            restored, clean = model.restore(lq), model.restore(reference)
            write_rgb(directory / f"{model_id}.png", restored)
            write_rgb(directory / f"{model_id}-clean.png", clean)
            full = quality_metrics(reference, restored, valid, perceptual)
            full["qualityScore"] = metric_quality_score(full)
            clean_metrics = quality_metrics(reference, clean, valid, perceptual)
            clean_metrics["qualityScore"] = metric_quality_score(clean_metrics)
            regions = []
            for region in case["regions"]:
                entry = {"name": region["name"], "available": region["available"]}
                if region["available"]:
                    x0, y0, x1, y1 = region["box"]
                    mask = valid[y0:y1, x0:x1]
                    if mask.sum() < 100 or min(mask.shape) < 32:
                        entry.update({"available": False, "reason": "insufficient-valid-region"})
                    else:
                        entry["metrics"] = quality_metrics(reference[y0:y1, x0:x1], restored[y0:y1, x0:x1], mask, perceptual)
                        entry["metrics"]["qualityScore"] = metric_quality_score(entry["metrics"])
                regions.append(entry)
            roi_scores = [entry["metrics"]["qualityScore"] for entry in regions if entry["available"]]
            denominator = 1.0 if roi_scores else 0.7
            case_score = (0.5*full["qualityScore"] + 0.2*clean_metrics["qualityScore"]
                          + (0.3*float(np.mean(roi_scores)) if roi_scores else 0)) / denominator
            item = {"case": case["case"], "full": full, "cleanPreservation": clean_metrics,
                    "regions": regions, "qualityScore": case_score,
                    "outputSha256": sha256_file(directory / f"{model_id}.png"),
                    "cleanOutputSha256": sha256_file(directory / f"{model_id}-clean.png")}
            model_results.append(item)
            print(f"{model_id} {case['case']} quality={case_score:.3f}", flush=True)
        summary = {"model": model_id, "provenance": model.provenance, "cases": model_results,
                   "meanQualityScore": float(np.mean([item["qualityScore"] for item in model_results])),
                   "meanPsnrDb": float(np.mean([item["full"]["psnrDb"] for item in model_results])),
                   "meanSsim": float(np.mean([item["full"]["ssim"] for item in model_results])),
                   "meanLpips": float(np.mean([item["full"]["lpips"] for item in model_results])),
                   "meanCleanScore": float(np.mean([item["cleanPreservation"]["qualityScore"] for item in model_results]))}
        report["models"].append(summary)
        save_json(output / "results.json", report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["qualityOrder"] = [item["model"] for item in sorted(report["models"], key=lambda item: -item["meanQualityScore"])]
    report["selectionStatus"] = "measured original-relative quality order; human real-degradation/identity review still required before default deployment"
    save_json(output / "results.json", report)
    make_panel(protocol, report, output)
    return report


def make_panel(protocol, report, output):
    import html
    columns = [("Reference (original defects retained)", "reference.png"), ("Down4 + bicubic", "bicubic.png")]
    columns += [(model, f"{model}.png") for model in MODELS]
    sections = []
    for case in protocol["cases"]:
        figures = ''.join(f'<figure><figcaption>{html.escape(label)}</figcaption><img src="{case["case"]}/{filename}"></figure>' for label, filename in columns)
        clean = ''.join(f'<figure><figcaption>{model} clean preservation</figcaption><img src="{case["case"]}/{model}-clean.png"></figure>' for model in MODELS)
        sections.append(f'<section><h2>{case["case"]}</h2><div class="row">{figures}</div><details><summary>Clean input preservation</summary><div class="row">{clean}</div></details></section>')
    rows = ''.join(f'<tr><td>{model["model"]}</td><td>{model["meanQualityScore"]:.3f}</td><td>{model["meanPsnrDb"]:.3f}</td><td>{model["meanSsim"]:.4f}</td><td>{model["meanLpips"]:.4f}</td><td>{model["meanCleanScore"]:.3f}</td></tr>' for model in report["models"])
    page = '<!doctype html><meta charset="utf-8"><title>Private restoration quality comparison</title><style>body{background:#101713;color:#d9e9df;font:14px system-ui;padding:24px}.row{display:flex;overflow:auto;gap:10px}figure{margin:0;min-width:240px}img{width:240px}figcaption{margin:8px 0}td,th{padding:9px;text-align:left}section{margin:30px 0}summary{cursor:pointer}</style>'
    page += '<h1>6 locked sources / 3 frozen models</h1><p>Original-relative synthetic down4 only. Reference retains real defects; NOT pristine HQ. No training, AI target, performance ranking or identity ground truth. Eye/mouth windows are nonexpert exploratory source references; exclusions retained.</p>'
    page += '<table><tr><th>Model</th><th>Quality score</th><th>PSNR dB</th><th>SSIM</th><th>LPIPS</th><th>Clean score</th></tr>' + rows + '</table>' + ''.join(sections)
    (output / "comparison.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "workspace" / ".vision-evaluation" / "restoration" / "six-source-v1")
    parser.add_argument("--assets", type=Path, default=PROJECT_ROOT / "workspace" / ".vision-models" / "restoration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=["prepare", "run", "all"], default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((PROJECT_ROOT / "workspace" / ".vision-evaluation").resolve())
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if args.stage in ["prepare", "all"]:
        if protocol_path.exists():
            raise ValueError("Existing locked protocol retained; use --stage run or choose a new private output directory")
        protocol = prepare(args.reference_root.resolve(), output)
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if args.stage in ["run", "all"]:
        run_models(protocol, output, args.assets.resolve(), args.device)


if __name__ == "__main__":
    main()
