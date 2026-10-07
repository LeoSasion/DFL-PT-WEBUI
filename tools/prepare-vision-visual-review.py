"""Prepare blinded, hash-bound visual review sheets from saved inference.

This renders evidence only; it does not edit source media or run any network.
The sealed key is for the coordinator, never the visual assessor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui/python"))
from vision_assets import atomic_json, file_sha256, prepare_evaluation_output


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


FONT = ImageFont.load_default(size=18)
WIDTH = 380
YELLOW = (255, 210, 0)


def tile(image, crop, size, label):
    panel = Image.new("RGB", (size[0], size[1] + 30), (20, 20, 20))
    panel.paste(image.crop(crop).resize(size, Image.Resampling.LANCZOS), (0, 30))
    ImageDraw.Draw(panel).text((8, 5), label, font=FONT, fill="white")
    return panel


def finite_points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Invalid saved coordinates")
    return points


def line(image, points, closed=True):
    points = finite_points(points)
    draw = ImageDraw.Draw(image)
    curve = [tuple(p) for p in points]
    draw.line(curve + curve[:1] if closed else curve, fill=YELLOW, width=2)
    for x, y in curve:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=YELLOW)


def region_crop(all_points, image_size, height):
    points = np.concatenate(all_points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    w = max(float(hi[0] - lo[0]) + 48, 180)
    h = max(float(hi[1] - lo[1]) + 48, w * height / WIDTH)
    w = max(w, h * WIDTH / height)
    # One shared crop per region; differences cannot change the displayed scale.
    return tuple((center - [w / 2, h / 2]).tolist() + (center + [w / 2, h / 2]).tolist())


def sheet(images, labels, crops, name, output):
    heights = [380, 180, 180, 220] if len(crops) == 4 else [512]
    canvas = Image.new("RGB", (WIDTH * len(images), sum(h + 30 for h in heights)), (20, 20, 20))
    for col, (image, label) in enumerate(zip(images, labels)):
        top = 0
        for row, (crop, height) in enumerate(zip(crops, heights)):
            names = ["whole face", "image-left eye", "image-right eye", "inner mouth"] if len(crops) == 4 else ["full image"]
            panel = tile(image, crop, (WIDTH, height), f"{label} | {names[row]}")
            canvas.paste(panel, (col * WIDTH, top))
            top += height + 30
    path = output / name
    canvas.save(path)
    return {"file": name, "sha256": file_sha256(path), "size": list(canvas.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--restoration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    landmark = read(args.landmarks)
    detector = read(args.detection)
    refs = read(args.reference)["entries"]
    output = prepare_evaluation_output(args.output)
    rng = random.Random(args.seed)
    sealed = {"seed": args.seed, "mapping": [], "sources": [], "inputRecords": []}
    for path in (args.landmarks, args.detection, args.reference, args.masks / "report.json"):
        sealed["inputRecords"].append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    manifests = {key: {"schemaVersion": 1, "kind": key, "cases": []} for key in ("landmarks", "detection", "masks")}
    for key in manifests:
        (output / key).mkdir()
    sealed["preparerSha256"] = file_sha256(__file__)
    sample_ids = sorted({r["sample_id"] for r in landmark["records"]})
    for index, sample_id in enumerate(sample_ids, 1):
        source = read(args.hardset / sample_id / "predictions.json")
        source_path = Path(source["source"])
        if file_sha256(source_path) != source["sha256"]:
            raise ValueError("Source image changed")
        original = Image.open(source_path).convert("RGB")
        records = [r for r in landmark["records"] if r["sample_id"] == sample_id]
        if any(r["source_sha256"] != source["sha256"] for r in records):
            raise ValueError("Prediction source mismatch")
        rng.shuffle(records)
        tokens = [f"L{index:02d}-{p}" for p in range(1, len(records) + 1)]
        rendered = []
        regional = {key: [] for key in ("image_left_eye", "image_right_eye", "inner_mouth")}
        for token, record in zip(tokens, records):
            sealed["mapping"].append({"kind": "landmarks", "case": index, "token": token, "model_id": record["model_id"], "sample_id": sample_id})
            prediction = record["prediction"]
            points = finite_points(prediction["points_original"])
            image = original.copy()
            for region, indices in prediction["point_definition"]["regions"].items():
                curve = points[indices]
                regional[region].append(curve)
                line(image, curve)
            rendered.append(image)
        x0, y0, x1, y1 = source["bbox"]
        padding = .12 * max(x1 - x0, y1 - y0)
        full = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
        crops = [full] + [region_crop(regional[r], original.size, h) for r, h in zip(regional, (180, 180, 220))]
        reference_image = original.copy()
        for ref in refs:
            if ref["case"] == sample_id and not ref.get("excluded"):
                line(reference_image, ref["points"], bool(ref.get("closed", True)))
        reference_entry = sheet([original, reference_image], ["SOURCE", "OLD REFERENCE"], crops, f"case-{index:02d}-reference.png", output / "landmarks")
        prediction_entry = sheet([original] + rendered, ["SOURCE"] + tokens, crops, f"case-{index:02d}-candidates.png", output / "landmarks")
        manifests["landmarks"]["cases"].append({"case": index, "reference": reference_entry, "candidates": prediction_entry, "tokens": tokens})
        sealed["sources"].append({"case": index, "sample_id": sample_id, "path": str(source_path), "sha256": source["sha256"]})
        detection_records = [r for r in detector["records"] if r["sample_id"] == sample_id]
        rng.shuffle(detection_records)
        d_tokens = [f"D{index:02d}-{p}" for p in range(1, len(detection_records) + 1)]
        d_images = [original]
        for token, record in zip(d_tokens, detection_records):
            if record["sourceSha256"] != source["sha256"]:
                raise ValueError("Detector source mismatch")
            image = original.copy()
            draw = ImageDraw.Draw(image)
            for number, detection in enumerate(record["prediction"]["detections"], 1):
                box = detection["box_xyxy"]
                draw.rectangle(box, outline=YELLOW, width=3)
                draw.text((box[0] + 4, box[1] + 4), str(number), font=FONT, fill=YELLOW)
            d_images.append(image)
            sealed["mapping"].append({"kind": "detection", "case": index, "token": token, "model_id": record["model_id"], "sample_id": sample_id})
        entry = sheet(d_images, ["SOURCE"] + d_tokens, [(0, 0, *original.size)], f"case-{index:02d}.png", output / "detection")
        manifests["detection"]["cases"].append({"case": index, "sheet": entry, "tokens": d_tokens})
        if (args.masks / sample_id).is_dir():
            original = Image.open(args.restoration / sample_id / "reference.png").convert("RGB")
            valid = np.asarray(Image.open(args.restoration / sample_id / "valid-mask.png").convert("L")) > 0
            base = np.asarray(original).copy()
            base[~valid] = (70, 70, 70)
            original = Image.fromarray(base)
            mask_files = [args.masks / sample_id / f"{m}.png" for m in ("xseg-wf", "bisenet-celebamaskhq", "sam2.1-hiera-large")]
            rng.shuffle(mask_files)
            m_tokens = [f"M{index:02d}-{p}" for p in range(1, len(mask_files) + 1)]
            m_images = [original]
            for token, path in zip(m_tokens, mask_files):
                mask = np.asarray(Image.open(path).convert("L")) > 127
                if mask.shape != valid.shape:
                    raise ValueError("Mask canvas mismatch")
                overlay = base.astype(float)
                overlay[mask & valid] = .70 * overlay[mask & valid] + .30 * np.array(YELLOW)
                # Boundary is a thin, contrasting curve; filling alone can hide leaks.
                import cv2
                boundary = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
                overlay[(boundary > 0) & valid] = YELLOW
                m_images.append(Image.fromarray(overlay.astype(np.uint8)))
                sealed["mapping"].append({"kind": "masks", "case": index, "token": token, "model_id": path.stem, "sample_id": sample_id, "mask_sha256": file_sha256(path)})
            entry = sheet(m_images, ["SOURCE"] + m_tokens, [(0, 0, *original.size)], f"case-{index:02d}.png", output / "masks")
            manifests["masks"]["cases"].append({"case": index, "sheet": entry, "tokens": m_tokens})
    for kind, manifest in manifests.items():
        atomic_json(output / kind / "manifest.json", manifest)
    atomic_json(output / "sealed-key.json", sealed)
    atomic_json(output / "protocol.json", {"schemaVersion": 1, "assessorModel": "gpt-6-astra", "reasoningEffort": "low", "modelIdentityHidden": True, "previousScoresHidden": True, "caseOrder": "all locked cases", "columnOrder": "independently shuffled for every case", "candidateLine": "yellow, native region geometry, 2 source pixels", "referencePolicy": "review old reference first; source is primary evidence; no imported exclusions or reasons", "score": {"scale": [0, 1, 2, 3, 4], "anchors": {"4": "visible contour follows anatomy; no clearly material deviation", "3": "mostly accurate with local minor deviation", "2": "repeated moderate displacement or partial structural error", "1": "major displacement over a substantial visible region", "0": "wrong structure/object or unusable visible-region prediction"}, "unknown": None, "aggregation": "eyes averaged where scorable, mouth equally weighted, cases equally weighted; ordinal mean times 25 is a visual-review index, not accuracy percent"}, "limitations": ["LLM visual review, not expert or human ground truth", "small fixed exploratory sample", "original crops can contain makeup, blur and occlusion", "mask semantics and alignment differ; no pooled mask winner", "no model default promotion from this review alone"]})
    print(json.dumps({"output": str(output), "cases": {k: len(v["cases"]) for k, v in manifests.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
