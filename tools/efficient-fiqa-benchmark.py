"""Small locked, source-preserving Efficient-FIQA versus current quality review.

Use prepare, then score. Give only blind/ to an independent assessor. Analysis
is a separate script and requires a locked assessor result before reading keys.
This is an exploratory perceptual-quality comparison, never identity accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "webui/python"), str(ROOT / "_internal/DeepFaceLab")]
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from DFLIMG import DFLIMG
from facelib import LandmarksProcessor
from dfl_asset_tool import bounded_image_metrics
from face_quality_review import quality_review

DEFAULT_OUT = ROOT / "workspace/.vision-evaluation/efficient-fiqa-20261007"
# Caller supplies a source list in ignored evaluation storage. Never bind the
# repository or its distributed scripts to this workstation's private media.
VARIANTS = ("original", "minor-blur", "moderate-blur", "severe-blur",
            "gaussian-noise", "jpeg-8", "dark", "clipped-bright")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_image(path):
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f"Unreadable image: {path}")
    return value


def variants(image, source_index):
    scale = min(image.shape[:2]) / 256
    rng = np.random.default_rng(97207 + source_index)
    _, compressed = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 8])
    return {
        "original": image.copy(),
        "minor-blur": cv2.GaussianBlur(image, (0, 0), .7 * scale),
        "moderate-blur": cv2.GaussianBlur(image, (0, 0), 2.2 * scale),
        "severe-blur": cv2.GaussianBlur(image, (0, 0), 4.5 * scale),
        "gaussian-noise": np.clip(image.astype(np.float32) + rng.normal(0, 16, image.shape), 0, 255).astype(np.uint8),
        "jpeg-8": cv2.imdecode(compressed, cv2.IMREAD_COLOR),
        "dark": np.rint(image.astype(np.float32) * .25).astype(np.uint8),
        "clipped-bright": np.clip(image.astype(np.float32) * 1.8 + 60, 0, 255).astype(np.uint8),
    }


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def panel(out, source_items, landmarks, group_id):
    """Full face plus fixed same-source eye/mouth region, no model scores."""
    canvas = Image.new("RGB", (1536, 1344), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"{group_id}  |  Anonymous face quality review", font=font(26), fill="black")
    for index, item in enumerate(source_items):
        image = read_image(out / "blind/images" / item["file"])
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x, y = (index % 4) * 384, 62 + (index // 4) * 632
        draw.text((x + 12, y), item["id"], font=font(24), fill="black")
        canvas.paste(rgb.resize((360, 360), Image.Resampling.LANCZOS), (x + 12, y + 36))
        # Crop bounds depend only on the original geometry and are identical
        # for every variant. They are detail views, not additional samples.
        detail_points = np.asarray(landmarks)[[36, 39, 42, 45, 48, 54, 57]]
        low, high = np.min(detail_points, axis=0), np.max(detail_points, axis=0)
        margin = np.array([.12, .10]) * np.maximum(high - low, 20)
        low = np.maximum(np.floor(low - margin).astype(int), 0)
        high = np.minimum(np.ceil(high + margin).astype(int), [image.shape[1], image.shape[0]])
        crop = rgb.crop((int(low[0]), int(low[1]), int(high[0]), int(high[1])))
        crop.thumbnail((360, 210), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x + 12 + (360 - crop.width) // 2, y + 408))
    filename = f"{group_id}.png"
    canvas.save(out / "blind/panels" / filename)
    return filename


def prepare(out, sources):
    if (out / "sealed-key.json").exists() or (out / "blind").exists():
        raise ValueError("Refusing to replace a prepared or reviewed benchmark")
    for directory in ("blind/images", "blind/panels", "original-bytes"):
        (out / directory).mkdir(parents=True, exist_ok=False)
    source_records, key_items, groups = [], [], []
    rng = random.Random(61007081)
    for source_index, source in enumerate(sources):
        image, dfl = read_image(source), DFLIMG.load(source)
        if dfl is None or not dfl.has_data():
            raise ValueError(f"Not a real DFL aligned source: {source}")
        landmarks = np.asarray(dfl.get_landmarks(), dtype=np.float32)
        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            raise ValueError("Need the source's original valid aligned geometry")
        sid = f"source-{source_index + 1:02d}"
        group = f"G{source_index + 1:02d}"
        digest = sha(source)
        copy = out / "original-bytes" / (sid + source.suffix)
        shutil.copyfile(source, copy)
        pose = np.rad2deg(LandmarksProcessor.estimate_pitch_yaw_roll(landmarks, size=image.shape[0])).tolist()
        source_records.append({"id": sid, "path": str(source.resolve()), "sha256": digest,
                               "copiedByteSha256": sha(copy), "sourceFilename": dfl.get_source_filename(),
                               "shape": list(image.shape), "brightness": float(image.mean() / 255),
                               "existing68PoseDegrees": pose,
                               "poseLimitation": "Existing 68-point PnP estimate; not angle ground truth; some roll values saturate"})
        items = list(variants(image, source_index).items())
        rng.shuffle(items)
        public_items = []
        for index, (kind, pixels) in enumerate(items):
            anonymous_id = f"{group}-{chr(65 + index)}"
            filename = anonymous_id + ".png"
            target = out / "blind/images" / filename
            cv2.imencode(".png", pixels)[1].tofile(target)
            item = {"id": anonymous_id, "file": filename, "sha256": sha(target)}
            public_items.append(item)
            key_items.append({**item, "sourceId": sid, "variant": kind})
        groups.append({"id": group, "items": public_items,
                       "panel": panel(out, public_items, landmarks, group)})
        print(f"Prepared anonymous {group}", flush=True)
    protocol = {
        "schemaVersion": 1, "assessorModel": "gpt-6-astra", "reasoningEffort": "low",
        "scope": "Perceptual face image quality, training usability as an auxiliary judgement",
        "blind": True, "modelScoresHidden": True, "degradationNamesHidden": True,
        "primaryScore": {"field": "quality", "range": [0, 4], "step": .5,
                         "anchors": {"0": "Essential facial details unusable", "1": "Major detail, exposure or artifact problems",
                                     "2": "Noticeable problems; some eye/mouth detail remains", "3": "Good facial detail with minor problems",
                                     "4": "Clear, faithful visible eye/mouth/skin detail; no material pixel defect"}},
        "instructions": ["Inspect all anonymous images in each group, plus fixed eye/mouth detail views.",
                         "Rate each item independently; score 0-4 in increments of 0.5; ties are allowed.",
                         "Do not punish a profile, raised/lowered head, closed eye or open mouth merely for that state.",
                         "Do not infer real identity correctness, landmark accuracy or eventual training performance.",
                         "Also record trainingUsable as yes/borderline/no, shortReason and group ranking with ties.",
                         "Only read files inside blind/. Never inspect source paths, sealed keys, model scores or tools source.",
                         "Write a result with protocolSha256, manifestSha256, groups each containing id/items with id/quality/trainingUsable/shortReason.",
                         "Keep pixels and metadata paths anonymous until the result file has been SHA-locked."],
        "limitations": ["One LLM visual assessor; not human expert ground truth", "Eight originals, seven controlled variants each; source video correlation",
                        "Controlled defect direction is a sanity check, not a visual-accuracy label", "Full panels use display resizing; individual anonymous PNGs preserve source resolution"],
        "predeclaredPromotionRule": {
            "primary": "Mean within-source Spearman against locked quality scores",
            "minimumMeanRhoGain": .10, "minimumWithinSourcePairAgreementGain": .10,
            "minimumImprovedSourceGroups": 2, "maximumSubstantiallyWorsenedSourceGroups": 1,
            "substantialRhoDifference": .10,
            "requirePositiveSourceBootstrap95LowerGain": True,
            "severeInversionVeto": "Efficient prefers severe blur or clipped-bright over its original while current and visual assessor prefer the original",
            "defaultWithoutClearBenefit": "Keep current quality; prioritize global representative subset selection",
            "bootstrapLimitation": "Exploratory source-group bootstrap; correlated sources, not a formal population confidence guarantee"},
    }
    write_json(out / "blind/protocol.json", protocol)
    write_json(out / "blind/manifest.json", {"schemaVersion": 1, "groups": groups})
    write_json(out / "sealed-key.json", {"schemaVersion": 1, "sources": source_records, "items": key_items,
               "variantParameters": {"blurSigmaAt256": [.7, 2.2, 4.5], "noiseStd": 16, "jpegQuality": 8, "darkGain": .25, "brightGain": 1.8, "brightOffset": 60},
               "inputBytesNeverChanged": True})
    lock = {"protocolSha256": sha(out / "blind/protocol.json"), "manifestSha256": sha(out / "blind/manifest.json"),
            "sealedKeySha256": sha(out / "sealed-key.json"), "originalsUnchanged": all(sha(record["path"]) == record["sha256"] for record in source_records),
            "sourceCount": len(sources), "caseCount": len(key_items), "trainingPerformed": False}
    if not lock["originalsUnchanged"]:
        raise RuntimeError("Original source changed")
    write_json(out / "prepare-lock.json", lock)
    print(json.dumps({"blindDirectory": str(out / "blind"), **lock}), flush=True)


def score(out, assets_root):
    from vision_fiqa import make_quality_scorer
    if (out / "sealed-scores.json").exists():
        raise ValueError("Refusing to replace locked model scores")
    lock = json.loads((out / "prepare-lock.json").read_text(encoding="utf-8"))
    if sha(out / "sealed-key.json") != lock["sealedKeySha256"]:
        raise ValueError("Prepared key changed")
    key = json.loads((out / "sealed-key.json").read_text(encoding="utf-8"))
    sources = {record["id"]: record for record in key["sources"]}
    records, start = [], time.monotonic()
    cv2.setNumThreads(2)
    with make_quality_scorer(device="cpu", assets_root=assets_root) as fiqa:
        for index, item in enumerate(key["items"]):
            source = sources[item["sourceId"]]
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Original source changed")
            dfl = DFLIMG.load(Path(source["path"]))
            image = read_image(out / "blind/images" / item["file"])
            if sha(out / "blind/images" / item["file"]) != item["sha256"]:
                raise ValueError("Anonymous image changed")
            classical = quality_review(image, bounded_image_metrics(image, dfl.get_xseg_mask()), dfl,
                                       pose_estimator=LandmarksProcessor.estimate_pitch_yaw_roll)
            model = fiqa(image, dfl.get_dict())
            records.append({"id": item["id"], "current": {"score": classical["score"] * 100,
                            "method": classical["method"], "components": classical["components"],
                            "detailCandidates": classical["detailCandidates"]}, "efficientFiqa": model})
            print(f"Scored anonymous {index + 1}/{len(key['items'])}", flush=True)
    output = {"schemaVersion": 1, "items": records, "elapsedSeconds": time.monotonic() - start,
              "primaryComparison": "Within-source order of current aggregate quality vs Efficient-FIQA perceptual quality",
              "currentMetric": "Existing quality_review aggregate with foreground Tenengrad, exposure, source resolution and alignment consistency",
              "unfairComparisonsAvoided": "No cross-source resolution advantage used in primary outcome; do not call sharpness alone the current aggregate",
              "originalsUnchanged": all(sha(record["path"]) == record["sha256"] for record in key["sources"]),
              "trainingPerformed": False}
    if not output["originalsUnchanged"]:
        raise RuntimeError("Original source changed")
    write_json(out / "sealed-scores.json", output)
    lock["sealedScoresSha256"] = sha(out / "sealed-scores.json")
    write_json(out / "score-lock.json", lock)
    print(json.dumps({"scoredCount": len(records), "elapsedSeconds": output["elapsedSeconds"], "sealedScoresSha256": lock["sealedScoresSha256"], "originalsUnchanged": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sources-json", type=Path)
    parser.add_argument("--assets-root", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.sources_json is None:
            parser.error("prepare requires --sources-json, a caller-supplied JSON array of read-only real aligned paths")
        sources = [Path(item) for item in json.loads(args.sources_json.read_text(encoding="utf-8"))]
        if not 5 <= len(sources) <= 12:
            raise ValueError("This bounded exploratory review requires 5-12 real originals")
        prepare(args.output.resolve(), sources)
    else:
        score(args.output.resolve(), args.assets_root)


if __name__ == "__main__":
    main()
