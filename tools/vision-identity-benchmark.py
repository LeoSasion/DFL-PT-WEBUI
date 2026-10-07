"""Local identity quality observations; no automatic model selection.

Manifest samples must record label verification or leave identity_label null.
Source media remains read-only. AdaFace's three demonstration images have no
published class-label file, so their apparent pair relationships are provisional.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui/python"))
from vision_identity import (DEFAULT_ASSETS, IdentityPredictor, SFACE_PATH,
                             SFaceTorch, file_sha256, normalized, verified_identity)
from vision_assets import prepare_evaluation_output


def labelled_pairs(predictions, samples):
    labels = {sample["id"]: sample.get("identity_label") for sample in samples}
    hashes = {sample["id"]: sample.get("sha256") for sample in samples}
    statuses = {sample["id"]: sample.get("label_status", "provisional") for sample in samples}
    pairs = []
    keys, seen_hashes = [], set()
    for key in sorted(predictions):
        digest = hashes[key]
        if digest and digest in seen_hashes:
            continue
        keys.append(key)
        if digest:
            seen_hashes.add(digest)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            a, b = labels[left], labels[right]
            if a is None or b is None:
                continue
            pairs.append({"left": left, "right": right, "same_identity": a == b,
                          "label_status": "verified" if statuses[left] == statuses[right] == "verified" else "provisional",
                          "cosine": float(np.dot(predictions[left], predictions[right]))})
    return pairs


def equivalence(crops):
    """OpenCV DNN is an evaluation reference only, never the product backend."""
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    reference = cv2.dnn.readNetFromONNX(str(SFACE_PATH))
    reference.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    reference.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    model = SFaceTorch(device="cpu")
    cases = dict(crops)
    cases.update(zero=np.zeros((112, 112, 3), np.uint8),
                 white=np.full((112, 112, 3), 255, np.uint8),
                 random=np.random.default_rng(20261006).integers(0, 256, (112, 112, 3), dtype=np.uint8))
    rows, ref_vectors, torch_vectors = [], [], []
    for name, rgb in cases.items():
        bgr = rgb[:, :, ::-1].copy()
        reference.setInput(cv2.dnn.blobFromImage(bgr, 1, (112, 112), (0, 0, 0), True, False))
        a = reference.forward().reshape(-1)
        b = model.embedding_bgr(bgr)
        an, bn = normalized(a), normalized(b)
        rows.append({"sample": name, "raw_max_abs": float(np.max(np.abs(a - b))),
                     "normalized_max_abs": float(np.max(np.abs(an - bn))), "cosine": float(an @ bn)})
        ref_vectors.append(an)
        torch_vectors.append(bn)
    pair_diff = float(np.max(np.abs(np.array(ref_vectors) @ np.array(ref_vectors).T -
                                         np.array(torch_vectors) @ np.array(torch_vectors).T)))
    passed = all(row["raw_max_abs"] <= 1e-4 and row["normalized_max_abs"] <= 1e-5
                 and row["cosine"] >= 1 - 1e-6 for row in rows) and pair_diff <= 1e-5
    return {"passed": passed, "tolerances": {"raw_max_abs": 1e-4, "normalized_max_abs": 1e-5,
            "minimum_cosine": 1 - 1e-6, "pair_cosine_max_abs": 1e-5}, "cases": rows,
            "pair_cosine_max_abs": pair_diff, "torch": torch.__version__, "opencv": cv2.__version__}


def official_samples(assets, output):
    directory = Path(assets) / "AdaFace"
    identity = verified_identity(directory, expected_model="AdaFace")
    face_dir = directory / "source/face_alignment"
    # Author MTCNN executes in PyTorch. Its five points are shared by all models.
    sys.path.insert(0, str(face_dir))
    try:
        import mtcnn
        from mtcnn_pytorch.src.matlab_cp2tform import get_similarity_transform_for_cv2
        detector = mtcnn.MTCNN(device="cpu", crop_size=(112, 112))
        samples, crops = [], {}
        for index in range(1, 4):
            path = face_dir / f"test_images/img{index}.jpeg"
            rgb = np.asarray(Image.open(path).convert("RGB"))
            boxes, landmarks = detector.detect_faces(Image.fromarray(rgb), detector.min_face_size,
                        detector.thresholds, detector.nms_thresholds, detector.factor)
            if not len(boxes):
                raise ValueError(f"Author alignment found no face in img{index}")
            points = np.stack((landmarks[0][:5], landmarks[0][5:]), axis=1).astype(np.float32)
            matrix = get_similarity_transform_for_cv2(points, detector.refrence.astype(np.float32))
            crop = cv2.warpAffine(rgb, matrix, (112, 112))
            name = f"img{index}"
            crop_path = output / f"{name}-shared-112.png"
            cv2.imwrite(str(crop_path), crop[:, :, ::-1])
            samples.append({"id": name, "image": str(path.resolve()), "sha256": file_sha256(path),
                            "aligned_image": str(crop_path.resolve()), "aligned_sha256": file_sha256(crop_path),
                            "source_to_aligned_affine": matrix.tolist(), "landmarks5": points.tolist(),
                            "identity_label": "example-person-A" if index < 3 else "example-person-B",
                            "label_status": "provisional",
                            "label_source": "interpretation of author demonstration; no published class-label file"})
            crops[name] = crop
        return {"samples": samples, "alignment": "official AdaFace Torch MTCNN, shared five-point similarity 112",
                "alignment_asset_identity": identity}, crops
    finally:
        sys.path.remove(str(face_dir))


def manifest_crops(manifest):
    samples = manifest.get("samples", [])
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Identity comparison requires nonempty samples with unique IDs")
    crops = {}
    for sample in samples:
        path = Path(sample["aligned_image"])
        if file_sha256(path) != sample["aligned_sha256"]:
            raise ValueError("Shared aligned sample hash mismatch")
        source = Path(sample["image"])
        if file_sha256(source) != sample["sha256"]:
            raise ValueError("Source sample hash mismatch")
        crops[sample["id"]] = np.asarray(Image.open(path).convert("RGB"))
    return crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-samples", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=["sface", "adaface-ir100-webface12m", "magface-iresnet100"])
    args = parser.parse_args()
    args.output = prepare_evaluation_output(args.output)
    if args.official_samples:
        manifest, crops = official_samples(args.assets, args.output)
    elif args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        crops = manifest_crops(manifest)
    else:
        parser.error("Specify --official-samples or --manifest")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    parity = equivalence(crops)
    results = {"sface_numerical_equivalence": parity, "models": {}, "errors": {},
               "quality_claim": "three author demonstration images; provisional pair interpretation; no winner selection",
               "far": None, "tar": None, "ranking": None,
               "pending": "More independent consented and verified identity labels required",
               "implementation_sha256": file_sha256(ROOT / "webui/python/vision_identity.py"),
               "benchmark_sha256": file_sha256(__file__)}
    for model_id in args.models:
        model = None
        try:
            model = IdentityPredictor(model_id, args.assets, args.device)
            predictions, robustness = {}, []
            for sample_index, (name, rgb) in enumerate(crops.items()):
                prediction = model.predict(rgb)
                # Caller IDs are metadata, never output paths.
                (args.output / f"{sample_index+1:03d}-{model.candidate_id}.json").write_text(
                    json.dumps(dict(sample_id=name, **prediction), indent=2) + "\n", encoding="utf-8")
                predictions[name] = prediction["embedding"]
                degraded = {
                    "blur_sigma_1.5": cv2.GaussianBlur(rgb, (0, 0), 1.5),
                    "downsample_28": cv2.resize(cv2.resize(rgb, (28, 28), interpolation=cv2.INTER_AREA), (112, 112)),
                    "jpeg_quality_35": cv2.imdecode(cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 35])[1], cv2.IMREAD_COLOR)[:, :, ::-1],
                }
                for degradation, image in degraded.items():
                    vector = model.predict(image)["embedding"]
                    robustness.append({"sample": name, "degradation": degradation,
                                       "cosine_to_same_original": float(np.dot(vector, prediction["embedding"]))})
            pairs = labelled_pairs(predictions, manifest["samples"])
            results["models"][model_id] = {"asset_identity": model.identity, "preprocessing": model.preprocessing,
                "pairs": pairs, "robustness_same_image_only": robustness,
                "far": None, "tar": None, "inferred_samples": len(predictions),
                "verified_identity_samples": sum(sample.get("label_status") == "verified" for sample in manifest["samples"]),
                "positive_pairs": sum(pair["same_identity"] for pair in pairs),
                "negative_pairs": sum(not pair["same_identity"] for pair in pairs)}
            print(model_id, json.dumps(pairs), flush=True)
        except Exception as error:
            results["errors"][model_id] = f"{type(error).__name__}: {error}"
            print(model_id, results["errors"][model_id], flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("sface equivalence", parity["passed"], "errors", len(results["errors"]), flush=True)
    return 1 if results["errors"] or not parity["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
