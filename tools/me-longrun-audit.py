"""Read-only ME long-run evidence audit; never claims visual quality acceptance.

The selected aligned samples are frozen by content fingerprint in a small
manifest beside the model. Run after a verified safe stop, before changing the
datasets or resuming training. Use --sample-manifest with a new path when
intentionally starting a new data phase.
"""

import argparse
from collections import deque
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_internal" / "DeepFaceLab"))
sys.path.insert(0, str(ROOT / "webui" / "python"))

from me_backend.config import MEConfig
from me_backend.data import read_aligned
from me_backend.engine import MEEngine
from me_backend.export import export_dfm
from pose_probe_contract import probe_dataset_inventory, sha256_file


MIN_LONG_RUN_HOURS = 12.0
DFM_INPUT = "in_face:0"
DFM_OUTPUTS = ("out_face_mask:0", "out_celeb_face:0", "out_celeb_face_mask:0")


def _timestamp(value, line_number):
    if not isinstance(value, str):
        raise ValueError(f"loss history line {line_number}: missing UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"loss history line {line_number}: invalid timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"loss history line {line_number}: timestamp has no timezone")
    return parsed


def inspect_history(target, checkpoint_iteration, max_gap_seconds=300.0):
    """Stream the entire JSONL history without holding long runs in memory."""
    first = previous = None
    first_losses = last_losses = None
    first_window, last_window = [], deque(maxlen=100)
    count = pauses = 0
    step_seconds = observed_seconds = largest_gap = 0.0
    with Path(target).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"loss history line {line_number}: empty record")
            if len(line) > 1_000_000:
                raise ValueError(f"loss history line {line_number}: record exceeds 1 MiB")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"loss history line {line_number}: invalid JSON") from error
            if not isinstance(record, dict) or type(record.get("iteration")) is not int:
                raise ValueError(f"loss history line {line_number}: invalid iteration")
            iteration = record["iteration"]
            if iteration < 1 or (previous and iteration != previous["iteration"] + 1):
                raise ValueError(f"loss history line {line_number}: iteration gap or duplicate")
            seconds = record.get("seconds")
            losses = (record.get("src_loss"), record.get("dst_loss"))
            if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds < 0
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) or value < 0 for value in losses)):
                raise ValueError(f"loss history line {line_number}: nonfinite loss or duration")
            at = _timestamp(record.get("at"), line_number)
            if previous:
                gap = (at - previous["at"]).total_seconds()
                if gap < 0:
                    raise ValueError(f"loss history line {line_number}: timestamp moved backwards")
                largest_gap = max(largest_gap, gap)
                if gap > max_gap_seconds:
                    pauses += 1
                    observed_seconds += seconds
                else:
                    # Includes loading and saving, but never a long offline gap.
                    observed_seconds += max(gap, seconds)
            else:
                first = {"iteration": iteration, "at": at}
                observed_seconds += seconds
            previous = {"iteration": iteration, "at": at}
            first_losses = first_losses or losses
            last_losses = losses
            if len(first_window) < 100:
                first_window.append(losses)
            last_window.append(losses)
            step_seconds += seconds
            count += 1
    if not count:
        raise ValueError("loss history is empty")
    if first["iteration"] != 1:
        raise ValueError("loss history does not start at iteration 1")
    if previous["iteration"] != checkpoint_iteration:
        raise ValueError("checkpoint iteration and final loss record differ; audit after a safe stop")
    start_mean = np.mean(np.asarray(first_window, dtype=np.float64), axis=0)
    end_mean = np.mean(np.asarray(last_window, dtype=np.float64), axis=0)
    return {
        "records": count,
        "firstIteration": first["iteration"],
        "lastIteration": previous["iteration"],
        "firstAt": first["at"].isoformat(),
        "lastAt": previous["at"].isoformat(),
        "wallSpanHours": round((previous["at"] - first["at"]).total_seconds() / 3600, 5),
        "stepComputeHours": round(step_seconds / 3600, 5),
        "observedActiveHours": round(observed_seconds / 3600, 5),
        "pauseCount": pauses,
        "largestGapSeconds": round(largest_gap, 3),
        "maxCountedGapSeconds": max_gap_seconds,
        "firstLoss": {"src": first_losses[0], "dst": first_losses[1]},
        "lastLoss": {"src": last_losses[0], "dst": last_losses[1]},
        "first100MeanLoss": {"src": float(start_mean[0]), "dst": float(start_mean[1])},
        "last100MeanLoss": {"src": float(end_mean[0]), "dst": float(end_mean[1])},
    }


def _atomic_json(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _selected_images(images, count):
    if not images:
        raise ValueError("aligned dataset contains no JPG samples")
    amount = min(len(images), count)
    indexes = [0] if amount == 1 else [round(index * (len(images) - 1) / (amount - 1))
                                        for index in range(amount)]
    return [images[index] for index in indexes]


def freeze_samples(source, destination, target, count=3):
    """Create once, then require identical SRC/DST content and member selection."""
    if not 1 <= count <= 12:
        raise ValueError("sample count must be 1..12 per side")
    inventories = {}
    cached = {}
    datasets = {}
    for side, dataset in (("src", source), ("dst", destination)):
        resolved = Path(dataset).resolve()
        if resolved not in cached:
            cached[resolved] = probe_dataset_inventory(resolved)
        kind, fingerprint, images = cached[resolved]
        selected = _selected_images(images, count)
        inventories[side] = (kind, fingerprint, {image.member: image for image in images})
        records = [{"member": image.member, "sha256": image.sha256} for image in selected]
        datasets[side] = {"kind": kind, "fingerprint": fingerprint,
                          "inventoryCount": len(images), "samples": records}
    expected = {"schemaVersion": 1, "sampleCountPerSide": count, "datasets": datasets}
    target = Path(target)
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("fixed sample manifest differs from current datasets; use a new manifest path for a new phase")
        created = False
    else:
        _atomic_json(target, expected)
        created = True
    selected_refs = {side: [inventories[side][2][item["member"]]
                            for item in datasets[side]["samples"]]
                     for side in ("src", "dst")}
    return selected_refs, {"path": str(target.resolve()), "created": created,
                           "datasets": datasets}


def _check_tensor(value, shape, name):
    if tuple(value.shape) != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    low, high = float(value.min()), float(value.max())
    if low < -1e-4 or high > 1.0001:
        raise ValueError(f"{name} leaves the normalized image range")
    return {"shape": list(shape), "min": low, "max": high}


def inspect_predictions(engine, selected):
    resolution = engine.config.resolution
    report = {}
    inputs = {"src": [], "dst": []}
    for side in ("src", "dst"):
        report[side] = []
        for image_ref in selected[side]:
            image, _, _ = read_aligned(image_ref.packed_sample or image_ref.path, engine.config)
            batch = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
            swap, source_mask, destination_mask = engine.predict(batch)
            details = {
                "member": image_ref.member,
                "sha256Prefix": image_ref.sha256[:16],
                "swap": _check_tensor(swap, (1, 3, resolution, resolution), "swap"),
                "sourceMask": _check_tensor(source_mask, (1, 1, resolution, resolution), "source mask"),
                "destinationMask": _check_tensor(destination_mask, (1, 1, resolution, resolution), "destination mask"),
            }
            report[side].append(details)
            inputs[side].append(image)
    return report, inputs


def inspect_dfm(engine, model_directory, destination_images, existing_dfm=None):
    import onnxruntime as ort

    with tempfile.TemporaryDirectory(prefix="me-longrun-dfm-") as temporary:
        if existing_dfm:
            dfm = Path(existing_dfm)
            if not dfm.is_file():
                raise FileNotFoundError(f"DFM does not exist: {dfm}")
        else:
            with redirect_stdout(io.StringIO()):
                dfm = export_dfm(model_directory, Path(temporary) / "model.dfm")
        session = ort.InferenceSession(str(dfm), providers=["CPUExecutionProvider"])
        if [item.name for item in session.get_inputs()] != [DFM_INPUT]:
            raise ValueError("DFM input contract differs from DeepFaceLive")
        if tuple(item.name for item in session.get_outputs()) != DFM_OUTPUTS:
            raise ValueError("DFM output contract differs from DeepFaceLive")
        # Exercise the exported dynamic batch axis even with a one-image smoke set.
        dfm_inputs = destination_images if len(destination_images) > 1 else destination_images * 2
        batch = np.ascontiguousarray(np.stack(dfm_inputs), dtype=np.float32)
        actual_destination, actual_swap, actual_source = session.run(None, {DFM_INPUT: batch})
        expected_swap, expected_source, expected_destination = engine.predict(
            batch.transpose(0, 3, 1, 2))
        expected = (expected_destination, expected_swap, expected_source)
        actual = (actual_destination, actual_swap, actual_source)
        deviations = {}
        for name, result, reference in zip(DFM_OUTPUTS, actual, expected):
            reference = reference.transpose(0, 2, 3, 1)
            if result.shape != reference.shape or not np.isfinite(result).all():
                raise ValueError(f"DFM output {name} has invalid shape or nonfinite values")
            deviations[name] = float(np.max(np.abs(result - reference)))
            if not np.allclose(result, reference, rtol=2e-4, atol=2e-6):
                raise ValueError(f"DFM output {name} differs from the PyTorch checkpoint")
        return {"file": str(dfm.resolve()) if existing_dfm else "temporary export",
                "batchSize": len(batch), "uniqueSampleCount": len(destination_images), "inputs": [DFM_INPUT],
                "outputs": list(DFM_OUTPUTS), "maxAbsDifference": deviations,
                "tolerance": {"rtol": 2e-4, "atol": 2e-6}}


def inspect_video(video, ffprobe, ffmpeg, timeout_seconds=600):
    target = Path(video)
    if not target.is_file():
        raise FileNotFoundError(f"video does not exist: {target}")
    metadata = subprocess.run([str(ffprobe), "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_name,codec_type,width,height,nb_frames", "-of", "json", str(target)],
        capture_output=True, text=True, timeout=60, check=True)
    payload = json.loads(metadata.stdout)
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if not streams:
        raise ValueError("output has no video stream")
    duration = float(payload.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("output video duration is invalid")
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise ValueError("output video dimensions are invalid")
    subprocess.run([str(ffmpeg), "-v", "error", "-xerror", "-i", str(target),
        "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True,
        timeout=timeout_seconds, check=True)
    return {"file": str(target.resolve()), "durationSeconds": duration,
            "width": stream["width"], "height": stream["height"],
            "codec": stream.get("codec_name"), "fullVideoDecode": True}


def _checkpoint(model_directory, device):
    directory = Path(model_directory)
    checkpoint = directory / "me.pt"
    before = sha256_file(checkpoint)
    engine = MEEngine.load(checkpoint, device)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    # Older ME metadata omits options that were added later with defaults.
    # Parse it with the same strict config rules used when loading the checkpoint.
    metadata_config = MEConfig.from_dict(metadata.get("config"))
    if (metadata.get("format") != "me-pytorch" or metadata.get("model_class") != "ME"
        or metadata.get("checkpoint") != "me.pt" or metadata.get("iteration") != engine.iteration
        or metadata_config != engine.config
        or metadata.get("name") != directory.name):
        raise ValueError("ME metadata does not match the checkpoint")
    return engine, {"name": metadata["name"], "iteration": engine.iteration,
                    "optimizerUpdates": engine.optimizer_updates, "sha256": before,
                    "resolution": engine.config.resolution, "architecture": engine.config.archi}


def audit(args):
    if args.minimum_hours < 0 or args.max_gap_seconds <= 0:
        raise ValueError("duration thresholds must be nonnegative and gap cap positive")
    model_directory = Path(args.model).resolve()
    sample_manifest = Path(args.sample_manifest).resolve() if args.sample_manifest else model_directory / "audit-samples.json"
    report = {"schemaVersion": 1, "materialLabel": args.material,
              "materialProvenance": "user-declared; content is not independently classified",
              "model": str(model_directory), "checks": {},
              "humanVisualReview": "pending", "deepFaceLiveReview": "pending"}
    context = {}

    def check(name, function, dependencies=()):
        if any(report["checks"].get(prerequisite, {}).get("status") != "passed"
               for prerequisite in dependencies):
            report["checks"][name] = {"status": "skipped", "reason": "prerequisite failed"}
            return
        try:
            report["checks"][name] = {"status": "passed", "details": function()}
        except Exception as error:
            report["checks"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:1000]}

    def checkpoint_check():
        context["engine"], details = _checkpoint(model_directory, args.device)
        context["checkpoint_sha256"] = details["sha256"]
        return details

    def history_check():
        target = Path(args.history) if args.history else model_directory / "loss-history.jsonl"
        details = inspect_history(target, context["engine"].iteration, args.max_gap_seconds)
        details["path"] = str(target.resolve())
        return details

    def samples_check():
        context["samples"], details = freeze_samples(args.src, args.dst, sample_manifest, args.sample_count)
        return details

    def prediction_check():
        details, inputs = inspect_predictions(context["engine"], context["samples"])
        context["inputs"] = inputs
        return details

    check("checkpoint", checkpoint_check)
    check("history", history_check, ("checkpoint",))
    check("fixedSamples", samples_check, ("checkpoint",))
    check("predictions", prediction_check, ("checkpoint", "fixedSamples"))
    check("dfm", lambda: inspect_dfm(context["engine"], model_directory,
        context["inputs"]["dst"], args.dfm), ("checkpoint", "predictions"))
    if args.video:
        check("video", lambda: inspect_video(args.video, args.ffprobe, args.ffmpeg,
            args.video_timeout_seconds))
    else:
        report["checks"]["video"] = {"status": "not-provided"}

    if context.get("checkpoint_sha256") and sha256_file(model_directory / "me.pt") != context["checkpoint_sha256"]:
        report["checks"]["checkpoint"] = {"status": "failed",
            "error": "checkpoint changed during the audit; rerun after a safe stop"}
    required = ("checkpoint", "history", "fixedSamples", "predictions", "dfm")
    if args.video:
        required += ("video",)
    technical_pass = all(report["checks"][name]["status"] == "passed" for name in required)
    history = report["checks"]["history"]
    active_hours = history.get("details", {}).get("observedActiveHours", 0.0)
    duration_pass = history["status"] == "passed" and active_hours >= args.minimum_hours
    long_run_evidence = (technical_pass and args.material == "real"
                         and args.minimum_hours >= MIN_LONG_RUN_HOURS
                         and active_hours >= args.minimum_hours)
    report["outcome"] = {"technicalPassed": technical_pass,
        "durationThresholdHours": args.minimum_hours,
        "durationThresholdPassed": duration_pass,
        "longRunEvidenceQualified": long_run_evidence,
        "visualQualityAccepted": False, "deepFaceLiveAccepted": False}
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ME model directory containing me.pt and metadata.json")
    parser.add_argument("--src", required=True, help="Fixed SRC aligned directory or explicit PAK/ZIP")
    parser.add_argument("--dst", required=True, help="Fixed DST aligned directory or explicit PAK/ZIP")
    parser.add_argument("--material", choices=("synthetic", "real"), required=True,
                        help="Explicit user-declared material provenance; never inferred")
    parser.add_argument("--history", help="Override loss-history.jsonl path")
    parser.add_argument("--sample-manifest", help="Path for the frozen sample identity manifest")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--minimum-hours", type=float, default=MIN_LONG_RUN_HOURS)
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    parser.add_argument("--device", default="cpu", help="Use cpu or an available CUDA device")
    parser.add_argument("--dfm", help="Existing DFM to compare; otherwise export to a temporary file")
    parser.add_argument("--video", help="Optional completed video for full decode verification")
    parser.add_argument("--ffprobe", default=str(ROOT / "_internal" / "ffmpeg" / "ffprobe.exe"))
    parser.add_argument("--ffmpeg", default=str(ROOT / "_internal" / "ffmpeg" / "ffmpeg.exe"))
    parser.add_argument("--video-timeout-seconds", type=int, default=600)
    parser.add_argument("--json-out", help="Optional durable JSON report destination")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_directory = Path(args.model).resolve()
    protected = {model_directory / "me.pt", model_directory / "metadata.json",
                 Path(args.history).resolve() if args.history else model_directory / "loss-history.jsonl"}
    protected.update(Path(value).resolve() for value in (args.dfm, args.video) if value)
    for value in (args.sample_manifest, args.json_out):
        if value and Path(value).resolve() in protected:
            raise ValueError("audit output cannot replace a checkpoint, metadata, history, DFM or video input")
    effective_manifest = Path(args.sample_manifest).resolve() if args.sample_manifest else model_directory / "audit-samples.json"
    if args.json_out and Path(args.json_out).resolve() == effective_manifest:
        raise ValueError("report and sample manifest must use different files")
    report = audit(args)
    if args.json_out:
        _atomic_json(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report["outcome"]["technicalPassed"]
                 and report["outcome"]["durationThresholdPassed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
