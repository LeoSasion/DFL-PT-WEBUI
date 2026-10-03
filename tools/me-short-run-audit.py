"""Read-only, at-most-one-hour ME development convergence audit.

Run after a safe stop. Loss trend and finite predictions are development
evidence; this tool never accepts final visual quality or a formal long run.
Only an explicitly requested JSON report is written.
"""

import argparse
from collections import deque
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAX_RUNTIME_SECONDS = 3600.0
sys.path.insert(0, str(ROOT / "_internal" / "DeepFaceLab"))
sys.path.insert(0, str(ROOT / "webui" / "python"))


def _timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}: missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label}: invalid timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label}: timestamp has no timezone")
    return parsed


def _finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _sha256(target):
    digest = hashlib.sha256()
    with Path(target).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_history(target, checkpoint_iteration, window_size=20, minimum_relative_drop=0.01):
    """Validate all records, then compare independent early/late median windows."""
    if type(window_size) is not int or not 5 <= window_size <= 1000:
        raise ValueError("window size must be an integer between 5 and 1000")
    if (not _finite_nonnegative(minimum_relative_drop)
            or not 0 < minimum_relative_drop <= 1):
        raise ValueError("minimum relative drop must be positive and at most 1")
    first = previous = None
    early, late = [], deque(maxlen=window_size)
    count = 0
    step_seconds = largest_gap = 0.0
    with Path(target).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            label = f"loss history line {line_number}"
            if not line.strip() or len(line) > 1_000_000:
                raise ValueError(f"{label}: empty or oversized record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{label}: invalid JSON") from error
            if not isinstance(record, dict) or type(record.get("iteration")) is not int:
                raise ValueError(f"{label}: invalid iteration")
            iteration = record["iteration"]
            if iteration < 1 or (previous and iteration != previous["iteration"] + 1):
                raise ValueError(f"{label}: iteration gap or duplicate")
            losses = (record.get("src_loss"), record.get("dst_loss"))
            seconds = record.get("seconds")
            if any(not _finite_nonnegative(value) for value in (*losses, seconds)):
                raise ValueError(f"{label}: nonfinite or negative loss/duration")
            at = _timestamp(record.get("at"), label)
            if previous:
                gap = (at - previous["at"]).total_seconds()
                if gap < 0:
                    raise ValueError(f"{label}: timestamp moved backwards")
                largest_gap = max(largest_gap, gap)
            else:
                first = {"iteration": iteration, "at": at, "seconds": seconds}
            previous = {"iteration": iteration, "at": at}
            if len(early) < window_size:
                early.append(losses)
            late.append(losses)
            step_seconds += seconds
            if not math.isfinite(step_seconds):
                raise ValueError(f"{label}: total step duration is nonfinite")
            count += 1
    if not count:
        raise ValueError("loss history is empty")
    if first["iteration"] != 1:
        raise ValueError("isolated development history must start at iteration 1")
    if previous["iteration"] != checkpoint_iteration:
        raise ValueError("checkpoint iteration and final loss record differ; audit after a safe stop")

    early_medians = {side: float(median(row[index] for row in early))
                     for index, side in enumerate(("src", "dst"))}
    late_medians = {side: float(median(row[index] for row in late))
                    for index, side in enumerate(("src", "dst"))}
    # Half-sums avoid overflowing for large, but finite, individual losses.
    early_overall = early_medians["src"] / 2 + early_medians["dst"] / 2
    late_overall = late_medians["src"] / 2 + late_medians["dst"] / 2
    reduction = ((early_overall - late_overall) / early_overall
                 if early_overall > 0 else None)
    enough = count >= 2 * window_size
    nonincreasing = {side: late_medians[side] <= early_medians[side]
                     for side in ("src", "dst")}
    demonstrated = (enough and all(nonincreasing.values()) and reduction is not None
                    and reduction >= minimum_relative_drop)
    reasons = []
    if not enough:
        reasons.append(f"need {2 * window_size} records for two nonoverlapping windows")
    if not all(nonincreasing.values()):
        reasons.append("at least one SRC/DST median increased")
    if reduction is None or reduction < minimum_relative_drop:
        reasons.append("combined median loss reduction is below the configured threshold")
    first_step_started = first["at"] - timedelta(seconds=first["seconds"])
    return {
        "path": str(Path(target).resolve()), "records": count,
        "firstIteration": first["iteration"], "lastIteration": previous["iteration"],
        "firstAt": first["at"].isoformat(), "lastAt": previous["at"].isoformat(),
        "firstStepStartedAt": first_step_started.isoformat(),
        "wallSpanSeconds": (previous["at"] - first["at"]).total_seconds(),
        "historyEnvelopeSeconds": (previous["at"] - first_step_started).total_seconds(),
        "stepComputeSeconds": step_seconds, "largestGapSeconds": largest_gap,
        "trend": {
            "status": "passed" if demonstrated else "inconclusive",
            "requestedWindowSize": window_size,
            "earlySampleCount": len(early), "lateSampleCount": len(late),
            "windowsNonoverlapping": enough,
            "earlyMedianLoss": early_medians, "lateMedianLoss": late_medians,
            "sideNonincreasing": nonincreasing,
            "combinedRelativeReduction": reduction,
            "minimumRelativeReduction": minimum_relative_drop,
            "reason": "; ".join(reasons) if reasons else "both medians non-increasing with overall loss reduction",
        },
    }


def inspect_runtime(history, started_at=None, stopped_at=None, maximum_seconds=MAX_RUNTIME_SECONDS,
                    allowance_seconds=0.0, saved_at=None):
    """Keep offline gaps and all declared load/save overhead inside the hard cap.

    An allowance is reserved headroom, never permission to exceed one hour.
    Loss timestamps alone cannot prove the process's loading/stop duration.
    """
    if (not _finite_nonnegative(maximum_seconds) or not 0 < maximum_seconds <= MAX_RUNTIME_SECONDS
            or not _finite_nonnegative(allowance_seconds) or allowance_seconds >= maximum_seconds):
        raise ValueError("runtime maximum must be positive and <=3600s; allowance must fit inside it")
    if bool(started_at) != bool(stopped_at):
        raise ValueError("provide both run-started-at and run-stopped-at")
    envelope = max(history["historyEnvelopeSeconds"], history["stepComputeSeconds"])
    details = {"maximumSeconds": maximum_seconds, "reservedAllowanceSeconds": allowance_seconds,
               "historyMeasuredSeconds": envelope, "offlineGapsIncluded": True,
               "measurementProvenance": "history timestamps; outer process timestamps are user-supplied"}
    if envelope + allowance_seconds > maximum_seconds:
        return {"status": "failed", "reason": "history duration plus reserved allowance exceeds the cap",
                **details}
    if not started_at:
        return {"status": "inconclusive", "reason": "outer process start/safe-stop timestamps are required to include loading and saving",
                **details}
    start = _timestamp(started_at, "run start")
    stop = _timestamp(stopped_at, "run stop")
    first = _timestamp(history["firstStepStartedAt"], "first step start")
    last = _timestamp(history["lastAt"], "last step completion")
    if not start <= first <= last <= stop:
        raise ValueError("outer process timestamps do not contain the complete loss history")
    if saved_at is not None and not last <= _timestamp(saved_at, "metadata savedAt") <= stop:
        raise ValueError("metadata save timestamp is outside the final step/safe-stop interval")
    measured = (stop - start).total_seconds()
    details.update({"runStartedAt": start.isoformat(), "runStoppedAt": stop.isoformat(),
                    "measuredWallSeconds": measured,
                    "measuredPlusAllowanceSeconds": measured + allowance_seconds})
    if measured + allowance_seconds > maximum_seconds:
        return {"status": "failed", "reason": "outer process duration plus reserved allowance exceeds the cap",
                **details}
    if history["stepComputeSeconds"] > measured:
        raise ValueError("reported step durations exceed the outer process duration")
    return {"status": "passed", **details}


def inspect_checkpoint(directory, device, expected_name=None):
    from me_backend.config import MEConfig
    from me_backend.engine import MEEngine
    from me_backend.web_bridge import validate_model_name

    directory = Path(directory)
    engine = MEEngine.load(directory / "me.pt", device)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("ME metadata must be an object")
    name = validate_model_name(metadata.get("name"))
    if expected_name is not None and name != validate_model_name(expected_name):
        raise ValueError("ME metadata name does not match the expected model identity")
    if (not isinstance(metadata, dict) or metadata.get("format") != "me-pytorch"
            or metadata.get("version") != 1 or metadata.get("model_class") != "ME"
            or metadata.get("checkpoint") != "me.pt"
            or type(metadata.get("iteration")) is not int or metadata["iteration"] != engine.iteration
            or MEConfig.from_dict(metadata.get("config")) != engine.config):
        raise ValueError("ME metadata does not match the checkpoint")
    return engine, metadata, {
        "name": name, "iteration": engine.iteration,
        "optimizerUpdates": engine.optimizer_updates, "config": engine.config.to_dict(),
    }


def _check_prediction(value, shape, label):
    import numpy as np

    if tuple(value.shape) != shape or not np.isfinite(value).all():
        raise ValueError(f"{label}: invalid shape or nonfinite prediction")
    low, high = float(value.min()), float(value.max())
    if low < -1e-4 or high > 1.0001:
        raise ValueError(f"{label}: prediction outside normalized image range")
    return {"shape": list(shape), "min": low, "max": high}


def inspect_predictions(engine, source, destination, count=1):
    """Deterministic selection with content hashes, without writing a manifest."""
    import numpy as np
    from me_backend.data import read_aligned
    from pose_probe_contract import probe_dataset_inventory

    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("sample count must be 1..3 per side")
    resolution = engine.config.resolution
    result, cache = {}, {}
    for side, dataset in (("src", source), ("dst", destination)):
        path = Path(dataset).resolve()
        if path not in cache:
            cache[path] = probe_dataset_inventory(path)
        kind, fingerprint, images = cache[path]
        if not images:
            raise ValueError(f"{side} aligned dataset has no JPG samples")
        amount = min(count, len(images))
        indexes = [0] if amount == 1 else [round(i * (len(images) - 1) / (amount - 1))
                                          for i in range(amount)]
        samples = []
        for index in indexes:
            reference = images[index]
            image, _, _ = read_aligned(reference.packed_sample or reference.path, engine.config)
            batch = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError(f"{side}: nonfinite aligned prediction input")
            swap, src_mask, dst_mask = engine.predict(batch)
            samples.append({
                "member": reference.member, "sha256": reference.sha256,
                "swap": _check_prediction(swap, (1, 3, resolution, resolution), "swap"),
                "sourceMask": _check_prediction(src_mask, (1, 1, resolution, resolution), "source mask"),
                "destinationMask": _check_prediction(dst_mask, (1, 1, resolution, resolution), "destination mask"),
            })
            current = (hashlib.sha256(reference.packed_sample.read_raw_file()).hexdigest()
                       if reference.packed_sample else _sha256(reference.path))
            if current != reference.sha256:
                raise ValueError(f"{side}: prediction sample changed during the audit")
        result[side] = {"datasetKind": kind, "datasetFingerprint": fingerprint,
                        "inventoryCount": len(images), "sampleCount": amount, "samples": samples}
    return result


def audit(args):
    model = Path(args.model).resolve()
    history_path = Path(args.history).resolve() if args.history else model / "loss-history.jsonl"
    report = {"schemaVersion": 1, "scope": "short development convergence only",
              "model": str(model), "materialLabel": args.material,
              "materialProvenance": "user-declared", "checks": {}}
    context = {}
    inputs = (model / "me.pt", model / "metadata.json", history_path)
    identities = {}
    for target in inputs:
        try:
            identities[target] = _sha256(target)
        except OSError:
            pass  # The corresponding detailed check reports missing input.

    def check(name, function, dependencies=()):
        if any(report["checks"].get(dep, {}).get("status") != "passed" for dep in dependencies):
            report["checks"][name] = {"status": "skipped", "reason": "prerequisite failed"}
            return
        try:
            report["checks"][name] = {"status": "passed", "details": function()}
        except Exception as error:
            report["checks"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:1000]}

    def checkpoint():
        context["engine"], context["metadata"], details = inspect_checkpoint(model, args.device,
                                                                           args.expected_model_name)
        details["sha256"] = identities.get(model / "me.pt")
        return details

    def history():
        details = inspect_history(history_path, context["engine"].iteration,
                                  args.window_size, args.minimum_relative_drop)
        context["history"] = details
        return details

    check("checkpoint", checkpoint)
    check("history", history, ("checkpoint",))
    if "history" in context:
        try:
            report["checks"]["runtime"] = inspect_runtime(
                context["history"], args.run_started_at, args.run_stopped_at,
                args.max_runtime_seconds, args.runtime_allowance_seconds,
                context["metadata"].get("savedAt"))
        except Exception as error:
            report["checks"]["runtime"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:1000]}
        report["checks"]["convergenceTrend"] = context["history"]["trend"]
    else:
        report["checks"]["runtime"] = report["checks"]["convergenceTrend"] = {
            "status": "skipped", "reason": "history prerequisite failed"}
    check("predictions", lambda: inspect_predictions(context["engine"], args.src, args.dst,
                                                      args.sample_count), ("checkpoint",))

    def input_identity():
        if len(identities) != len(inputs):
            raise ValueError("required input was missing at audit start")
        for target, before in identities.items():
            if _sha256(target) != before:
                raise ValueError(f"{target.name} changed during the audit; rerun after a safe stop")
        return {"inputSha256": {str(target): digest for target, digest in identities.items()},
                "inputsUnchanged": True}

    check("inputIdentity", input_identity)
    checks = report["checks"]
    technical = all(checks[name]["status"] == "passed"
                    for name in ("checkpoint", "history", "predictions", "inputIdentity"))
    passed = technical and all(checks[name]["status"] == "passed"
                              for name in ("runtime", "convergenceTrend"))
    failed = any(value["status"] == "failed" for value in checks.values())
    report["outcome"] = {
        "status": "passed" if passed else "failed" if failed else "inconclusive",
        "technicalPassed": technical, "shortDevelopmentPassed": passed,
        "longRunEvidenceQualified": False, "visualQualityAccepted": False,
        "deepFaceLiveAccepted": False,
    }
    return report


def _atomic_json(target, report):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_report_path(args):
    if not args.json_out:
        return
    output = Path(args.json_out).resolve()
    model = Path(args.model).resolve()
    protected = [model / "me.pt", model / "metadata.json",
                 Path(args.history).resolve() if args.history else model / "loss-history.jsonl"]
    protected.extend(Path(value).resolve() for value in (args.src, args.dst))
    for target in protected:
        if output == target or (output.exists() and target.exists() and output.samefile(target)):
            raise ValueError("audit report cannot replace a checkpoint, metadata, history or dataset input")
    for value in (args.src, args.dst):
        dataset = Path(value).resolve()
        if dataset.is_dir() and output.is_relative_to(dataset):
            raise ValueError("audit report cannot be written inside an aligned dataset")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-name", help="Optional expected metadata model identity; independent of folder name")
    parser.add_argument("--src", required=True, help="Aligned directory or explicit PAK/ZIP")
    parser.add_argument("--dst", required=True, help="Aligned directory or explicit PAK/ZIP")
    parser.add_argument("--material", choices=("real", "synthetic"), required=True)
    parser.add_argument("--history", help="Override loss-history.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=1, help="1..3 deterministic samples per side")
    parser.add_argument("--window-size", type=int, default=20, help="5..1000 records per independent median window")
    parser.add_argument("--minimum-relative-drop", type=float, default=0.01, help="Positive combined median reduction fraction")
    parser.add_argument("--run-started-at", help="Outer process start ISO timestamp, with timezone")
    parser.add_argument("--run-stopped-at", help="Completed safe stop ISO timestamp, with timezone")
    parser.add_argument("--max-runtime-seconds", type=float, default=MAX_RUNTIME_SECONDS, help="Hard cap, never above 3600")
    parser.add_argument("--runtime-allowance-seconds", type=float, default=0, help="Headroom reserved inside the cap, never extra permitted time")
    parser.add_argument("--json-out", help="Optional report; the only written file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _validate_report_path(args)
    report = audit(args)
    if args.json_out:
        _atomic_json(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"passed": 0, "failed": 1, "inconclusive": 2}[report["outcome"]["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
