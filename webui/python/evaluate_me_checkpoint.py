"""Evaluate an existing ME checkpoint against an existing, fixed pose manifest.

This command never calls the training or checkpoint-save APIs. All durable output
is placed below the explicitly selected evaluation root.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "_internal" / "DeepFaceLab"))

from me_backend.engine import MEEngine  # noqa: E402
from me_backend.web_bridge import evaluate  # noqa: E402
from training_evaluation import MAX_ONLINE_SNAPSHOTS, load_evaluation_manifest  # noqa: E402


SNAPSHOT_ID = re.compile(r"^iter-\d{8,12}-(?:[a-f0-9]{8}|[a-f0-9]{24})$")


def sha256_file(target):
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains(parent, candidate):
    return candidate == parent or parent in candidate.parents


def checkpoint_path(model):
    selected = Path(model).resolve(strict=True)
    checkpoint = selected / "me.pt" if selected.is_dir() else selected
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise ValueError("--model must select an ME .pt checkpoint or a directory containing me.pt")
    return checkpoint.resolve(strict=True)


def validate_paths(checkpoint, src, dst, evaluation_root):
    """Reject output roots that contain or sit inside an input or model tree."""
    root = Path(evaluation_root).resolve()
    for label, raw in (("checkpoint", checkpoint.parent), ("SRC", src), ("DST", dst)):
        selected = Path(raw).resolve(strict=True)
        if _contains(selected, root) or _contains(root, selected):
            raise ValueError(f"evaluation root overlaps {label}; choose an isolated output directory")
    return root


def check_snapshot_capacity(root):
    snapshots = root / "snapshots"
    if snapshots.exists() and sum(
        1 for entry in snapshots.iterdir()
        if entry.is_dir() and SNAPSHOT_ID.fullmatch(entry.name)
    ) >= MAX_ONLINE_SNAPSHOTS:
        raise ValueError("evaluation snapshot limit reached; archive an older snapshot first")


@contextmanager
def evaluation_environment(manifest, root, model_key, timeout_seconds, src, dst):
    values = {
        "DFL_WEB_EVAL_MANIFEST": str(manifest),
        "DFL_WEB_EVAL_ROOT": str(root),
        "DFL_WEB_EVAL_MODEL_KEY": model_key,
        "DFL_WEB_EVAL_TIMEOUT_SECONDS": str(timeout_seconds),
        "DFL_WEB_EVAL_SRC": str(src),
        "DFL_WEB_EVAL_DST": str(dst),
        "DFL_WEBUI_PYTHON": str(Path(__file__).resolve().parent),
    }
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_evaluation(args, *, before_publish=None):
    checkpoint = checkpoint_path(args.model)
    src = Path(args.src).resolve(strict=True)
    dst = Path(args.dst).resolve(strict=True)
    root = validate_paths(checkpoint, src, dst, args.evaluation_root)
    manifest_path = Path(args.manifest).resolve(strict=True)
    manifest, _ = load_evaluation_manifest(
        manifest_path, root, args.model_key, {"src": src, "dst": dst},
    )
    if manifest.get("modelName") != args.name or manifest.get("modelClass") != "ME":
        raise ValueError("evaluation manifest does not select the requested ME model")
    before = sha256_file(checkpoint)
    expected = getattr(args, "checkpoint_sha256", None)
    if expected and before != expected:
        raise ValueError("ME checkpoint SHA-256 differs from --checkpoint-sha256")
    check_snapshot_capacity(root)

    engine = MEEngine.load(checkpoint, args.device)
    iteration = engine.iteration
    stage = Path(tempfile.mkdtemp(prefix="_pending-cli-", dir=root)).resolve(strict=True)
    if stage.parent != root:
        raise ValueError("evaluation staging directory exceeds the selected root")
    published = False
    try:
        stage_manifest = stage / "manifests" / manifest_path.name
        stage_manifest.parent.mkdir()
        stage_manifest.write_bytes(manifest_path.read_bytes())
        with evaluation_environment(stage_manifest, stage, args.model_key,
                                    args.timeout_seconds, src, dst):
            summary = evaluate(engine, args.name, src, dst)
        snapshot_id = summary.get("snapshotId")
        if (not isinstance(snapshot_id, str) or not SNAPSHOT_ID.fullmatch(snapshot_id)
                or summary.get("modelKey") != args.model_key
                or summary.get("manifestId") != manifest["manifestId"]
                or summary.get("iteration") != iteration):
            raise ValueError("staged ME evaluation snapshot identity is invalid")
        snapshot = (stage / "snapshots" / snapshot_id).resolve(strict=True)
        if snapshot.parent != (stage / "snapshots").resolve(strict=True):
            raise ValueError("staged ME evaluation snapshot exceeds its root")
        # Reject changes made by another process during inference. This second
        # inventory scan also catches archive metadata replacements.
        if sha256_file(checkpoint) != before or engine.iteration != iteration:
            raise RuntimeError("ME checkpoint changed during evaluation")
        load_evaluation_manifest(manifest_path, root, args.model_key, {"src": src, "dst": dst})
        load_evaluation_manifest(stage_manifest, stage, args.model_key, {"src": src, "dst": dst})
        attestation = {
            "schemaVersion": 1,
            "checkpoint": str(checkpoint),
            "checkpointSha256": before,
            "iteration": iteration,
            "manifestId": manifest["manifestId"],
            "modelKey": args.model_key,
            "datasets": {
                side: {
                    "path": str(path),
                    "kind": manifest["datasets"][side]["kind"],
                    "fingerprint": manifest["datasets"][side]["fingerprint"],
                }
                for side, path in (("src", src), ("dst", dst))
            },
        }
        temporary = snapshot / "attestation.json.tmp"
        target = snapshot / "attestation.json"
        temporary.write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        if before_publish is not None:
            before_publish()

        # The Web index scans only root/snapshots. It sees either the complete
        # directory (summary plus attestation) or no snapshot from this run.
        check_snapshot_capacity(root)
        final_root = root / "snapshots"
        final_root.mkdir(exist_ok=True)
        if final_root.resolve(strict=True).parent != root:
            raise ValueError("evaluation snapshots directory exceeds the selected root")
        final = (final_root / snapshot_id).resolve()
        if final.parent != final_root.resolve() or final.exists():
            raise FileExistsError("evaluation snapshot ID collision")
        os.replace(snapshot, final)
        published = True
        return {"snapshot": str(final), "summary": summary, "attestation": attestation}
    finally:
        # A hard process kill may leave this hidden staging directory, but it
        # never appears in the visible snapshots index.
        if stage.exists():
            if stage.resolve().parent != root:
                raise ValueError("evaluation staging directory moved outside the selected root")
            error_in_flight = sys.exc_info()[0] is not None
            try:
                shutil.rmtree(stage)
            except OSError as error:
                if not published and not error_in_flight:
                    raise
                print(f"Warning: could not remove hidden ME evaluation staging directory {stage}: {error}",
                      file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "src", "dst", "name", "manifest", "evaluation-root", "model-key"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu", help="ME device (cpu, cuda, or cuda:0,1)")
    parser.add_argument("--timeout-seconds", type=int, default=120,
                        help="Inference deadline, 1–3600 seconds (default: 120)")
    parser.add_argument("--checkpoint-sha256", help="Optional exact checkpoint digest to require")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be an integer from 1 to 3600")
    if args.checkpoint_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", args.checkpoint_sha256):
        parser.error("--checkpoint-sha256 must be 64 lowercase hex characters")
    result = run_evaluation(args)
    print(json.dumps({
        "snapshot": result["snapshot"],
        "checkpointSha256": result["attestation"]["checkpointSha256"],
        "manifestId": result["attestation"]["manifestId"],
        "iteration": result["attestation"]["iteration"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
