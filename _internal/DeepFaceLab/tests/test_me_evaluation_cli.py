"""The offline CLI must bind a frozen checkpoint and arbitrary faceset roots."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CLI = REPOSITORY_ROOT / "webui" / "scripts" / "evaluate-me-checkpoint.mjs"
LOCAL_NODE = REPOSITORY_ROOT / "_internal" / "node" / "bin" / "node.exe"
NODE = str(LOCAL_NODE) if LOCAL_NODE.is_file() else shutil.which("node")
sys.path.insert(0, str(REPOSITORY_ROOT / "webui" / "python"))

from dfl_asset_tool import build_pose_probe_manifest  # noqa: E402
import evaluate_me_checkpoint as offline_cli  # noqa: E402
from evaluate_me_checkpoint import run_evaluation  # noqa: E402
from training_evaluation import manifest_content_id  # noqa: E402


def _run_cli(*arguments, expected=0, extra_env=None):
    command = [NODE, str(CLI), *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            timeout=180, env={**os.environ, "PYTHONUTF8": "1",
                                              **(extra_env or {})})
    assert result.returncode == expected, result.stdout + result.stderr
    return result


@pytest.mark.skipif(NODE is None, reason="Node.js runtime is unavailable")
@pytest.mark.parametrize("format_name", ["directory", "pak", "zip"])
def test_offline_cli_evaluates_external_facesets_without_changing_checkpoint(tmp_path, format_name):
    src = make_aligned(tmp_path / "outside-project-src", count=1)
    dst = make_aligned(tmp_path / "outside-project-dst", count=1, offset=20)
    if format_name != "directory":
        src = pack_fixture(src, format_name)
        dst = pack_fixture(dst, format_name)
    model = tmp_path / "frozen-model"
    checkpoint = model / "me.pt"
    MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=1), "cpu").save(checkpoint)
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "isolated-output"
    name = "offline-fixture"
    common = ("--model", checkpoint, "--src", src, "--dst", dst,
              "--name", name, "--evaluation-root", output, "--device", "cpu")

    result = _run_cli(*common, "--create-manifest", "--timeout-seconds", "300")
    assert "ME pose manifest:" in result.stdout
    manifest_path, = output.glob("*/manifests/*.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["datasets"]["src"]["kind"] == format_name
    assert manifest["datasets"]["dst"]["kind"] == format_name
    assert manifest["datasets"]["src"]["sampleCount"] == 1
    snapshots = list(output.glob("*/snapshots/iter-*/summary.json"))
    assert len(snapshots) == 1
    summary = json.loads(snapshots[0].read_text(encoding="utf-8"))
    attestation = json.loads((snapshots[0].parent / "attestation.json").read_text(encoding="utf-8"))
    assert snapshots[0].parent.name == summary["snapshotId"]
    assert len(summary["snapshotId"].rsplit("-", 1)[-1]) == 24
    assert summary["manifestId"] == manifest["manifestId"]
    assert len(summary["samples"]) == 2
    assert attestation["checkpointSha256"] == checkpoint_digest
    assert attestation["datasets"]["src"]["fingerprint"] == manifest["datasets"]["src"]["fingerprint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    _run_cli(*common, "--manifest", manifest_path,
             "--checkpoint-sha256", checkpoint_digest,
             extra_env={"DFL_WEB_EVAL_SRC": str(dst), "DFL_WEB_EVAL_DST": str(src)})
    replay_snapshots = list(output.glob("*/snapshots/iter-*/summary.json"))
    assert len(replay_snapshots) == 2
    replay_path, = (candidate for candidate in replay_snapshots if candidate != snapshots[0])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["samples"] == summary["samples"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_digest

    if format_name == "directory":
        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(original_checkpoint + b"tampered")
        tampered = _run_cli(*common, "--manifest", manifest_path,
                            "--checkpoint-sha256", checkpoint_digest, expected=1)
        assert "differs from --checkpoint-sha256" in tampered.stderr
        assert len(list(output.glob("*/snapshots/iter-*/summary.json"))) == 2
        checkpoint.write_bytes(original_checkpoint)

        image = src / "000.jpg"
        image.write_bytes(image.read_bytes() + b"changed")
    elif format_name == "pak":
        content = bytearray(src.read_bytes())
        content[-3] ^= 1
        src.write_bytes(content)
    else:
        with zipfile.ZipFile(src, "a") as archive:
            archive.comment = b"changed"
    changed = _run_cli(*common, "--manifest", manifest_path, expected=1)
    assert "dataset changed" in changed.stderr
    assert len(list(output.glob("*/snapshots/iter-*/summary.json"))) == 2
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_digest


@pytest.mark.skipif(NODE is None, reason="Node.js runtime is unavailable")
def test_offline_cli_rejects_output_overlapping_inputs_before_writing(tmp_path):
    src = make_aligned(tmp_path / "source", count=1)
    dst = make_aligned(tmp_path / "destination", count=1)
    model = tmp_path / "model"
    MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=1), "cpu").save(model / "me.pt")
    before = sorted(target.relative_to(src).as_posix() for target in src.rglob("*"))
    result = _run_cli("--model", model, "--src", src, "--dst", dst,
                      "--name", "overlap-fixture", "--evaluation-root", src,
                      "--create-manifest", expected=2)
    assert "overlaps SRC" in result.stderr
    assert sorted(target.relative_to(src).as_posix() for target in src.rglob("*")) == before


def test_offline_cli_failure_after_attestation_never_publishes_snapshot(tmp_path, monkeypatch, capsys):
    src = make_aligned(tmp_path / "src", count=1)
    dst = make_aligned(tmp_path / "dst", count=1, offset=19)
    checkpoint = tmp_path / "model" / "me.pt"
    MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=1), "cpu").save(checkpoint)
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = tmp_path / "evaluation"
    probes = {side: build_pose_probe_manifest(directory, side)
              for side, directory in (("src", src), ("dst", dst))}
    manifest = {
        "schemaVersion": 1, "modelKey": "offline-failure", "modelName": "failure-fixture",
        "modelClass": "ME",
        "poseBins": {"yaw": probes["src"]["yawTicks"], "pitch": probes["src"]["pitchTicks"]},
        "datasets": {
            side: {"kind": probe["datasetKind"], "fingerprint": probe["datasetFingerprint"],
                   "sampleCount": probe["sampleCount"]}
            for side, probe in probes.items()
        },
        "samples": probes["src"]["samples"] + probes["dst"]["samples"],
    }
    manifest["manifestId"] = manifest_content_id(manifest)
    target = root / "manifests" / f"{manifest['manifestId']}.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    args = SimpleNamespace(model=checkpoint, src=src, dst=dst, name="failure-fixture",
                           manifest=target, evaluation_root=root, model_key="offline-failure",
                           device="cpu", timeout_seconds=120, checkpoint_sha256=checkpoint_digest)

    def fail_before_publish():
        stages = list(root.glob("_pending-cli-*"))
        assert len(stages) == 1
        staged_summary, = stages[0].glob("snapshots/iter-*/summary.json")
        staged_attestation, = stages[0].glob("snapshots/iter-*/attestation.json")
        assert json.loads(staged_summary.read_text(encoding="utf-8"))["manifestId"] == manifest["manifestId"]
        assert json.loads(staged_attestation.read_text(encoding="utf-8"))["checkpointSha256"] == checkpoint_digest
        staged_manifest = stages[0] / "manifests" / target.name
        assert hashlib.sha256(staged_manifest.read_bytes()).hexdigest() == manifest_digest
        assert not list(root.glob("snapshots/iter-*"))
        raise RuntimeError("injected failure before final publication")

    with pytest.raises(RuntimeError, match="injected failure before final publication"):
        run_evaluation(args, before_publish=fail_before_publish)
    assert not list(root.glob("snapshots/iter-*"))
    assert not list(root.glob("_pending-cli-*"))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest
    assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest_digest

    crash_root = tmp_path / "crash-evaluation"
    crash_manifest = crash_root / "manifests" / target.name
    crash_manifest.parent.mkdir(parents=True)
    shutil.copyfile(target, crash_manifest)
    script = """
import os
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from evaluate_me_checkpoint import run_evaluation
args = SimpleNamespace(model=Path(sys.argv[2]), src=Path(sys.argv[3]), dst=Path(sys.argv[4]),
    name='failure-fixture', manifest=Path(sys.argv[5]), evaluation_root=Path(sys.argv[6]),
    model_key='offline-failure', device='cpu', timeout_seconds=120,
    checkpoint_sha256=sys.argv[7])
run_evaluation(args, before_publish=lambda: os._exit(77))
"""
    crashed = subprocess.run([
        sys.executable, "-c", script, str(REPOSITORY_ROOT / "webui" / "python"),
        str(checkpoint), str(src), str(dst), str(crash_manifest), str(crash_root),
        checkpoint_digest,
    ], capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert crashed.returncode == 77, crashed.stdout + crashed.stderr
    assert not list(crash_root.glob("snapshots/iter-*"))
    crash_stage, = crash_root.glob("_pending-cli-*")
    crash_summary, = crash_stage.glob("snapshots/iter-*/summary.json")
    crash_attestation, = crash_stage.glob("snapshots/iter-*/attestation.json")
    assert json.loads(crash_summary.read_text(encoding="utf-8"))["manifestId"] == manifest["manifestId"]
    assert json.loads(crash_attestation.read_text(encoding="utf-8"))["checkpointSha256"] == checkpoint_digest
    assert hashlib.sha256((crash_stage / "manifests" / target.name).read_bytes()).hexdigest() == manifest_digest
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest

    original_remove = offline_cli.shutil.rmtree

    def deny_stage_cleanup(directory):
        if Path(directory).name.startswith("_pending-cli-"):
            raise PermissionError("injected Windows cleanup denial")
        return original_remove(directory)

    with monkeypatch.context() as patch:
        patch.setattr(offline_cli.shutil, "rmtree", deny_stage_cleanup)
        result = run_evaluation(args)
    assert Path(result["snapshot"]).joinpath("summary.json").is_file()
    assert Path(result["snapshot"]).joinpath("attestation.json").is_file()
    assert "could not remove hidden ME evaluation staging directory" in capsys.readouterr().err
    assert len(list(root.glob("_pending-cli-*"))) == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest

    def original_failure():
        raise RuntimeError("injected original evaluation failure")

    with monkeypatch.context() as patch:
        patch.setattr(offline_cli.shutil, "rmtree", deny_stage_cleanup)
        with pytest.raises(RuntimeError, match="injected original evaluation failure"):
            run_evaluation(args, before_publish=original_failure)
    assert "could not remove hidden ME evaluation staging directory" in capsys.readouterr().err
    assert len(list(root.glob("snapshots/iter-*"))) == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest

    with monkeypatch.context() as patch:
        patch.setattr(offline_cli, "MAX_ONLINE_SNAPSHOTS", 1)
        with pytest.raises(ValueError, match="snapshot limit reached"):
            run_evaluation(args)
    assert len(list(root.glob("snapshots/iter-*"))) == 1
