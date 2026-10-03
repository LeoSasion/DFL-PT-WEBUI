"""Synthetic technical smoke only; never a human quality acceptance."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "me-longrun-audit.py"
spec = importlib.util.spec_from_file_location("me_longrun_audit", SCRIPT)
audit_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tool)

import torch
from me_backend.config import MEConfig
from me_backend.data import AlignedDataset
from me_backend.engine import MEEngine
from tests.me_fixtures import make_aligned

torch.set_num_threads(2)


def _history(target, rows):
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _row(iteration, when, src=0.4, dst=0.5, seconds=0.2):
    return {"iteration": iteration, "at": when.isoformat(),
            "src_loss": src, "dst_loss": dst, "seconds": seconds}


def test_history_rejects_iteration_gaps_and_does_not_count_offline_hours(tmp_path):
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    target = tmp_path / "loss-history.jsonl"
    _history(target, [_row(1, start), _row(3, start + timedelta(hours=13))])
    with pytest.raises(ValueError, match="iteration gap"):
        audit_tool.inspect_history(target, 3)
    _history(target, [_row(1, start), _row(2, start + timedelta(hours=13))])
    result = audit_tool.inspect_history(target, 2)
    assert result["wallSpanHours"] == 13
    assert result["observedActiveHours"] < 0.01
    assert result["pauseCount"] == 1
    with pytest.raises(ValueError, match="checkpoint iteration"):
        audit_tool.inspect_history(target, 3)
    _history(target, [_row(1, start), _row(2, start + timedelta(seconds=2), src=float("nan"))])
    with pytest.raises(ValueError, match="nonfinite"):
        audit_tool.inspect_history(target, 2)


def test_frozen_sample_manifest_rejects_changed_content(tmp_path):
    src = make_aligned(tmp_path / "src", count=2)
    dst = make_aligned(tmp_path / "dst", count=2, offset=25)
    target = tmp_path / "fixed.json"
    selected, created = audit_tool.freeze_samples(src, dst, target, 1)
    assert created["created"] is True
    assert selected["src"][0].member == "000.jpg"
    _, reused = audit_tool.freeze_samples(src, dst, target, 1)
    assert reused["created"] is False
    image = src / "000.jpg"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fixed sample manifest differs"):
        audit_tool.freeze_samples(src, dst, target, 1)


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_synthetic_checkpoint_cli_audits_predictions_dfm_and_short_history(tmp_path, legacy_metadata):
    src = make_aligned(tmp_path / "src", count=1)
    dst = make_aligned(tmp_path / "dst", count=1, offset=25)
    model = tmp_path / "model" / "synthetic-smoke"
    model.mkdir(parents=True)
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=1)
    engine = MEEngine(config, "cpu", seed=3)
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    with AlignedDataset(src, config, True, 1) as source, AlignedDataset(dst, config, False, 2) as destination:
        rows = []
        for index in (1, 2):
            result = engine.train_step(source.batch(), destination.batch())
            rows.append(_row(index, start + timedelta(seconds=index * 2),
                             src=result["src_loss"], dst=result["dst_loss"]))
    engine.save(model / "me.pt")
    metadata_config = config.to_dict()
    if legacy_metadata:
        # The 24 fields written before the newer ME training options existed.
        legacy_fields = {
            "archi", "resolution", "ae_dims", "e_dims", "d_dims", "d_mask_dims",
            "face_type", "batch_size", "masked_training", "eyes_prio", "mouth_prio",
            "loss_function", "background_power", "face_style_power", "bg_style_power",
            "blur_out_mask", "lr", "adabelief", "lr_dropout", "clipgrad",
            "random_warp", "random_src_flip", "random_dst_flip", "random_hsv_power",
        }
        metadata_config = {key: value for key, value in metadata_config.items()
                           if key in legacy_fields}
    (model / "metadata.json").write_text(json.dumps({
        "format": "me-pytorch", "version": 1, "model_class": "ME",
        "name": model.name, "checkpoint": "me.pt", "iteration": engine.iteration,
        "config": metadata_config,
    }), encoding="utf-8")
    _history(model / "loss-history.jsonl", rows)
    before = audit_tool.sha256_file(model / "me.pt")
    output = tmp_path / "audit.json"
    run = subprocess.run([sys.executable, str(SCRIPT), "--model", str(model),
        "--src", str(src), "--dst", str(dst), "--material", "synthetic",
        "--minimum-hours", "0", "--sample-count", "1", "--json-out", str(output)],
        capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["outcome"]["technicalPassed"] is True
    assert report["outcome"]["longRunEvidenceQualified"] is False
    assert report["outcome"]["visualQualityAccepted"] is False
    assert report["outcome"]["deepFaceLiveAccepted"] is False
    assert report["checks"]["dfm"]["status"] == "passed"
    assert report["checks"]["dfm"]["details"]["batchSize"] == 2
    assert report["checks"]["history"]["details"]["lastIteration"] == 2
    assert audit_tool.sha256_file(model / "me.pt") == before
    if legacy_metadata:
        metadata_config["batch_size"] = 2
        metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
        metadata["config"] = metadata_config
        (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="metadata does not match"):
            audit_tool._checkpoint(model, "cpu")


def test_optional_video_is_probed_and_fully_decoded(tmp_path):
    ffmpeg = ROOT / "_internal" / "ffmpeg" / "ffmpeg.exe"
    ffprobe = ROOT / "_internal" / "ffmpeg" / "ffprobe.exe"
    if not ffmpeg.is_file() or not ffprobe.is_file():
        pytest.skip("bundled FFmpeg tools are unavailable")
    output = tmp_path / "sample.mp4"
    command = subprocess.run([str(ffmpeg), "-v", "error", "-f", "lavfi", "-i",
        "color=c=green:s=64x64:r=2:d=1", "-frames:v", "2", "-c:v", "mpeg4",
        "-y", str(output)], capture_output=True, text=True, timeout=30)
    assert command.returncode == 0, command.stderr
    result = audit_tool.inspect_video(output, ffprobe, ffmpeg)
    assert result["fullVideoDecode"] is True
    assert result["width"] == result["height"] == 64
    assert result["durationSeconds"] > 0


def test_report_cannot_overwrite_checkpoint_or_frozen_manifest(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = model / "me.pt"
    checkpoint.write_bytes(b"protected")
    common = ["--model", str(model), "--src", str(tmp_path), "--dst", str(tmp_path),
              "--material", "synthetic", "--minimum-hours", "0"]
    with pytest.raises(ValueError, match="cannot replace"):
        audit_tool.main(common + ["--json-out", str(checkpoint)])
    with pytest.raises(ValueError, match="different files"):
        audit_tool.main(common + ["--json-out", str(model / "audit-samples.json")])
    assert checkpoint.read_bytes() == b"protected"
