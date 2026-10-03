"""Short development evidence checks, with synthetic history and inference mocks.

These tests neither train an engine nor require CUDA or long-run execution.
"""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("me_short_run_audit", ROOT / "tools" / "me-short-run-audit.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
START = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _row(iteration, src=1.0, dst=2.0, seconds=0.2, at=None):
    return {"iteration": iteration, "at": (at or START + timedelta(seconds=iteration)).isoformat(),
            "seconds": seconds, "src_loss": src, "dst_loss": dst}


def _history(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _decreasing(path, count=10):
    return _history(path, [_row(i, src=1.0 if i <= 5 else 0.5,
                               dst=2.0 if i <= 5 else 1.0) for i in range(1, count + 1)])


def test_median_trend_uses_explicit_independent_windows_and_resists_outliers(tmp_path):
    rows = [_row(i, src=1 if i <= 5 else 0.5, dst=2 if i <= 5 else 1) for i in range(1, 11)]
    rows[-1]["src_loss"] = 1000  # Mean would obscure the actual window median.
    history = tool.inspect_history(_history(tmp_path / "history", rows), 10, window_size=5)
    trend = history["trend"]
    assert trend["status"] == "passed"
    assert trend["earlySampleCount"] == trend["lateSampleCount"] == 5
    assert trend["windowsNonoverlapping"] is True
    assert trend["earlyMedianLoss"] == {"src": 1, "dst": 2}
    assert trend["lateMedianLoss"] == {"src": 0.5, "dst": 1}
    assert trend["combinedRelativeReduction"] == 0.5
    assert history["historyEnvelopeSeconds"] == pytest.approx(9.2)


def test_too_few_records_are_inconclusive_even_when_loss_falls(tmp_path):
    history = tool.inspect_history(_decreasing(tmp_path / "history", 9), 9, window_size=5)
    assert history["trend"]["status"] == "inconclusive"
    assert history["trend"]["windowsNonoverlapping"] is False
    assert "10 records" in history["trend"]["reason"]


@pytest.mark.parametrize("late_src,late_dst", [(1, 2), (0.999, 2), (1.01, 0.1), (0, 0)])
def test_insufficient_drop_or_increasing_side_is_not_a_pass(tmp_path, late_src, late_dst):
    early_src, early_dst = (0, 0) if (late_src, late_dst) == (0, 0) else (1, 2)
    rows = [_row(i, src=early_src if i <= 5 else late_src,
                  dst=early_dst if i <= 5 else late_dst) for i in range(1, 11)]
    result = tool.inspect_history(_history(tmp_path / "history", rows), 10, window_size=5)
    assert result["trend"]["status"] == "inconclusive"


@pytest.mark.parametrize("field,value", [
    ("src_loss", float("nan")), ("dst_loss", float("inf")), ("src_loss", True),
    ("seconds", -1), ("seconds", float("inf")), ("dst_loss", -0.1),
    ("iteration", True), ("at", "2026-10-04T00:00:00"),
])
def test_corrupt_history_values_are_rejected(tmp_path, field, value):
    row = _row(1)
    row[field] = value
    with pytest.raises(ValueError):
        tool.inspect_history(_history(tmp_path / "history", [row]), 1, window_size=5)


@pytest.mark.parametrize("iterations", [(1, 1), (1, 3), (2, 3)])
def test_duplicate_missing_or_noninitial_iterations_are_rejected(tmp_path, iterations):
    with pytest.raises(ValueError, match="iteration|start"):
        tool.inspect_history(_history(tmp_path / "history", [_row(i) for i in iterations]),
                             iterations[-1], window_size=5)


def test_checkpoint_and_final_history_iteration_must_match(tmp_path):
    with pytest.raises(ValueError, match="checkpoint iteration"):
        tool.inspect_history(_decreasing(tmp_path / "history"), 11, window_size=5)


@pytest.mark.parametrize("text", ["", "\n", "{}\n", "{broken\n"])
def test_empty_or_malformed_history_is_rejected(tmp_path, text):
    target = tmp_path / "history"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        tool.inspect_history(target, 1, window_size=5)


def test_time_cannot_move_backwards(tmp_path):
    rows = [_row(1, at=START + timedelta(seconds=2)), _row(2, at=START)]
    with pytest.raises(ValueError, match="backwards"):
        tool.inspect_history(_history(tmp_path / "history", rows), 2, window_size=5)


def test_runtime_includes_pauses_and_does_not_grant_extra_allowance(tmp_path):
    details = tool.inspect_history(_decreasing(tmp_path / "history"), 10, window_size=5)
    bounded = tool.inspect_runtime(details, START.isoformat(), (START + timedelta(seconds=3600)).isoformat())
    assert bounded["status"] == "passed"
    assert bounded["measuredWallSeconds"] == 3600
    reserved = tool.inspect_runtime(details, START.isoformat(), (START + timedelta(seconds=3599)).isoformat(),
                                    allowance_seconds=60)
    assert reserved["status"] == "failed"
    assert reserved["measuredPlusAllowanceSeconds"] == 3659
    with pytest.raises(ValueError, match="<=3600"):
        tool.inspect_runtime(details, maximum_seconds=3601)
    paused = dict(details, historyEnvelopeSeconds=4000)
    assert tool.inspect_runtime(paused)["status"] == "failed"


def test_history_duration_is_not_full_process_duration_proof(tmp_path):
    details = tool.inspect_history(_decreasing(tmp_path / "history"), 10, window_size=5)
    assert tool.inspect_runtime(details)["status"] == "inconclusive"
    with pytest.raises(ValueError, match="both"):
        tool.inspect_runtime(details, started_at=START.isoformat())
    with pytest.raises(ValueError, match="contain"):
        tool.inspect_runtime(details, (START + timedelta(seconds=1)).isoformat(),
                             (START + timedelta(seconds=11)).isoformat())
    with pytest.raises(ValueError, match="save timestamp"):
        tool.inspect_runtime(details, START.isoformat(), (START + timedelta(seconds=11)).isoformat(),
                             saved_at=(START + timedelta(seconds=12)).isoformat())
    inconsistent = dict(details, stepComputeSeconds=30)
    with pytest.raises(ValueError, match="step durations"):
        tool.inspect_runtime(inconsistent, START.isoformat(), (START + timedelta(seconds=11)).isoformat())


@pytest.mark.parametrize("window,drop", [(4, 0.01), (1001, 0.01), (5, 0), (5, float("nan")), (5, 2)])
def test_trend_thresholds_cannot_disable_evidence_requirements(tmp_path, window, drop):
    with pytest.raises(ValueError):
        tool.inspect_history(_decreasing(tmp_path / "history"), 10, window, drop)


@pytest.fixture
def saved_model(tmp_path):
    from me_backend.config import MEConfig

    model = tmp_path / "model"
    model.mkdir()
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1)
    engine = SimpleNamespace(iteration=10, optimizer_updates=10, config=config)
    metadata = {"name": model.name, "format": "me-pytorch", "version": 1, "model_class": "ME",
                "iteration": 10, "checkpoint": "me.pt", "config": config.to_dict(),
                "savedAt": (START + timedelta(seconds=10.5)).isoformat()}
    (model / "me.pt").write_bytes(b"strict-loading-is-mocked")
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _decreasing(model / "loss-history.jsonl")
    return model, engine, metadata


def test_checkpoint_uses_strict_loader_and_config_metadata_agreement(saved_model, monkeypatch):
    from me_backend.engine import MEEngine

    model, engine, metadata = saved_model
    calls = []
    def load(path, device):
        calls.append((path, device))
        return engine
    monkeypatch.setattr(MEEngine, "load", load)
    loaded, _, details = tool.inspect_checkpoint(model, "cpu")
    assert loaded is engine
    assert calls == [(model / "me.pt", "cpu")]
    assert details["optimizerUpdates"] == 10
    # Legacy metadata can omit new options which have unchanged defaults.
    metadata["config"].pop("random_jpeg")
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    tool.inspect_checkpoint(model, "cpu")
    metadata["config"]["batch_size"] = 2
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata does not match"):
        tool.inspect_checkpoint(model, "cpu")


@pytest.mark.parametrize("field,value", [("iteration", True), ("checkpoint", "other.pt")])
def test_checkpoint_identity_metadata_cannot_mismatch(saved_model, monkeypatch, field, value):
    from me_backend.engine import MEEngine

    model, engine, metadata = saved_model
    monkeypatch.setattr(MEEngine, "load", lambda path, device: engine)
    metadata[field] = value
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata does not match"):
        tool.inspect_checkpoint(model, "cpu")


def test_explicit_model_name_is_independent_of_directory_but_can_be_pinned(saved_model, monkeypatch):
    from me_backend.engine import MEEngine

    model, engine, metadata = saved_model
    monkeypatch.setattr(MEEngine, "load", lambda path, device: engine)
    metadata["name"] = "short-run"
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert tool.inspect_checkpoint(model, "cpu", "short-run")[2]["name"] == "short-run"
    with pytest.raises(ValueError, match="expected model identity"):
        tool.inspect_checkpoint(model, "cpu", "other")
    metadata["name"] = "../outside"
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="model name"):
        tool.inspect_checkpoint(model, "cpu")


@pytest.fixture
def prediction_inputs(tmp_path, monkeypatch):
    import me_backend.data as data
    import pose_probe_contract as contract

    directories, refs = [], {}
    for side in ("src", "dst"):
        directory = tmp_path / side
        directory.mkdir()
        images = []
        for index in range(4):
            target = directory / f"{index}.jpg"
            target.write_bytes(f"{side}-{index}".encode())
            images.append(SimpleNamespace(member=target.name, path=target, packed_sample=None,
                                          sha256=tool._sha256(target)))
        refs[directory] = images
        directories.append(directory)
    monkeypatch.setattr(contract, "probe_dataset_inventory", lambda path: ("directory", "fingerprint", refs[path]))
    monkeypatch.setattr(data, "read_aligned", lambda path, config: (np.full((64, 64, 3), 0.5, np.float32), None, None))
    return directories, refs


def _fake_prediction(batch):
    return np.full_like(batch, 0.5), np.full((1, 1, 64, 64), 0.5), np.full((1, 1, 64, 64), 0.5)


def test_small_fixed_predictions_are_content_identified_and_read_only(prediction_inputs):
    directories, refs = prediction_inputs
    before = {ref.path: ref.path.read_bytes() for images in refs.values() for ref in images}
    engine = SimpleNamespace(config=SimpleNamespace(resolution=64), predict=_fake_prediction)
    results = tool.inspect_predictions(engine, *directories, count=3)
    assert [sample["member"] for sample in results["src"]["samples"]] == ["0.jpg", "2.jpg", "3.jpg"]
    assert results["dst"]["sampleCount"] == 3
    assert results["src"]["samples"][0]["sha256"] == refs[directories[0]][0].sha256
    assert all(path.read_bytes() == content for path, content in before.items())
    assert sum(len(list(directory.iterdir())) for directory in directories) == 8


@pytest.mark.parametrize("corruption", ["nan", "shape", "range", "changed"])
def test_prediction_corruption_or_changed_sample_is_rejected(prediction_inputs, corruption):
    directories, refs = prediction_inputs
    def predict(batch):
        swap, src, dst = _fake_prediction(batch)
        if corruption == "nan":
            swap[0, 0, 0, 0] = np.nan
        elif corruption == "shape":
            dst = dst[:, :, :, :32]
        elif corruption == "range":
            src[0, 0, 0, 0] = 2
        else:
            refs[directories[0]][0].path.write_bytes(b"changed")
        return swap, src, dst
    engine = SimpleNamespace(config=SimpleNamespace(resolution=64), predict=predict)
    with pytest.raises(ValueError):
        tool.inspect_predictions(engine, *directories)


def _arguments(model, tmp_path, *extra):
    return tool.parse_args(["--model", str(model), "--src", str(tmp_path / "src"),
                            "--dst", str(tmp_path / "dst"), "--material", "synthetic",
                            "--window-size", "5", *extra])


def _mock_load_and_predictions(monkeypatch, saved_model):
    model, engine, metadata = saved_model
    monkeypatch.setattr(tool, "inspect_checkpoint", lambda directory, device, expected_name: (
        engine, metadata, {"iteration": engine.iteration}))
    monkeypatch.setattr(tool, "inspect_predictions", lambda *args: {"src": {}, "dst": {}})
    return model


def test_report_distinguishes_pass_from_missing_runtime_without_final_quality_claims(saved_model, monkeypatch, tmp_path):
    model = _mock_load_and_predictions(monkeypatch, saved_model)
    incomplete = tool.audit(_arguments(model, tmp_path))
    assert incomplete["outcome"]["status"] == "inconclusive"
    assert incomplete["outcome"]["technicalPassed"] is True
    args = _arguments(model, tmp_path, "--run-started-at", START.isoformat(), "--run-stopped-at",
                      (START + timedelta(seconds=11)).isoformat())
    report = tool.audit(args)
    assert report["outcome"] == {"status": "passed", "technicalPassed": True,
                                 "shortDevelopmentPassed": True, "longRunEvidenceQualified": False,
                                 "visualQualityAccepted": False, "deepFaceLiveAccepted": False}


@pytest.mark.parametrize("filename", ["me.pt", "metadata.json", "loss-history.jsonl"])
def test_audit_fails_if_input_changes_while_predictions_run(saved_model, monkeypatch, tmp_path, filename):
    model = _mock_load_and_predictions(monkeypatch, saved_model)
    def predictions(*args):
        target = model / filename
        target.write_bytes(target.read_bytes() + b"changed")
        return {}
    monkeypatch.setattr(tool, "inspect_predictions", predictions)
    report = tool.audit(_arguments(model, tmp_path))
    assert report["checks"]["inputIdentity"]["status"] == "failed"
    assert report["outcome"]["status"] == "failed"


def test_corrupt_checkpoint_is_reported_as_failure_without_writing_inputs(saved_model, monkeypatch, tmp_path):
    from me_backend.engine import MEEngine

    model, _, _ = saved_model
    before = {path: path.read_bytes() for path in model.iterdir()}
    def fail(*args):
        raise ValueError("Incomplete ME checkpoint")
    monkeypatch.setattr(MEEngine, "load", fail)
    report = tool.audit(_arguments(model, tmp_path))
    assert report["checks"]["checkpoint"]["status"] == "failed"
    assert report["checks"]["history"]["status"] == "skipped"
    assert report["outcome"]["status"] == "failed"
    assert all(path.read_bytes() == content for path, content in before.items())


def test_json_report_cannot_overwrite_model_inputs_or_dataset_members(saved_model, tmp_path):
    model, _, _ = saved_model
    for name in ("me.pt", "metadata.json", "loss-history.jsonl"):
        with pytest.raises(ValueError, match="cannot replace"):
            tool._validate_report_path(_arguments(model, tmp_path, "--json-out", str(model / name)))
    alias = tmp_path / "alias.json"
    alias.hardlink_to(model / "me.pt")
    with pytest.raises(ValueError, match="cannot replace"):
        tool._validate_report_path(_arguments(model, tmp_path, "--json-out", str(alias)))
    source = tmp_path / "src"
    source.mkdir()
    with pytest.raises(ValueError, match="inside an aligned dataset"):
        tool._validate_report_path(_arguments(model, tmp_path, "--json-out", str(source / "image.jpg")))


def test_cli_writes_only_requested_report_and_uses_inconclusive_exit_code(saved_model, monkeypatch, tmp_path, capsys):
    model = _mock_load_and_predictions(monkeypatch, saved_model)
    before = {path: path.read_bytes() for path in model.iterdir()}
    output = model / "short-audit.json"
    args = ["--model", str(model), "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"),
            "--material", "synthetic", "--window-size", "5", "--json-out", str(output)]
    assert tool.main(args) == 2
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text(encoding="utf-8")) == printed
    assert printed["outcome"]["status"] == "inconclusive"
    assert all(path.read_bytes() == content for path, content in before.items())
    assert set(model.iterdir()) == set(before) | {output}
