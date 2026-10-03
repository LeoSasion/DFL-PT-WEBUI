"""Web evaluation chooses its own facesets without changing Trainer inputs."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from torch import nn

from me_backend.config import MEConfig
from me_backend.web_bridge import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "webui" / "python"))
import training_evaluation


def test_web_evaluation_uses_optional_holdout_paths_and_legacy_training_defaults(
        tmp_path, monkeypatch):
    observed = []

    def load_manifest(_manifest_path, _root, _model_key, datasets):
        observed.append(datasets)
        return {"manifestId": "a" * 24, "modelName": "fixture", "modelClass": "ME"}, {
            "src": [], "dst": [],
        }

    class Snapshot:
        def __init__(self, *_, **__):
            pass

        def publish(self):
            return {"snapshotId": "fixture-snapshot"}

        def abort(self):
            raise AssertionError("No snapshot should fail in this test")

    monkeypatch.setattr(training_evaluation, "load_evaluation_manifest", load_manifest)
    monkeypatch.setattr(training_evaluation, "AtomicEvaluationSnapshot", Snapshot)
    monkeypatch.setenv("DFL_WEB_EVAL_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("DFL_WEB_EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("DFL_WEB_EVAL_MODEL_KEY", "fixture")
    engine = SimpleNamespace(config=MEConfig(), iteration=7, network=nn.Identity())

    monkeypatch.delenv("DFL_WEB_EVAL_SRC", raising=False)
    monkeypatch.delenv("DFL_WEB_EVAL_DST", raising=False)
    evaluate(engine, "fixture", "training-src", "training-dst")
    assert observed[-1] == {"src": "training-src", "dst": "training-dst"}

    monkeypatch.setenv("DFL_WEB_EVAL_SRC", "holdout-src")
    monkeypatch.setenv("DFL_WEB_EVAL_DST", "holdout-dst")
    evaluate(engine, "fixture", "training-src", "training-dst")
    assert observed[-1] == {"src": "holdout-src", "dst": "holdout-dst"}

    monkeypatch.delenv("DFL_WEB_EVAL_DST")
    with pytest.raises(ValueError, match="configured together"):
        evaluate(engine, "fixture", "training-src", "training-dst")
    monkeypatch.setenv("DFL_WEB_EVAL_DST", "holdout-dst")
    for timeout in ("0", "3601", "invalid"):
        monkeypatch.setenv("DFL_WEB_EVAL_TIMEOUT_SECONDS", timeout)
        with pytest.raises(ValueError, match="timeout must be 1..3600"):
            evaluate(engine, "fixture", "training-src", "training-dst")
