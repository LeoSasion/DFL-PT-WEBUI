"""Online ME snapshots identify the exact generator weights used for inference."""

import json
from pathlib import Path
import sys

import torch

from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.web_bridge import evaluate, network_weights_sha256
from tests.me_fixtures import make_aligned


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "webui" / "python"))
from dfl_asset_tool import build_pose_probe_manifest  # noqa: E402
from training_evaluation import manifest_content_id  # noqa: E402


def test_same_iteration_and_config_with_different_weights_have_distinct_snapshot_digests(
        tmp_path, monkeypatch):
    torch.set_num_threads(1)
    dataset = make_aligned(tmp_path / "aligned", count=1)
    src = build_pose_probe_manifest(dataset, "src")
    dst = build_pose_probe_manifest(dataset, "dst")
    model_key, name = "weight-identity-fixture", "weight-fixture"
    evaluation_root = tmp_path / "evaluation"
    manifests = evaluation_root / "manifests"
    manifests.mkdir(parents=True)
    manifest = {
        "schemaVersion": 1, "modelKey": model_key, "modelName": name,
        "modelClass": "ME",
        "datasets": {side: {"kind": probe["datasetKind"],
                            "fingerprint": probe["datasetFingerprint"],
                            "sampleCount": probe["sampleCount"]}
                     for side, probe in (("src", src), ("dst", dst))},
        "samples": src["samples"] + dst["samples"],
    }
    manifest_id = manifest_content_id(manifest)
    manifest["manifestId"] = manifest_id
    target = manifests / f"{manifest_id}.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    for key, value in {
        "DFL_WEB_EVAL_MANIFEST": target,
        "DFL_WEB_EVAL_ROOT": evaluation_root,
        "DFL_WEB_EVAL_MODEL_KEY": model_key,
        "DFL_WEBUI_PYTHON": Path(__file__).resolve().parents[3] / "webui" / "python",
    }.items():
        monkeypatch.setenv(key, str(value))

    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=1)
    first_engine = MEEngine(config, "cpu", seed=1)
    second_engine = MEEngine(config, "cpu", seed=2)
    first = evaluate(first_engine, name, dataset, dataset)
    second = evaluate(second_engine, name, dataset, dataset)

    assert first["iteration"] == second["iteration"] == 0
    assert first["modelSignature"] == second["modelSignature"]
    assert first["modelWeightsDigestVersion"] == second["modelWeightsDigestVersion"] == 1
    assert first["modelWeightsSha256"] != second["modelWeightsSha256"]
    assert first["modelWeightsSha256"] == network_weights_sha256(first_engine.network)
    assert second["modelWeightsSha256"] == network_weights_sha256(second_engine.network)
    sample_id = src["samples"][0]["id"]
    def reconstruction(summary):
        return (evaluation_root / "snapshots" / summary["snapshotId"]
                / "samples" / sample_id / "reconstruction.webp").read_bytes()
    assert reconstruction(first) != reconstruction(second)
