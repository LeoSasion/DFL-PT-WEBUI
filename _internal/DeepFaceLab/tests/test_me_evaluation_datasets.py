"""Probe and snapshot inputs must resolve the same ME faceset members."""

import json
from pathlib import Path
import sys
import zipfile

import pytest

from DFLIMG import DFLJPG
from me_backend.config import MEConfig
from me_backend.data import read_aligned
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "webui" / "python"))
from dfl_asset_tool import build_pose_probe_manifest
from training_evaluation import load_evaluation_manifest, manifest_content_id


def _dataset(directory):
    for person in ("alice", "bob"):
        make_aligned(directory / person, count=1, offset=0 if person == "alice" else 25)
        image = directory / person / "000.jpg"
        dfl = DFLJPG.load(str(image))
        dfl.set_source_filename(f"{person}-frame.png")
        dfl.save()
    return directory


def _manifest(tmp_path, dataset):
    src = build_pose_probe_manifest(dataset, "src")
    dst = build_pose_probe_manifest(dataset, "dst")
    assert src["datasetFingerprint"] == dst["datasetFingerprint"]
    assert {sample["member"] for sample in src["samples"]} == {"alice/000.jpg", "bob/000.jpg"}
    key = "evaluation-fixture"
    root = tmp_path / "evaluation"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    manifest = {
        "schemaVersion": 1, "modelKey": key,
        "modelName": "fixture", "modelClass": "ME",
        "datasets": {side: {"kind": probe["datasetKind"],
                            "fingerprint": probe["datasetFingerprint"],
                            "sampleCount": probe["sampleCount"]}
                     for side, probe in (("src", src), ("dst", dst))},
        "samples": src["samples"] + dst["samples"],
    }
    identifier = manifest_content_id(manifest)
    manifest["manifestId"] = identifier
    target = manifests / f"{identifier}.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target, root, key, src


def test_manifest_content_tampering_is_rejected(tmp_path):
    dataset = _dataset(tmp_path / "aligned")
    target, root, key, _ = _manifest(tmp_path, dataset)
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["samples"][0]["yaw"] += 77
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is invalid"):
        load_evaluation_manifest(target, root, key, {"src": dataset, "dst": dataset})


@pytest.mark.parametrize("format_name", ["directory", "pak", "zip"])
def test_probe_and_evaluator_share_content_identity_for_me_facesets(tmp_path, format_name):
    directory = _dataset(tmp_path / "aligned")
    dataset = directory if format_name == "directory" else pack_fixture(directory, format_name)
    target, root, key, probe = _manifest(tmp_path, dataset)
    assert probe["datasetKind"] == format_name
    _, samples = load_evaluation_manifest(target, root, key, {"src": dataset, "dst": dataset})
    config = MEConfig(resolution=64, batch_size=1)
    for side in ("src", "dst"):
        assert len(samples[side]) == 2
        for sample in samples[side]:
            image, full, encoded = read_aligned(sample["path"], config)
            assert image.shape == (64, 64, 3)
            assert full.shape == (64, 64, 1)
            assert encoded.shape == (64, 64, 1)

    if format_name == "directory":
        image = directory / "alice" / "000.jpg"
        image.write_bytes(image.read_bytes() + b"changed")
    elif format_name == "pak":
        content = bytearray(dataset.read_bytes())
        content[-3] ^= 1
        dataset.write_bytes(content)
    else:
        with zipfile.ZipFile(dataset, "a") as archive:
            archive.comment = b"changed"
    with pytest.raises(ValueError, match="dataset changed"):
        load_evaluation_manifest(target, root, key, {"src": dataset, "dst": dataset})


def test_explicit_zip_probe_does_not_follow_implicit_pak_priority(tmp_path):
    directory = _dataset(tmp_path / "aligned")
    pak = pack_fixture(directory, "pak", delete_images=False)
    pak_probe = build_pose_probe_manifest(directory, "src")
    assert pak_probe["datasetKind"] == "pak"
    assert pak_probe["datasetFingerprint"] == build_pose_probe_manifest(pak, "src")["datasetFingerprint"]
    make_aligned(directory / "alice", count=1, offset=67)
    zip_file = pack_fixture(directory, "zip")
    zip_probe = build_pose_probe_manifest(zip_file, "src")
    assert zip_probe["datasetKind"] == "zip"
    assert zip_probe["datasetFingerprint"] != pak_probe["datasetFingerprint"]
