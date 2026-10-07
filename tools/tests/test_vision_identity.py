import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui/python"))
sys.path.insert(0, str(ROOT / "_internal/DeepFaceLab"))
from vision_identity import (SFACE_PATH, SFaceTorch, file_sha256, normalized,
                             validate_crop, verified_identity, IDENTITY_CONTRACTS)
from role_grouping import REFERENCE_POINTS, group_directory
from vision_assets import canonical_digest
from tools.tests.asset_link_fixtures import link_directory
spec = importlib.util.spec_from_file_location("identity_benchmark", ROOT / "tools/vision-identity-benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class IdentityContractTests(unittest.TestCase):
    def fixture(self, directory):
        source = directory / "source/net.py"
        weight = directory / "weights/adaface_ir101_webface12m.ckpt"
        source.parent.mkdir(parents=True)
        weight.parent.mkdir(parents=True)
        source.write_bytes(b"source fixture; never executed")
        weight.write_bytes(b"checkpoint fixture; never loaded")
        contract = dict(IDENTITY_CONTRACTS["AdaFace"], weight_sha256=file_sha256(weight))
        identity = {"model": "AdaFace", "official_source": contract["official_source"],
                    "revision": contract["revision"], "license": "MIT",
                    "source_files": [{"path": "source/net.py", "sha256": file_sha256(source)}],
                    "weights": [{"path": contract["weight_path"], "sha256": file_sha256(weight)}]}
        contract["source_digest"] = canonical_digest({"source_files": identity["source_files"], "license": identity["license"]})
        (directory / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
        return identity, contract

    def test_shared_crop_requires_uint8_112_rgb(self):
        for image in (np.zeros((112, 112), np.uint8), np.zeros((224, 224, 3), np.uint8),
                      np.zeros((112, 112, 3), np.float32)):
            with self.assertRaises(ValueError):
                validate_crop(image)

    def test_invalid_descriptors_rejected(self):
        for vector in ([0, 0], [np.nan, 1], [np.inf, 1]):
            with self.assertRaises(ValueError):
                normalized(vector)
        np.testing.assert_allclose(normalized([3, 4]), [.6, .8])

    def test_unlabelled_and_duplicate_media_do_not_make_accuracy_pairs(self):
        samples = [{"id": "a", "identity_label": "same", "sha256": "hash1"},
                   {"id": "b", "identity_label": "same", "sha256": "hash2"},
                   {"id": "copy", "identity_label": "same", "sha256": "hash1"},
                   {"id": "unknown", "identity_label": None, "sha256": "hash3"}]
        pairs = benchmark.labelled_pairs({item["id"]: [1, 0] for item in samples}, samples)
        self.assertEqual([(p["left"], p["right"]) for p in pairs], [("a", "b")])
        self.assertTrue(all(p["same_identity"] for p in pairs))

    def test_empty_and_duplicate_id_manifests_cannot_report_complete(self):
        for manifest in ({"samples": []}, {"samples": [{"id": "a"}, {"id": "a"}]}):
            with self.assertRaisesRegex(ValueError, "nonempty samples with unique IDs"):
                benchmark.manifest_crops(manifest)

    def test_asset_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            identity, contract = self.fixture(directory)
            with patch.dict(IDENTITY_CONTRACTS, {"AdaFace": contract}):
                self.assertEqual(verified_identity(directory, "AdaFace"), identity)
                (directory / "source/net.py").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    verified_identity(directory, "AdaFace")

    def test_absolute_parent_escape_and_symlink_source_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            directory = parent / "assets"
            directory.mkdir()
            identity, contract = self.fixture(directory)
            outside = parent / "outside.py"
            outside.write_bytes(b"outside")
            paths = [str(outside), "../outside.py"]
            link = directory / "source/escape.py"
            try:
                link.symlink_to(outside)
                paths.append("source/escape.py")
            except OSError:
                pass  # The absolute/traversal checks still run without link privileges.
            with patch.dict(IDENTITY_CONTRACTS, {"AdaFace": contract}):
                for path in paths:
                    changed = dict(identity, source_files=identity["source_files"] + [{"path": path, "sha256": file_sha256(outside)}])
                    (directory / "identity.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        verified_identity(directory, "AdaFace")

    def test_execution_source_must_be_registered_and_model_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            identity, contract = self.fixture(directory)
            with patch.dict(IDENTITY_CONTRACTS, {"AdaFace": contract}):
                with self.assertRaisesRegex(ValueError, "model does not match"):
                    verified_identity(directory, "MagFace")
                identity["source_files"] = []
                (directory / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Required execution source"):
                    verified_identity(directory, "AdaFace")

    def test_rewritten_source_and_manifest_cannot_change_reviewed_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            identity, contract = self.fixture(directory)
            (directory / "source/net.py").write_bytes(b"modified executable")
            identity["source_files"][0]["sha256"] = file_sha256(directory / "source/net.py")
            (directory / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            with patch.dict(IDENTITY_CONTRACTS, {"AdaFace": contract}):
                with self.assertRaisesRegex(ValueError, "reviewed source and license fingerprint"):
                    verified_identity(directory, "AdaFace")

    def test_registered_symlink_cannot_escape_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            directory = parent / "assets"
            directory.mkdir()
            identity, contract = self.fixture(directory)
            source = directory / "source/net.py"
            outside_dir = parent / "outside-source"
            outside_dir.mkdir()
            outside = outside_dir / "net.py"
            outside.write_bytes(source.read_bytes())
            source.unlink()
            source.parent.rmdir()
            try:
                link_directory(source.parent, outside_dir)
            except (OSError, Exception) as error:
                self.skipTest(f"Directory links unavailable: {type(error).__name__}")
            with patch.dict(IDENTITY_CONTRACTS, {"AdaFace": contract}):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    verified_identity(directory, "AdaFace")

    def test_sface_unpinned_weights_rejected_before_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.onnx"
            path.write_bytes(b"not the approved weights")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                SFaceTorch(path)


@unittest.skipUnless(SFACE_PATH.is_file(), "Optional pinned SFace asset absent")
class SFaceRuntimeTests(unittest.TestCase):
    def test_real_graph_numerically_matches_reference(self):
        crops = {f"random-{seed}": np.random.default_rng(seed).integers(0, 256, (112, 112, 3), dtype=np.uint8)
                 for seed in range(8)}
        result = benchmark.equivalence(crops)
        self.assertTrue(result["passed"], result)

    def test_role_grouping_preserves_reference_group_and_scores(self):
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        crops = [np.random.default_rng(9).integers(0, 256, (112, 112, 3), dtype=np.uint8)] * 2
        crops.append(np.random.default_rng(10).integers(0, 256, (112, 112, 3), dtype=np.uint8))
        points = np.zeros((68, 2), np.float32)
        points[36:42], points[42:48] = REFERENCE_POINTS[0], REFERENCE_POINTS[1]
        points[30], points[48], points[54] = REFERENCE_POINTS[2:]
        class DFL:
            def get_landmarks(self): return points
            def get_source_rect(self): return [0, 0, 112, 112]
            def get_source_filename(self): return None
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for index, image in enumerate(crops):
                cv2.imwrite(str(directory / f"face{index}.png"), image)
            result = group_directory(directory, SFACE_PATH, lambda _: DFL(), lambda *args: None, threshold=.85)
            self.assertEqual(result["method"], "sface-2021dec-complete-linkage")
            self.assertEqual(result["analyzedCount"], 3)
            self.assertEqual(result["invalidCount"], 0)
            self.assertEqual(sorted(len(group["members"]) for group in result["groups"]), [1, 2])
            duplicated = next(group for group in result["groups"] if len(group["members"]) == 2)
            self.assertTrue(all(member["score"] == 1 for member in duplicated["members"]))


if __name__ == "__main__":
    unittest.main()
