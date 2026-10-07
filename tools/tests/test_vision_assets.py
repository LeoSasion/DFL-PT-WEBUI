import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from vision_assets import StageCache, atomic_json, canonical_digest, file_sha256, resolve_asset_path, verified_asset, prepare_evaluation_output


class VisionAssetTests(unittest.TestCase):
    def test_evidence_cannot_replace_source_or_previous_run(self):
        with tempfile.TemporaryDirectory() as directory, patch("vision_assets.PROJECT_ROOT", Path(directory)):
            source = Path(directory) / "source.png"
            source.write_bytes(b"unchanged-source")
            with self.assertRaises(ValueError):
                prepare_evaluation_output(Path(directory))
            self.assertEqual(source.read_bytes(), b"unchanged-source")
            output = Path(directory) / "workspace/.vision-evaluation/run1"
            self.assertEqual(prepare_evaluation_output(output), output.resolve())
            with self.assertRaises(ValueError):
                prepare_evaluation_output(output)
            with self.assertRaises(ValueError):
                prepare_evaluation_output(output.parent)

    def test_stage_dependencies_and_result_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory)
            deps = {"sourceSha256": "abc", "modelSha256": "def", "geometry": [1, 0, 0, 1], "precision": "fp32"}
            self.assertIsNone(cache.read("detect", deps))
            key = cache.write("detect", deps, {"boxes": [[1, 2, 3, 4]]})
            self.assertEqual(cache.read("detect", deps), {"boxes": [[1, 2, 3, 4]]})
            self.assertIsNone(cache.read("detect", {**deps, "modelSha256": "new"}))
            path = Path(directory) / "detect" / (key + ".json")
            record = json.loads(path.read_text())
            record["result"]["boxes"][0][0] = 99
            atomic_json(path, record)
            self.assertIsNone(cache.read("detect", deps))
            record["status"] = "cancelled"
            atomic_json(path, record)
            self.assertIsNone(cache.read("detect", deps))

    def test_no_partial_or_nonfinite_results(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory)
            with self.assertRaises(ValueError):
                cache.write("detect", {}, {"score": float("nan")})
            self.assertEqual(list(Path(directory).rglob("*.json")), [])
            with self.assertRaises(ValueError):
                cache.write("../escape", {}, {})

    def test_relative_paths_and_weight_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "detectors" / "test.bin"
            path.parent.mkdir()
            path.write_bytes(b"fixed-model-bytes")
            entry = {"id": "test", "path": "detectors/test.bin", "sizeBytes": path.stat().st_size, "sha256": file_sha256(path)}
            atomic_json(path.parent / "assets.json", {"schemaVersion": 1, "models": [entry]})
            self.assertEqual(verified_asset("test", assets_root=root)[0], path.resolve())
            path.write_bytes(b"changed-model-bytes")
            with self.assertRaises(ValueError):
                verified_asset("test", assets_root=root)
            for relative in ("../outside", str(root / "absolute")):
                with self.assertRaises(ValueError):
                    resolve_asset_path(root, relative)

    def test_digest_is_order_independent_and_finite(self):
        self.assertEqual(canonical_digest({"a": 1, "b": 2}), canonical_digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical_digest({"v": float("inf")})


if __name__ == "__main__":
    unittest.main()
