import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui/python"))
from vision_landmarks import LANDMARK_CONTRACTS, file_sha256, point_definition, regression_denormalize, square_affine, transform_points, validate_box, verified_identity
from vision_assets import canonical_digest
from tools.tests.asset_link_fixtures import link_directory

spec = importlib.util.spec_from_file_location("landmark_benchmark", ROOT / "tools/vision-landmark-benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GeometryContractTests(unittest.TestCase):
    def fixture(self, root):
        (root / "source").mkdir()
        (root / "source/Prompt.py").write_bytes(b"official source fixture; never executed")
        (root / "source/LICENSE").write_bytes(b"test license")
        (root / "weight.bin").write_bytes(b"official weights")
        identity = {"model": "TUFA", "repository": LANDMARK_CONTRACTS["TUFA"]["repository"],
                    "revision": LANDMARK_CONTRACTS["TUFA"]["revision"], "license": "test",
                    "weights": [{"file": "weight.bin", "sha256": file_sha256(root / "weight.bin"), "bytes": 16}],
                    "source_files": {"Prompt.py": file_sha256(root / "source/Prompt.py")},
                    "source_license_record": "source/LICENSE", "official_weight_url": "source", "distribution": "evaluation"}
        contract = dict(LANDMARK_CONTRACTS["TUFA"], required_sources=("Prompt.py",), weights=identity["weights"],
                        source_digest=canonical_digest({"source_files": identity["source_files"], "license": identity["license"]}),
                        license_sha256=file_sha256(root / "source/LICENSE"))
        (root / "identity.json").write_text(json.dumps(identity))
        return identity, contract

    def test_regression_official_pixel_center_inverse(self):
        points = np.array([[-1., -1.], [0., 0.], [1., 1.]])
        np.testing.assert_array_equal(regression_denormalize(points),
                                      [[-.5, -.5], [127.5, 127.5], [255.5, 255.5]])

    def test_random_and_out_of_frame_affine_roundtrips(self):
        rng = np.random.default_rng(51)
        for _ in range(50):
            origin = rng.uniform(-5000, 5000, 2)
            box = np.r_[origin, origin + rng.uniform(1, 2000, 2)]
            matrix = square_affine(box, 1.2)
            points = rng.uniform(-5000, 5000, (98, 2))
            np.testing.assert_allclose(transform_points(transform_points(points, matrix), cv2.invertAffineTransform(matrix)), points, atol=1e-9)
            np.testing.assert_allclose(transform_points(np.array([(box[:2]+box[2:])/2]), matrix), [[128, 128]])

    def test_invalid_boxes_and_nonfinite_points_are_rejected(self):
        for box in ([0, 0, 0, 10], [0, 1, 10, 0], [0, 0, float("nan"), 2], [0, 1, 2]):
            with self.assertRaises(ValueError):
                validate_box(box)
        with self.assertRaises(ValueError):
            transform_points([[float("inf"), 1]], np.eye(2, 3))

    def test_native_topologies_stay_distinct_and_do_not_claim_3d(self):
        a, b = point_definition(98), point_definition(68)
        self.assertEqual(a["name"], "WFLW98")
        self.assertEqual(b["name"], "IBUG68")
        self.assertEqual(len(a["regions"]["image_left_eye"]), 8)
        self.assertEqual(len(b["regions"]["image_left_eye"]), 6)
        self.assertFalse(a["three_dimensional"])
        with self.assertRaises(ValueError):
            point_definition(314)

    def test_asset_source_and_weights_are_bound_to_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity, contract = self.fixture(root)
            with patch.dict(LANDMARK_CONTRACTS, {"TUFA": contract}):
                self.assertEqual(verified_identity(root)["model"], "TUFA")
                (root / "source/Prompt.py").write_text("changed source")
                with self.assertRaisesRegex(ValueError, "source identity"):
                    verified_identity(root)

    def test_rewritten_source_manifest_and_license_cannot_repin_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity, contract = self.fixture(root)
            (root / "source/Prompt.py").write_bytes(b"different source")
            identity["source_files"]["Prompt.py"] = file_sha256(root / "source/Prompt.py")
            (root / "identity.json").write_text(json.dumps(identity))
            with patch.dict(LANDMARK_CONTRACTS, {"TUFA": contract}):
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    verified_identity(root)

    def test_required_source_and_outside_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "assets"
            root.mkdir()
            identity, contract = self.fixture(root)
            with patch.dict(LANDMARK_CONTRACTS, {"TUFA": contract}):
                missing = dict(identity, source_files={})
                (root / "identity.json").write_text(json.dumps(missing))
                with self.assertRaisesRegex(ValueError, "Required landmark execution"):
                    verified_identity(root)
                (root / "identity.json").write_text(json.dumps(identity))
                source = root / "source/Prompt.py"
                outside_dir = parent / "outside-source"
                outside_dir.mkdir()
                outside = outside_dir / "Prompt.py"
                outside.write_bytes(source.read_bytes())
                source.unlink()
                (source.parent / "LICENSE").unlink()
                source.parent.rmdir()
                try:
                    link_directory(source.parent, outside_dir)
                except (OSError, Exception) as error:
                    self.skipTest(f"Directory links unavailable: {type(error).__name__}")
                with self.assertRaisesRegex(ValueError, "escapes"):
                    verified_identity(root)


class QualityMetricTests(unittest.TestCase):
    def test_identity_symmetry_translation_and_subdivision(self):
        a = np.array([[0, 0], [10, 0]])
        b = a + [0, 3]
        d = benchmark.symmetric_curve_distance
        self.assertAlmostEqual(d(a, False, a, False), 0)
        self.assertAlmostEqual(d(a, False, b, False), 3)
        self.assertAlmostEqual(d(b, False, a, False), 3)
        self.assertAlmostEqual(d([[0, 0], [5, 0], [10, 0]], False, b, False), 3)

    def test_excluded_regions_and_empty_references_do_not_invent_scores(self):
        prediction = {"points_original": np.zeros((98, 2)).tolist(), "point_definition": point_definition(98)}
        result = benchmark.score_prediction(prediction, [{"region": "left_eye", "excluded": True}], [0, 0, 10, 10])
        self.assertIsNone(result["image_quality_error_head256_px"])
        self.assertEqual(result["regions"], [])

    def test_bootstrap_pairs_only_shared_cases_and_is_reproducible(self):
        records = [{"model_id": model, "sample_id": case, "quality": {"image_quality_error_head256_px": value}} for model, case, value in [("tufa98", "a", 1), ("tufa98", "b", 2), ("orformer98", "a", 4), ("orformer98", "c", 99)]]
        a = benchmark.paired_uncertainty(records, repeats=100)
        self.assertEqual(a, benchmark.paired_uncertainty(records, repeats=100))
        self.assertEqual(a[0]["paired_images"], 1)
        self.assertEqual(a[0]["mean_error_difference_head256_px"], 3)
        self.assertEqual(a[0]["conditional_bootstrap_95pct"], [3, 3])


if __name__ == "__main__":
    unittest.main()
