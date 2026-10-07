"""Review completeness and ordinal scoring must not silently invent accuracy."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("visual_review_summary", ROOT / "tools/summarize-vision-visual-review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def regions(left, right, mouth):
    return {key: {"score": value, "confidence": "medium", "reason": "visible evidence"}
            for key, value in zip(review.REGIONS, (left, right, mouth))}


def visibility(left=True, right=True, mouth=True):
    return {key: {"scorable": value, "reason": "source-only boundary"}
            for key, value in zip(review.REGIONS, (left, right, mouth))}


class VisualReviewTests(unittest.TestCase):
    def test_ordinal_scale_rejects_fraction_bool_and_nonfinite(self):
        for value in (True, False, -.1, 5, 2.5, float("nan"), "4"):
            with self.assertRaises(ValueError):
                review.score(value)
        self.assertEqual(review.score(0), 0)
        self.assertEqual(review.score(4), 4)
        self.assertIsNone(review.score(None))

    def test_two_eyes_do_not_double_weight_mouth(self):
        result = review.landmark_case(regions(4, 2, 0), visibility())
        self.assertEqual(result, {"eyes": 3.0, "mouth": 0, "overall": 1.5})
        self.assertEqual(review.landmark_case(regions(4, None, None), visibility(True, False, False))["overall"], 4)

    def test_unknown_region_is_neither_failure_nor_selective_exclusion(self):
        with self.assertRaises(ValueError):
            review.landmark_case(regions(None, 2, 3), visibility())
        with self.assertRaises(ValueError):
            review.landmark_case(regions(4, 2, 3), visibility(False, True, True))

    def test_paired_interval_uses_shared_cases_and_fixed_seed(self):
        rows = [{"model_id": m, "case": i, "overall": v}
                for m, i, v in (("tufa98", 1, 3), ("tufa98", 2, 1), ("other", 1, 4), ("other", 2, 2), ("other", 3, 0))]
        a = review.paired_intervals(rows)
        self.assertEqual(a, review.paired_intervals(rows))
        self.assertEqual(a[0]["pairedCases"], 2)
        self.assertEqual(a[0]["visualIndexDifference"], 25)
        self.assertEqual(a[0]["conditional95Interval"], [25, 25])

    def test_coverage_blinding_and_image_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as task_dir:
            directory = Path(task_dir)
            path = directory / "sheet.png"
            path.write_bytes(b"locked sheet")
            manifest = {"cases": [{"case": 1, "sheet": {"file": "sheet.png", "sha256": review.file_sha256(path)}, "tokens": ["D01-1"]}]}
            record = {"assessor": {"model": "gpt-6-astra", "reasoningEffort": "low", "identitiesHidden": True,
                                   "previousScoresHidden": True, "visualReviewNotGroundTruth": True},
                      "cases": [{"case": 1, "inspectedFiles": ["sheet.png"], "candidates": [{"token": "D01-1", "faceCoverageScore": 4,
                                  "boxPlacementScore": 3, "confidence": "high", "reason": "visible face covered"}]}]}
            self.assertEqual(len(review.validate_review("detection", manifest, record, directory)), 1)
            record["cases"][0]["candidates"].append(record["cases"][0]["candidates"][0].copy())
            with self.assertRaises(ValueError):
                review.validate_review("detection", manifest, record, directory)
            record["cases"][0]["candidates"].pop()
            record["assessor"]["previousScoresHidden"] = False
            with self.assertRaises(ValueError):
                review.validate_review("detection", manifest, record, directory)
            record["assessor"]["previousScoresHidden"] = True
            path.write_bytes(b"changed sheet")
            with self.assertRaises(ValueError):
                review.validate_review("detection", manifest, record, directory)


if __name__ == "__main__":
    unittest.main()
