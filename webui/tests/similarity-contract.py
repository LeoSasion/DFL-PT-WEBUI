import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import dfl_asset_tool as tools
from face_quality_review import quality_review
from similarity_review import consistent_groups, same_scale_ssim, ssim_features


class SimilarityContract(unittest.TestCase):
    def inspect(self, count=503, **options):
        files = [Path(f"{index:04d}.png") for index in range(count)]
        read = []

        def image(target, _flags):
            read.append(Path(target).name)
            index = int(Path(target).stem)
            return np.full((4, 4, 3), index % 256, dtype=np.uint8)

        def descriptor(image):
            # Every sample is an exact duplicate here; production decoding and
            # descriptors are checked separately with real, temporary PNG files.
            return np.array([1, 0], dtype=np.float32)

        with patch.object(tools, "iter_images", return_value=files), \
                patch.object(tools.cv2, "imread", side_effect=image), \
                patch.object(tools, "similarity_descriptor", side_effect=descriptor), \
                patch.object(tools, "same_scale_ssim", return_value=1.0), \
                patch.object(tools, "load_dfl_image", side_effect=ValueError("synthetic pixels only")), \
                redirect_stderr(io.StringIO()):
            result = tools.group_similar_images(Path("unused"), **options)
        return result, read

    def test_large_dataset_can_reach_tail_without_exceeding_500(self):
        first, first_read = self.inspect(limit=1000)
        self.assertEqual(len(first_read), 500)
        self.assertEqual(first["windowSize"], 500)
        self.assertEqual(first["pageCount"], 2)
        self.assertTrue(first["hasNext"])
        tail, tail_read = self.inspect(offset=500)
        self.assertEqual(tail_read, ["0500.png", "0501.png", "0502.png"])
        self.assertEqual(tail["selectedCount"], 3)
        self.assertEqual(tail["windows"][0]["start"], 501)
        self.assertEqual(tail["windows"][0]["end"], 503)
        self.assertEqual(tail["pageIndex"], 1)
        self.assertFalse(tail["hasNext"])
        self.assertTrue(tail["hasPrevious"])
        self.assertFalse(set(first_read).intersection(tail_read))

    def test_paired_scope_is_two_distinct_250_item_windows(self):
        result, read = self.inspect(offset=0, compare_offset=250)
        self.assertEqual(len(read), 500)
        self.assertEqual(result["mode"], "paired")
        self.assertEqual(result["windowSize"], 250)
        self.assertEqual(result["pageCount"], 3)
        self.assertEqual([window["count"] for window in result["windows"]], [250, 250])
        self.assertEqual(len(set(read)), 500)
        self.assertTrue(all(group["crossBatch"] for group in result["groups"]))
        self.assertEqual({member["batch"] for member in result["groups"][0]["members"]}, {0, 1})
        self.assertEqual(sum(member["representative"] for member in result["groups"][0]["members"]), 1)

    def test_small_paired_budget_and_tail_keep_actual_counts(self):
        result, read = self.inspect(offset=0, compare_offset=502, limit=6)
        self.assertEqual(len(read), 4)
        self.assertEqual(result["windowSize"], 3)
        self.assertEqual(result["windows"][1]["count"], 1)
        self.assertEqual(result["analyzedCount"], 4)

    def test_outside_inventory_is_an_empty_window(self):
        result, read = self.inspect(offset=503)
        self.assertEqual(read, [])
        self.assertEqual(result["groups"], [])
        self.assertIsNone(result["windows"][0]["start"])
        self.assertEqual(result["selectedCount"], 0)

    def test_invalid_parameters_and_overlapping_windows_are_rejected(self):
        for options in [{"offset": -1}, {"offset": 1.5}, {"offset": True},
                        {"compare_offset": -1}, {"compare_offset": 1.5},
                        {"compare_offset": True}, {"compare_offset": 0},
                        {"compare_offset": 249}, {"threshold": float("nan")},
                        {"threshold": float("inf")}, {"limit": float("inf")},
                        {"limit": 2.5}, {"limit": True}]:
            with self.subTest(options=options), self.assertRaises((ValueError, OverflowError)):
                self.inspect(**options)

    def test_only_groups_with_both_windows_are_returned_in_paired_mode(self):
        files = [Path(f"{index:04d}.png") for index in range(4)]
        with patch.object(tools, "iter_images", return_value=files), \
                patch.object(tools.cv2, "imread", side_effect=lambda name, flags:
                             np.full((2, 2, 3), int(Path(name).stem), dtype=np.uint8)), \
                patch.object(tools, "similarity_descriptor", side_effect=lambda image:
                             np.array([1, 0]) if image[0, 0, 0] < 2 else np.array([0, 1])), \
                patch.object(tools, "same_scale_ssim", return_value=1.0), \
                patch.object(tools, "load_dfl_image", side_effect=ValueError("synthetic pixels only")), \
                redirect_stderr(io.StringIO()):
            batch = tools.group_similar_images(Path("unused"), limit=4)
            paired = tools.group_similar_images(Path("unused"), limit=4, compare_offset=2)
        self.assertEqual(batch["groupCount"], 2)
        self.assertEqual(paired["groupCount"], 0)
        self.assertEqual(paired["ungroupedCount"], 4)

    def test_real_decode_is_read_only_and_counts_invalid_tail_image(self):
        with tempfile.TemporaryDirectory(prefix="dfl-similarity-contract-") as temporary:
            directory = Path(temporary)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.rectangle(image, (6, 6), (20, 22), (10, 180, 220), -1)
            for index in range(4):
                self.assertTrue(cv2.imwrite(str(directory / f"{index:04d}.png"), image))
            (directory / "0004.png").write_bytes(b"invalid-image")
            before = {target.name: (target.read_bytes(), target.stat().st_mtime_ns)
                      for target in directory.iterdir()}
            with redirect_stderr(io.StringIO()):
                result = tools.group_similar_images(directory, limit=6, offset=0, compare_offset=3)
            self.assertEqual(result["selectedCount"], 5)
            self.assertEqual(result["analyzedCount"], 4)
            self.assertEqual(result["invalidCount"], 1)
            self.assertEqual(result["windows"][1]["invalidCount"], 1)
            self.assertTrue(result["groups"][0]["crossBatch"])
            after = {target.name: (target.read_bytes(), target.stat().st_mtime_ns)
                     for target in directory.iterdir()}
            self.assertEqual(before, after)

    def test_a_b_c_chain_is_not_a_group_even_when_b_matches_both_ends(self):
        accepted = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
        for order in [[0, 1, 2], [1, 0, 2], [2, 1, 0]]:
            groups = consistent_groups(accepted, order)
            self.assertEqual(sorted(map(len, groups)), [1, 2])
            for group in groups:
                self.assertTrue(all(accepted[left, right] for left in group for right in group))

    def test_dct_hsv_edge_collision_of_flat_dark_and_midgray_is_rejected_by_ssim(self):
        with tempfile.TemporaryDirectory(prefix="dfl-similarity-semantic-") as temporary:
            directory = Path(temporary)
            black = np.full((128, 128, 3), 8, dtype=np.uint8)
            white = np.full_like(black, 128)
            self.assertGreater(float(np.dot(tools.similarity_descriptor(black), tools.similarity_descriptor(white))), 0.98)
            cv2.imwrite(str(directory / "black.png"), black)
            cv2.imwrite(str(directory / "white.png"), white)
            with redirect_stderr(io.StringIO()):
                result = tools.group_similar_images(directory)
            self.assertEqual(result["verification"]["candidatePairCount"], 1)
            self.assertEqual(result["verification"]["verifiedPairCount"], 0)
            self.assertEqual(result["groups"], [])

    def test_grouped_representative_preserves_the_better_quality_member(self):
        with tempfile.TemporaryDirectory(prefix="dfl-similarity-quality-") as temporary:
            directory = Path(temporary)
            image = np.full((128, 128, 3), 125, dtype=np.uint8)
            cv2.rectangle(image, (25, 25), (100, 100), (80, 80, 80), -1)
            cv2.imwrite(str(directory / "a-dim.png"), np.uint8(image.astype(np.float32) * 0.85))
            cv2.imwrite(str(directory / "z-clear.png"), image)
            with redirect_stderr(io.StringIO()):
                result = tools.group_similar_images(directory, threshold=0.86)
            self.assertEqual(result["groupCount"], 1)
            group = result["groups"][0]
            self.assertEqual(group["representativeName"], "z-clear.png")
            self.assertGreaterEqual(group["minimumSsim"], result["threshold"])
            self.assertGreaterEqual(group["minimumScore"], result["threshold"])
            self.assertEqual(group["consistency"], "all-pairs-verified")
            self.assertTrue(all("qualityEvidence" in member for member in group["members"]))


class SsimSemantics(unittest.TestCase):
    def test_identical_pixels_at_different_original_resolutions_share_a_scale(self):
        image = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        larger = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.assertAlmostEqual(same_scale_ssim(ssim_features(image), ssim_features(larger)), 1.0, places=8)

    def test_uniform_luminance_matches_ssim_analytic_formula(self):
        left = np.full((64, 64, 3), 100, dtype=np.uint8)
        right = np.full_like(left, 150)
        expected = (2 * 100 * 150 + 2.55 ** 2) / (100 ** 2 + 150 ** 2 + 2.55 ** 2)
        self.assertAlmostEqual(same_scale_ssim(ssim_features(left), ssim_features(right)), expected, places=6)

    def test_same_shape_histogram_but_different_structure_is_not_a_duplicate(self):
        image = np.full((128, 128, 3), 30, dtype=np.uint8)
        image[20:70, 15:55] = 220
        shifted = np.roll(image, 50, axis=1)
        self.assertLess(same_scale_ssim(ssim_features(image), ssim_features(shifted)), 0.86)

    def test_distinct_aspect_ratios_are_not_warped_into_duplicates(self):
        self.assertEqual(same_scale_ssim(ssim_features(np.zeros((128, 128, 3), dtype=np.uint8)),
                                        ssim_features(np.zeros((64, 128, 3), dtype=np.uint8))), 0.0)


class Metadata:
    def __init__(self, rect=None, landmarks=None, source_landmarks=None, matrix=None):
        self.rect, self.landmarks, self.source_landmarks, self.matrix = rect, landmarks, source_landmarks, matrix
    def get_source_rect(self): return self.rect
    def get_landmarks(self): return self.landmarks
    def get_source_landmarks(self): return self.source_landmarks
    def get_image_to_face_mat(self): return self.matrix


class QualitySemantics(unittest.TestCase):
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    metrics = {"sharpness": 0.8, "brightness": 0.5}
    points = np.column_stack((np.linspace(25, 95, 68), np.sin(np.linspace(0, 6, 68)) * 25 + 60))

    def test_missing_metadata_is_unavailable_and_never_guesses_occlusion(self):
        evidence = quality_review(self.image, self.metrics)
        self.assertFalse(evidence["components"]["sourceResolution"]["available"])
        self.assertIsNone(evidence["components"]["sourceResolution"]["score"])
        self.assertFalse(evidence["alignment"]["available"])
        self.assertFalse(evidence["pose"]["available"])
        self.assertFalse(evidence["occlusion"]["available"])
        self.assertAlmostEqual(evidence["score"], (0.55 * 0.8 + 0.20) / 0.75)

    def test_source_size_is_from_detection_metadata_and_does_not_confuse_upscaled_output(self):
        small = quality_review(self.image, self.metrics, Metadata(rect=[0, 0, 24, 30]))
        large = quality_review(self.image, self.metrics, Metadata(rect=[0, 0, 256, 300]))
        self.assertEqual(small["sourceFace"]["minimumDimension"], 24)
        self.assertIn("small_source_face", small["flags"])
        self.assertIn("strong_source_upscale", small["flags"])
        self.assertGreater(large["score"], small["score"])

    def test_geometry_residual_detects_inconsistent_source_to_aligned_metadata(self):
        valid = Metadata(rect=[0, 0, 128, 128], landmarks=self.points,
                         source_landmarks=self.points, matrix=[[1, 0, 0], [0, 1, 0]])
        good = quality_review(self.image, self.metrics, valid)
        self.assertTrue(good["alignment"]["available"])
        self.assertAlmostEqual(good["alignment"]["residualFraction"], 0)
        valid.source_landmarks = self.points + 12
        wrong = quality_review(self.image, self.metrics, valid)
        self.assertIn("source_alignment_mismatch", wrong["flags"])
        self.assertLess(wrong["score"], good["score"])

    def test_pose_is_an_estimate_and_profile_views_are_not_penalized(self):
        metadata = Metadata(landmarks=self.points)
        frontal = quality_review(self.image, self.metrics, metadata, lambda *a, **k: (0, 0, 0))
        profile = quality_review(self.image, self.metrics, metadata, lambda *a, **k: (0, -1.2, 0))
        self.assertEqual(frontal["score"], profile["score"])
        self.assertTrue(profile["pose"]["available"])
        self.assertGreater(profile["pose"]["yaw"], 60)

    def test_nonfinite_and_degenerate_metadata_does_not_become_truth(self):
        metadata = Metadata(rect=[0, 0, float("nan"), 120], landmarks=np.zeros((68, 2)),
                            source_landmarks=self.points, matrix=[[0, 0, 0], [0, 0, 0]])
        result = quality_review(self.image, self.metrics, metadata)
        self.assertFalse(result["sourceFace"]["available"])
        self.assertFalse(result["pose"]["available"])
        self.assertFalse(result["alignment"]["available"])
        self.assertIn("degenerate_landmarks", result["flags"])
        self.assertIn("degenerate_alignment_transform", result["flags"])
        import json
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
