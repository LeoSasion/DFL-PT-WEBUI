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


class SimilarityContract(unittest.TestCase):
    def inspect(self, count=503, **options):
        files = [Path(f"{index:04d}.png") for index in range(count)]
        read = []

        def image(target, _flags):
            read.append(Path(target).name)
            index = int(Path(target).stem)
            return np.full((4, 4, 3), index, dtype=np.float32)

        def descriptor(image):
            # Every sample is an exact duplicate here; production decoding and
            # descriptors are checked separately with real, temporary PNG files.
            return np.array([1, 0], dtype=np.float32)

        with patch.object(tools, "iter_images", return_value=files), \
                patch.object(tools.cv2, "imread", side_effect=image), \
                patch.object(tools, "similarity_descriptor", side_effect=descriptor), \
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
                             np.full((2, 2, 3), int(Path(name).stem))), \
                patch.object(tools, "similarity_descriptor", side_effect=lambda image:
                             np.array([1, 0]) if image[0, 0, 0] < 2 else np.array([0, 1])), \
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


if __name__ == "__main__":
    unittest.main()
