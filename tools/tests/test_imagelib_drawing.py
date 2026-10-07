from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.imagelib.draw import draw_polygon, draw_rect


class DebugDrawingTests(unittest.TestCase):
    def test_float_detector_rect_draws_closed_outline(self):
        image = np.zeros((24, 24, 3), np.uint8)
        rect = np.array([2.2, 3.2, 18.2, 19.2], dtype=np.float32)
        draw_rect(image, rect, (255, 0, 0))
        self.assertTrue(np.all(image[3, 2:19, 0] == 255))
        self.assertTrue(np.all(image[19, 2:19, 0] == 255))
        self.assertTrue(np.all(image[3:20, 2, 0] == 255))
        self.assertTrue(np.all(image[3:20, 18, 0] == 255))
        self.assertEqual(int(image[10, 10].sum()), 0)

    def test_subpixel_polygon_clips_at_image_boundary(self):
        image = np.zeros((12, 12, 3), np.uint8)
        draw_polygon(image, [(-2.2, 2.4), (8.2, 2.4), (8.2, 8.2)], (0, 255, 0))
        self.assertTrue(np.all(image[2, :9, 1] == 255))
        self.assertTrue(np.all(image[2:9, 8, 1] == 255))


if __name__ == '__main__':
    unittest.main()
