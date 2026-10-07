import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'webui/python'))
from dfl_asset_tool import inspect_native_landmarks


class NativeLandmarkInspectionTests(unittest.TestCase):
    def setUp(self):
        self.original = np.arange(196, dtype=np.float32).reshape(98, 2)
        self.matrix = np.array([[0.5, 0, 3], [0, 0.5, 7]], np.float32)
        self.raw = {'points_original': self.original.tolist(),
                    'points_aligned': (self.original * .5 + [3, 7]).tolist(),
                    'aligned_canvas_wh': [128, 128],
                    'source_to_aligned_affine': self.matrix.tolist(),
                    'point_definition': {'name': 'WFLW98', 'count': 98},
                    'model_id': 'tufa98'}

    def image(self, matrix=None):
        return SimpleNamespace(get_dict=lambda: {'native_landmarks98': self.raw},
                               get_image_to_face_mat=lambda: self.matrix if matrix is None else matrix)

    def test_independent98_is_exposed_with_no_visibility_claim(self):
        value = inspect_native_landmarks(self.image(), 128, 128)
        self.assertTrue(value['available'])
        self.assertEqual(value['count'], 98)
        self.assertFalse(value['visibilityEstimated'])

    def test_manual_realignment_and_changed_canvas_disable_old_audit(self):
        moved = self.matrix.copy()
        moved[0, 2] += 1
        self.assertFalse(inspect_native_landmarks(self.image(moved), 128, 128)['available'])
        self.assertFalse(inspect_native_landmarks(self.image(), 256, 256)['available'])

    def test_bad_topology_or_nonfinite_coordinates_do_not_break_inspection(self):
        self.raw['point_definition']['name'] = 'IBUG68'
        self.assertFalse(inspect_native_landmarks(self.image(), 128, 128)['available'])
        self.raw['point_definition']['name'] = 'WFLW98'
        self.raw['points_original'][0][0] = float('nan')
        self.assertFalse(inspect_native_landmarks(self.image(), 128, 128)['available'])


if __name__ == '__main__':
    unittest.main()
