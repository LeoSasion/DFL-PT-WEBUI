"""Exercise the ME preview file bridge against real PNG pixels."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal/DeepFaceLab'))
from me_backend.web_bridge import atomic_image

with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / 'nested/preview.png'
    sample = np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 2, 0.5]]], dtype=np.float32)
    atomic_image(target, sample)
    decoded = cv2.imread(str(target))
    np.testing.assert_array_equal(decoded, [[[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 128]]])
    atomic_image(target, np.zeros_like(sample))
    assert not cv2.imread(str(target)).any(), 'A refreshed preview must replace the previous PNG'
    assert list(target.parent.iterdir()) == [target], 'Preview must leave no pending files'
print('ME preview preserves BGR colors, clipping, and atomic refresh')
