"""Sparse eye/mouth dilation keeps the original complete mask bit-for-bit."""
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from facelib import LandmarksProcessor
from tae.config import TrainConfig
from me_backend.data import read_aligned
from tests.me_fixtures import make_aligned


def full_dilation(mask, kernel):
    return cv2.dilate(mask, kernel, iterations=1)


@pytest.mark.parametrize('shape', [(256, 256), (448, 448), (512, 512), (640, 640), (832, 832), (512, 640)])
def test_sparse_dilation_preserves_full_eye_and_mouth_masks_for_boundaries_and_random_landmarks(shape):
    height, width = shape
    canonical = LandmarksProcessor.get_canonical_68(height)
    canonical[:, 0] *= width / height
    rng = np.random.default_rng(height + width)
    cases = [canonical, canonical - height * .4, canonical + height * .4,
        canonical + height * 2, np.full_like(canonical, height / 2),
        canonical + rng.normal(0, height * .02, canonical.shape),
        rng.uniform(-height * .5, height * 1.5, canonical.shape)]
    for landmarks in cases:
        for function in [LandmarksProcessor.get_image_eye_mask, LandmarksProcessor.get_image_mouth_mask]:
            with patch.object(LandmarksProcessor, '_dilate_sparse_mask', full_dilation):
                expected = function((height, width, 3), landmarks)
            actual = function((height, width, 3), landmarks)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.float32 and actual.shape == (height, width, 1)


@pytest.mark.parametrize('resolution', [256, 448, 512, 640, 832])
@pytest.mark.parametrize('face_type', ['h', 'mf', 'f', 'wf', 'head'])
def test_aligned_rgb_full_and_encoded_masks_remain_exact_for_all_crop_types(tmp_path, resolution, face_type):
    folder = make_aligned(tmp_path / 'aligned', count=2, size=resolution)
    settings = TrainConfig(resolution=resolution, face_type=face_type, inter_dims=8, hidden_dims=8)
    for path in sorted(folder.glob('*.jpg')):
        with patch.object(LandmarksProcessor, '_dilate_sparse_mask', full_dilation):
            expected = read_aligned(path, settings)
        actual = read_aligned(path, settings)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('kind', ['empty', 'dense', 'upper-left', 'lower-right', 'center', 'non-contiguous'])
def test_sparse_dilation_matches_full_frame_for_binary_mask_extents(kind):
    mask = np.zeros((512, 640, 1), np.float32)
    if kind == 'dense':
        mask[:] = 1
    elif kind == 'upper-left':
        mask[:20, :30] = 1
    elif kind == 'lower-right':
        mask[-20:, -30:] = 1
    elif kind in ['center', 'non-contiguous']:
        mask[240:260, 300:330] = 1
    if kind == 'non-contiguous':
        mask = mask[:, ::2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    expected = full_dilation(mask, kernel)
    actual = LandmarksProcessor._dilate_sparse_mask(mask, kernel)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, mask)


def test_large_sparse_dilation_uses_bounded_region_and_small_kernel_retains_native_path():
    mask = np.zeros((512, 512, 1), np.float32)
    mask[250:260, 240:270] = 1
    native = cv2.dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    with patch.object(cv2, 'dilate', wraps=native) as dilated:
        actual = LandmarksProcessor._dilate_sparse_mask(mask, kernel)
    assert dilated.call_count == 1
    assert np.prod(dilated.call_args.args[0].shape[:2]) < np.prod(mask.shape[:2]) / 8
    np.testing.assert_array_equal(actual, native(mask, kernel, iterations=1))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    with patch.object(cv2, 'boundingRect', side_effect=AssertionError('small kernels retain the native path')):
        np.testing.assert_array_equal(LandmarksProcessor._dilate_sparse_mask(mask, small),
            native(mask, small, iterations=1))
