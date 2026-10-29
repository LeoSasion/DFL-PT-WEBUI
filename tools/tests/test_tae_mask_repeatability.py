"""Exact feather geometry and repeated TAE updates on isolated synthetic data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
from tae.config import TrainConfig
from tae.engine import TAEEngine
from tae.masking import expansion_geometry, image_loss_mask
from tae.vae import DEFAULT_WEIGHTS
from tae.checkpoint import frozen_digest


@pytest.mark.parametrize('resolution', [256, 448, 512, 640, 832])
def test_feather_matches_single_pixel_euclidean_geometry_bit_for_bit(resolution):
    mask = np.zeros((1, 1, resolution, resolution), np.float32)
    center = resolution // 2
    mask[0, 0, center, center] = 1
    radius, _ = expansion_geometry(resolution)
    yy, xx = np.mgrid[:resolution, :resolution]
    # Independent analytic oracle: the only nearest face pixel is the center.
    distance = np.sqrt(((yy-center)**2 + (xx-center)**2).astype(np.float64)).astype(np.float32)
    expected = np.clip((radius-distance)/radius, 0, 1)
    np.testing.assert_array_equal(image_loss_mask(mask, 'expanded_feather')[0, 0], expected)
    assert np.count_nonzero(mask) == 1


def test_feather_is_independent_of_layout_allocations_and_ipp_setting():
    size = 256
    yy, xx = np.mgrid[:size, :size]
    original = ((((xx-128)/73)**2 + ((yy-128)/91)**2) < 1).astype(np.float32)[None, None]
    original[0, 0, 10, 10] = .25
    ipp = cv2.ipp.useIPP()
    try:
        reference = image_loss_mask(original, 'expanded_feather')
        for enabled in (True, False):
            cv2.ipp.setUseIPP(enabled)
            actual_setting = cv2.ipp.useIPP()
            for offset in range(16):
                storage = np.empty(original.size+offset, np.float32)
                shifted = storage[offset:].reshape(original.shape)
                shifted[:] = original
                for values in (shifted, shifted[..., ::-1].copy()[..., ::-1]):
                    np.testing.assert_array_equal(image_loss_mask(values, 'expanded_feather'), reference)
                    np.testing.assert_array_equal(values, original)
                    assert cv2.ipp.useIPP() == actual_setting
    finally:
        cv2.ipp.setUseIPP(ipp)


def test_feather_preserves_each_threads_ipp_setting():
    mask = np.zeros((1, 1, 256, 256), np.float32)
    mask[:, :, 100:140, 100:140] = 1
    reference = image_loss_mask(mask, 'expanded_feather')
    caller_setting = cv2.ipp.useIPP()
    def calculate(enabled):
        previous = cv2.ipp.useIPP()
        try:
            cv2.ipp.setUseIPP(enabled)
            expected = cv2.ipp.useIPP()
            for _ in range(5):
                np.testing.assert_array_equal(image_loss_mask(mask, 'expanded_feather'), reference)
                assert cv2.ipp.useIPP() == expected
        finally:
            cv2.ipp.setUseIPP(previous)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(calculate, [True, False, True, False]))
    assert cv2.ipp.useIPP() == caller_setting


@pytest.mark.parametrize('enabled', [True, False])
def test_feather_restores_ipp_after_distance_transform_failure(enabled):
    mask = np.ones((1, 1, 256, 256), np.float32)
    previous = cv2.ipp.useIPP()
    try:
        cv2.ipp.setUseIPP(enabled)
        expected = cv2.ipp.useIPP()
        with mock.patch('tae.masking.cv2.distanceTransform', side_effect=RuntimeError('synthetic distance failure')):
            with pytest.raises(RuntimeError, match='synthetic distance failure'):
                image_loss_mask(mask, 'expanded_feather')
        assert cv2.ipp.useIPP() == expected
    finally:
        cv2.ipp.setUseIPP(previous)


@pytest.mark.skipif(not torch.cuda.is_available() or not DEFAULT_WEIGHTS.is_file(),
                    reason='requires single CUDA GPU and pinned TAEQI2.1 asset')
@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_real_codec_repeated_updates_keep_weights_metrics_and_optimizer_equal(precision):
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 requires a supported CUDA GPU')
    config = TrainConfig(resolution=256, face_type='f', inter_dims=8, hidden_dims=8,
        batch_size=1, precision=precision, decoder_mode='frozen', random_warp=False,
        random_flip=False, training_mask_mode='expanded_feather', learning_rate=1e-3,
        warmup_steps=3)
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[:256, :256]
    mask = ((((xx-128)/73)**2 + ((yy-128)/91)**2) < 1).astype(np.float32)[None, None]
    batches = []
    for _ in range(2):
        image = rng.random((1, 3, 256, 256), dtype=np.float32)
        batches.append((image.copy(), image.copy(), mask.copy(), (mask*1.5).copy()))
    initial_inputs = [[value.copy() for value in batch] for batch in batches]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    reference = None
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for _ in range(2):
            worker = TAEEngine(config, 'cuda:0', seed=17)
            frozen_before = frozen_digest(worker.network.vae)
            rows = []
            for _ in range(3):
                result = worker.train_step(*batches)
                rows.append({name: value for name, value in result.items() if not name.startswith('cuda_')})
            weights = {name: value.detach().cpu().clone() for name, value in worker.network.named_parameters() if value.requires_grad}
            adam = {name: {key: value.detach().cpu().clone() for key, value in worker.optimizer.state[parameter].items()}
                    for name, parameter in worker.network.named_parameters() if parameter.requires_grad}
            assert frozen_digest(worker.network.vae) == frozen_before
            if reference is None:
                reference = weights, adam, rows
            else:
                assert rows == reference[2]
                for name, value in weights.items():
                    torch.testing.assert_close(value, reference[0][name], rtol=0, atol=0)
                for name, state in adam.items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, reference[1][name][key], rtol=0, atol=0)
            for original, batch in zip(initial_inputs, batches):
                for expected, value in zip(original, batch):
                    np.testing.assert_array_equal(value, expected)
            del worker
            torch.cuda.empty_cache()
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = benchmark
