"""Same-image transform reuse preserves pixels, independent arrays and samplers."""
import copy

import numpy as np
import pytest

from me_backend import data
from me_backend.config import MEConfig
from me_backend.data_augmentation import REFERENCE_CT_MODES
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


def settings(**options):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                    d_mask_dims=16, batch_size=2, **{'random_warp': False,
                        'random_src_flip': True, 'random_dst_flip': True, **options})


def separate_input(original):
    # Force the old two-input path with equal but independently owned pixels.
    # This keeps the original augmentation implementation and RNG consumption.
    def separate(*args, **kwargs):
        return original(*args, **kwargs).copy()
    return separate


@pytest.mark.parametrize('source', [False, True])
@pytest.mark.parametrize('options', [
    {}, {'random_hsv_power': .05}, {'random_color': True},
    {'random_hsv_power': .05, 'random_color': True},
    {'random_blur': True}, {'random_noise': True}, {'random_jpeg': True},
    {'random_downsample': True}, {'ct_mode': 'rct'}, {'ct_mode': 'lct'},
    {'ct_mode': 'fs-aug'}, {'ct_mode': 'cc-aug'},
    {'random_warp': True, 'random_hsv_power': .05, 'random_color': True},
])
def test_same_input_path_preserves_all_pixels_without_aliasing(source, options, tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / 'faces', count=2)
    _, references, _ = data._load_inventory(folder)
    config = settings(**options)
    color_reference = references[1] if source and config.ct_mode in REFERENCE_CT_MODES else None
    arguments = (references[0], config, source, 19, 47, color_reference)
    original = data.distort_input
    global_rng = np.random.get_state()
    with monkeypatch.context() as patch:
        patch.setattr(data, 'distort_input', separate_input(original))
        expected = data._process_sample(*arguments)
    actual = data._process_sample(*arguments)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    final_rng = np.random.get_state()
    assert final_rng[0] == global_rng[0]
    np.testing.assert_array_equal(final_rng[1], global_rng[1])
    assert final_rng[2:] == global_rng[2:]
    for left in range(len(actual)):
        assert actual[left].flags.writeable
        for right in range(left):
            assert not np.shares_memory(actual[left], actual[right])
    target = actual[1].copy()
    actual[0].fill(.123)
    np.testing.assert_array_equal(actual[1], target)


@pytest.mark.parametrize('source,hsv,lab,warp,expected_geometry,expected_lab', [
    (True, 0., False, False, 3, 0),
    (False, 0., True, False, 3, 1),
    (True, .05, True, False, 3, 1),
    (False, .05, True, False, 4, 2),
    (True, 0., True, True, 4, 2),
])
def test_only_identical_rgb_paths_share_geometry_and_color(source, hsv, lab, warp,
        expected_geometry, expected_lab, tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / 'faces', count=1)
    _, references, _ = data._load_inventory(folder)
    config = MEConfig.from_dict({**settings().to_dict(), 'random_warp': warp,
        'random_hsv_power': hsv, 'random_color': lab})
    calls = {'geometry': 0, 'lab': 0}
    geometry, rotate = data.warp_by_params, data.rotate_lab_color

    def apply_geometry(*args, **kwargs):
        calls['geometry'] += 1
        return geometry(*args, **kwargs)

    def apply_lab(*args, **kwargs):
        calls['lab'] += 1
        return rotate(*args, **kwargs)

    monkeypatch.setattr(data, 'warp_by_params', apply_geometry)
    monkeypatch.setattr(data, 'rotate_lab_color', apply_lab)
    result = data._process_sample(references[0], config, source, 3, 5, None)
    assert calls == {'geometry': expected_geometry, 'lab': expected_lab}
    if not source and hsv:
        assert not np.array_equal(result[0], result[1]), 'DST HSV changes only its target'


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
@pytest.mark.parametrize('workers', [0, 1])
def test_pair_batches_prefetch_and_resume_equal_independent_rgb_paths(extension, workers,
        tmp_path, monkeypatch):
    src = make_aligned(tmp_path / 'src', count=3)
    dst = make_aligned(tmp_path / 'dst', count=3, offset=41)
    if extension:
        pack_fixture(src, extension)
        pack_fixture(dst, extension)
    config = settings(random_color=True, ct_mode='lct')
    actual_config = MEConfig.from_dict({**config.to_dict(), 'data_workers': workers})

    def pair(config, seed):
        return data.PairDataLoader(data.AlignedDataset(src, config, True, seed),
                                   data.AlignedDataset(dst, config, False, seed + 1))

    original = data.distort_input
    with pair(config, 31) as expected_loader, pair(actual_config, 31) as actual_loader:
        for _ in range(3):
            with monkeypatch.context() as patch:
                patch.setattr(data, 'distort_input', separate_input(original))
                expected = expected_loader.batch()
            actual = actual_loader.batch()
            for actual_side, expected_side in zip(actual, expected):
                for left, right in zip(actual_side, expected_side):
                    np.testing.assert_array_equal(left, right)
                    assert left.flags.c_contiguous and left.flags.writeable
                for index, left in enumerate(actual_side):
                    for right in actual_side[:index]:
                        assert not np.shares_memory(left, right)
            assert actual_loader.state_dict() == expected_loader.state_dict()
            assert actual_loader.checkpoint_snapshot().materialize() == expected_loader.checkpoint_snapshot().materialize()
        saved = copy.deepcopy(actual_loader.state_dict())
        with monkeypatch.context() as patch:
            patch.setattr(data, 'distort_input', separate_input(original))
            expected_next = expected_loader.batch()
    with pair(actual_config, 999) as resumed:
        resumed.load_state_dict(saved)
        for actual_side, expected_side in zip(resumed.batch(), expected_next):
            for left, right in zip(actual_side, expected_side):
                np.testing.assert_array_equal(left, right)
