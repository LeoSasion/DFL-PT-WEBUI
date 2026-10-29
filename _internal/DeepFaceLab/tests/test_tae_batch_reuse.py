"""Bounded per-batch aligned/reference reuse, using synthetic faces only."""
from collections import Counter

import numpy as np
import pytest

from tae import data
from tae.config import TrainConfig
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


def settings(**options):
    return TrainConfig(resolution=256, face_type='f', inter_dims=8, hidden_dims=8,
        batch_size=4, random_warp=True, random_flip=True,
        half_resolution_probability=.5, **options).validate()


def assert_pair_equal(actual, expected):
    for left_side, right_side in zip(actual, expected):
        for left, right in zip(left_side, right_side):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
@pytest.mark.parametrize('mode', ['none', 'rct', 'lct'])
def test_repeated_selection_reuses_only_this_batch_and_preserves_resume(
        extension, mode, tmp_path, monkeypatch):
    src = make_aligned(tmp_path / 'src', count=1, size=256)
    dst = make_aligned(tmp_path / 'dst', count=1, size=256, offset=49)
    if extension:
        pack_fixture(src, extension)
        pack_fixture(dst, extension)
    config = settings(ct_mode=mode)
    with data.build_pair(src, dst, config, 17) as pair:
        original, calls = data.read_aligned, []

        def read(reference, config, **kwargs):
            calls.append((reference.path, kwargs.get('include_masks', True)))
            return original(reference, config, **kwargs)

        monkeypatch.setattr(data, 'read_aligned', read)
        global_rng = np.random.get_state()
        first = pair.batch()
        # Each side's consecutive duplicate samples require one full read.
        expected = Counter({(str(src / '000.jpg'), True): 1,
                            (str(dst / '000.jpg'), True): 1})
        if mode != 'none':
            expected[(str(dst / '000.jpg'), False)] = 1
        assert Counter(calls) == expected
        first_state = pair.state_dict()
        assert first_state['src']['exposures'] == [4]
        assert first_state['dst']['exposures'] == [4]
        second = pair.batch()
        assert Counter(calls) == Counter({key: value * 2 for key, value in expected.items()})
        second_state = pair.state_dict()
        after_rng = np.random.get_state()
        assert after_rng[0] == global_rng[0]
        np.testing.assert_array_equal(after_rng[1], global_rng[1])
        assert after_rng[2:] == global_rng[2:]
        # Stacking still owns independent sample/target/mask storage.
        first_target = first[0][1].copy()
        other_row = first[0][0][1].copy()
        first[0][0][0].fill(.123)
        np.testing.assert_array_equal(first[0][1], first_target)
        np.testing.assert_array_equal(first[0][0][1], other_row)
    with data.build_pair(src, dst, config, 99) as resumed:
        resumed.load_state_dict(first_state)
        assert_pair_equal(resumed.batch(), second)
        assert resumed.state_dict() == second_state


def test_single_sample_reuse_never_hides_changes_between_batches(tmp_path):
    folder = make_aligned(tmp_path / 'faces', count=1, size=256)
    config = settings(ct_mode='none')
    with data.TAEAlignedDataset(folder, config, False, 19) as dataset:
        initial = dataset.state_dict()
        first = dataset.batch()
        make_aligned(folder, count=1, size=256, offset=79)
        dataset.load_state_dict(initial)
        second = dataset.batch()
        assert not np.array_equal(first[1], second[1])


def test_only_most_recent_sample_is_reused_without_dataset_sized_cache(tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / 'faces', count=3, size=256)
    config = settings(ct_mode='none')
    config.batch_size = 16
    with data.TAEAlignedDataset(folder, config, False, 19) as dataset:
        original, reads = data.read_aligned, []

        def read(reference, config, **kwargs):
            reads.append(reference.path)
            return original(reference, config, **kwargs)

        monkeypatch.setattr(data, 'read_aligned', read)
        dataset.batch()
        # Returning to an older file must read it again; the bound is one.
        assert len(reads) > len(set(reads))
        assert all(left != right for left, right in zip(reads, reads[1:]))


def test_color_reference_read_failure_rolls_back_pair_and_checkpoint(tmp_path, monkeypatch):
    src = make_aligned(tmp_path / 'src', count=1, size=256)
    dst = make_aligned(tmp_path / 'dst', count=1, size=256, offset=49)
    with data.build_pair(src, dst, settings(ct_mode='lct'), 17) as pair:
        before = pair.state_dict()
        checkpoint = pair.checkpoint_snapshot()
        original = data.read_aligned

        def read(reference, config, **kwargs):
            if not kwargs.get('include_masks', True):
                raise ValueError('synthetic color-reference read failure')
            return original(reference, config, **kwargs)

        monkeypatch.setattr(data, 'read_aligned', read)
        with pytest.raises(ValueError, match='synthetic color-reference read failure'):
            pair.batch()
        assert pair.state_dict() == before
        assert checkpoint.materialize() == before
        assert pair.source_data.closed and pair.destination_data.closed
