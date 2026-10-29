"""Unused warp-map work and archive indexes, using synthetic faces only."""
import copy
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from core.imagelib.warp import gen_warp_params, warp_by_params
from me_backend import data as me_data
from me_backend.config import MEConfig
from samplelib.PackedFaceset import PackedFaceset
from tae import data as tae_data
from tae.config import TrainConfig
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


def assert_pair_equal(actual, expected):
    for actual_side, expected_side in zip(actual, expected):
        for actual_array, expected_array in zip(actual_side, expected_side):
            np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.parametrize('size', [32, 64, 256])
def test_warp_default_retains_explicit_generation_with_a_shared_rng(size):
    # Legacy callers can deliberately share geometry/warp RNGs; the new opt-in
    # must not change their default map values or subsequent random draws.
    implicit_rng = np.random.RandomState(17)
    explicit_rng = np.random.RandomState(17)
    implicit = gen_warp_params(size, True, rnd_state=implicit_rng, warp_rnd_state=implicit_rng)
    explicit = gen_warp_params(size, True, rnd_state=explicit_rng, warp_rnd_state=explicit_rng,
                               generate_warp=True)
    for name in implicit:
        np.testing.assert_array_equal(implicit[name], explicit[name])
    np.testing.assert_array_equal(implicit_rng.random(20), explicit_rng.random(20))


@pytest.mark.parametrize('size', [32, 256, 512])
def test_unused_maps_preserve_affine_flip_and_skip_dense_resize(size, monkeypatch):
    calls = []
    original = cv2.resize

    def resize(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, 'resize', resize)
    full_rng, lean_rng = np.random.RandomState(5), np.random.RandomState(5)
    full = gen_warp_params(size, True, rnd_state=full_rng, warp_rnd_state=np.random.RandomState(11))
    assert len(calls) == 2
    calls.clear()
    lean = gen_warp_params(size, True, rnd_state=lean_rng, warp_rnd_state=np.random.RandomState(11),
                          generate_warp=False)
    assert calls == []
    assert lean['mapx'] is None and lean['mapy'] is None
    for name in ('rmat', 'umat', 'w', 'rw', 'flip'):
        np.testing.assert_array_equal(lean[name], full[name])
    image = np.random.default_rng(19).random((size, size, 3), dtype=np.float32)
    np.testing.assert_array_equal(warp_by_params(lean, image, False, True, True, True),
                                  warp_by_params(full, image, False, True, True, True))
    np.testing.assert_array_equal(full_rng.random(20), lean_rng.random(20))


@pytest.mark.parametrize('family', ['ME', 'TAE'])
@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
@pytest.mark.parametrize('random_warp', [False, True])
def test_batches_and_resume_match_eager_maps(family, extension, random_warp, tmp_path, monkeypatch):
    size = 256 if family == 'TAE' else 128
    src = make_aligned(tmp_path / 'src', count=2, size=size)
    dst = make_aligned(tmp_path / 'dst', count=2, offset=47, size=size)
    if extension:
        pack_fixture(src, extension)
        pack_fixture(dst, extension)
    if family == 'TAE':
        module = tae_data
        config = TrainConfig(resolution=256, face_type='f', inter_dims=8, hidden_dims=8,
            batch_size=2, random_warp=random_warp, random_flip=True, ct_mode='lct',
            half_resolution_probability=.5).validate()

        def pair(seed):
            return tae_data.build_pair(src, dst, config, seed)
    else:
        module = me_data
        config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16,
            batch_size=2, random_warp=random_warp, random_src_flip=True,
            random_dst_flip=True, ct_mode='rct', random_hsv_power=.05)

        def pair(seed):
            return me_data.PairDataLoader(me_data.AlignedDataset(src, config, True, seed),
                                         me_data.AlignedDataset(dst, config, False, seed + 1))

    original = module.gen_warp_params

    def eager(*args, **kwargs):
        kwargs['generate_warp'] = True
        return original(*args, **kwargs)

    with pair(23) as baseline, pair(23) as optimized:
        for step in range(3):
            with monkeypatch.context() as patch:
                patch.setattr(module, 'gen_warp_params', eager)
                expected = baseline.batch()
            actual = optimized.batch()
            assert_pair_equal(actual, expected)
            assert optimized.state_dict() == baseline.state_dict()
            assert optimized.checkpoint_snapshot().materialize() == baseline.checkpoint_snapshot().materialize()
            if step == 1:
                saved = copy.deepcopy(optimized.state_dict())
        expected_next = actual
    with pair(999) as resumed:
        resumed.load_state_dict(saved)
        assert_pair_equal(resumed.batch(), expected_next)


def test_inventory_does_not_hide_stat_errors_on_non_jpeg_entries(tmp_path, monkeypatch):
    root = make_aligned(tmp_path / 'aligned', count=1)
    denied = root / 'notes.txt'
    denied.write_text('synthetic inaccessible entry', encoding='utf8')
    original = Path.is_file

    def is_file(path):
        if path == denied:
            raise PermissionError('synthetic inventory stat failure')
        return original(path)

    monkeypatch.setattr(Path, 'is_file', is_file)
    with pytest.raises(PermissionError, match='synthetic inventory stat failure'):
        me_data._load_inventory(root)


def test_zip_load_indexes_members_once_and_still_rejects_missing_images(tmp_path, monkeypatch):
    root = make_aligned(tmp_path / 'aligned', count=8)
    expected = {path.name: path.read_bytes() for path in root.glob('*.jpg')}
    archive_path = pack_fixture(root, 'zip')
    calls = []
    original = zipfile.ZipFile.namelist

    def namelist(archive):
        calls.append(archive.filename)
        return original(archive)

    monkeypatch.setattr(zipfile.ZipFile, 'namelist', namelist)
    samples = PackedFaceset.load(root)
    assert len(calls) == 1
    assert {sample.filename: sample.read_raw_file() for sample in samples} == expected
    with zipfile.ZipFile(archive_path) as archive:
        entries = [(info.filename, archive.read(info)) for info in archive.infolist()
                   if info.filename != '000.jpg']
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match='Duplicate or missing ZIP faceset image'):
        PackedFaceset.load(root)
