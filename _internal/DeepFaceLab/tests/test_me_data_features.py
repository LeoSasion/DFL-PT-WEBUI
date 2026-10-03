"""Small CPU fixtures for ME faceset loading, augmentation and worker lifetime."""
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import struct
import zipfile

import cv2
import numpy as np
import pytest

from DFLIMG import DFLJPG
from facelib import FaceType
from me_backend.config import MEConfig
from me_backend.data import AlignedDataset, PairDataLoader
from me_backend.data_augmentation import CT_MODES, transfer_color
from samplelib import Sample
from samplelib.PackedFaceset import PackedFaceset
from tests.me_fixtures import make_aligned


def small(**changes):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                    d_mask_dims=16, batch_size=2, **changes)


def assert_batch_equal(actual, expected):
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def assert_pair_equal(actual, expected):
    for left, right in zip(actual, expected):
        assert_batch_equal(left, right)


def pack_fixture(directory, extension, delete_images=True):
    """Write the existing DFL v1 format, including person-prefixed members."""
    directory = Path(directory)
    paths = sorted(directory.rglob('*.jpg'))
    samples, payloads = [], []
    for path in paths:
        dfl = DFLJPG.load(str(path))
        sample = Sample(filename=path.name, face_type=FaceType.FULL,
                        shape=dfl.get_shape(), landmarks=dfl.get_landmarks(),
                        seg_ie_polys=dfl.get_seg_ie_polys(),
                        xseg_mask_compressed=dfl.get_xseg_mask_compressed(),
                        eyebrows_expand_mod=dfl.get_eyebrows_expand_mod(),
                        source_filename=dfl.get_source_filename(),
                        person_name=None if path.parent == directory else path.parent.name)
        config = sample.get_config()
        if 'pitch_yaw_roll' in dfl.get_dict():
            config['pitch_yaw_roll'] = dfl.get_dict()['pitch_yaw_roll']
        samples.append(config)
        payloads.append(path.read_bytes())
    metadata = pickle.dumps(samples, protocol=4)
    archive = directory / f'faceset.{extension}'
    if extension == 'pak':
        offsets = [0]
        for payload in payloads:
            offsets.append(offsets[-1] + len(payload))
        archive.write_bytes(struct.pack('<QQ', 1, len(metadata)) + metadata
                            + struct.pack(f'<{len(offsets)}Q', *offsets) + b''.join(payloads))
    else:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
            target.writestr('config.pak', metadata)
            for path, payload in zip(paths, payloads):
                target.writestr(path.relative_to(directory).as_posix(), payload)
    if delete_images:
        for path in paths:
            path.unlink()
    return archive


@pytest.mark.parametrize('extension', ['pak', 'zip'])
def test_packed_facesets_train_directly_with_person_directories(extension, tmp_path):
    folder = tmp_path / 'aligned'
    make_aligned(folder / 'alice', count=2)
    make_aligned(folder / 'bob', count=2, offset=37)
    config = small()
    with AlignedDataset(folder, config, True, 17) as loose:
        expected = loose.batch()
        expected_paths = [path.relative_to(folder).as_posix() for path in loose.paths]
    archive = pack_fixture(folder, extension)
    assert not list(folder.rglob('*.jpg'))
    with AlignedDataset(folder, config, True, 17) as packed:
        assert [path.relative_to(folder).as_posix() for path in packed.paths] == expected_paths
        assert_batch_equal(packed.batch(), expected)
        state = packed.state_dict()
    with AlignedDataset(archive, config, True, 99) as direct:
        direct.load_state_dict(state)
        batch = direct.batch()
        assert batch[0].shape == (2, 3, 64, 64)
        assert batch[2].shape == (2, 1, 64, 64)
        assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in batch)
        assert batch[3].max() > 2


@pytest.mark.parametrize('extension', ['pak', 'zip'])
def test_packed_dataset_identity_is_checked_on_resume(extension, tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    archive = pack_fixture(folder, extension)
    with AlignedDataset(folder, small(), True, 4) as dataset:
        state = dataset.state_dict()
    info = archive.stat()
    os.utime(archive, ns=(info.st_atime_ns, info.st_mtime_ns + 100_000_000))
    with AlignedDataset(folder, small(), True, 4) as changed:
        with pytest.raises(ValueError, match='dataset changed'):
            changed.load_state_dict(state)


def test_explicit_archive_selection_preserves_default_pak_priority(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    pak_path = pack_fixture(folder, 'pak', delete_images=False)
    config = small()
    with AlignedDataset(folder, config, True, 17) as packed:
        pak_expected = packed.batch()
        pak_fingerprint = packed.fingerprint
    # ZIP contains different pixels while sharing the same member names.
    make_aligned(folder, count=2, offset=67)
    zip_path = pack_fixture(folder, 'zip')
    with AlignedDataset(folder, config, True, 17) as default:
        assert default.fingerprint == pak_fingerprint
        assert_batch_equal(default.batch(), pak_expected)
    with AlignedDataset(pak_path, config, True, 17) as explicit_pak:
        assert_batch_equal(explicit_pak.batch(), pak_expected)
    with AlignedDataset(zip_path, config, True, 17) as explicit_zip:
        assert explicit_zip.fingerprint != pak_fingerprint
        assert not np.array_equal(explicit_zip.batch()[0], pak_expected[0])
        assert Path(explicit_zip._references[0].packed_sample._filename_offset_size[0]) == zip_path


def test_explicit_archive_path_validation(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    archive = pack_fixture(folder, 'zip')
    with pytest.raises(ValueError, match='inside the faceset directory'):
        PackedFaceset.load(tmp_path, archive_path=archive)
    with pytest.raises(ValueError, match='PAK/ZIP'):
        PackedFaceset.load(folder, archive_path=folder / 'unsupported.bin')
    with pytest.raises(FileNotFoundError, match='not found'):
        PackedFaceset.load(folder, archive_path=folder / 'missing.pak')


def test_flat_facesets_keep_existing_checkpoint_fingerprints(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    inventory = [(path.name, path.stat().st_size, path.stat().st_mtime_ns)
                 for path in sorted(folder.iterdir())]
    with AlignedDataset(folder, small(), True, 1) as dataset:
        assert dataset.fingerprint == hashlib.sha256(repr(inventory).encode()).hexdigest()


def test_bad_archive_never_falls_back_to_loose_images(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    (folder / 'faceset.pak').write_bytes(b'broken faceset')
    with pytest.raises(ValueError, match='Truncated faceset header'):
        AlignedDataset(folder, small(), True, 0)


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
def test_uniform_yaw_balances_rare_angles(extension, tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=20)
    for index, path in enumerate(sorted(folder.glob('*.jpg'))):
        dfl = DFLJPG.load(str(path))
        dfl.get_dict()['pitch_yaw_roll'] = (0., .9 if index == 19 else 0., 0.)
        dfl.save()
    if extension:
        pack_fixture(folder, extension)
    with AlignedDataset(folder, small(uniform_yaw=True), True, 9) as balanced:
        counts = [Path(arguments[0].path).name for _ in range(1000)
                  for arguments in balanced._plan_batch(balanced.rng)]
        assert len(balanced.yaw_groups) == 2
        assert .45 < counts.count('019.jpg') / len(counts) < .55
    with AlignedDataset(folder, small(), True, 9) as ordinary:
        counts = [Path(arguments[0].path).name for _ in range(1000)
                  for arguments in ordinary._plan_batch(ordinary.rng)]
        assert .03 < counts.count('019.jpg') / len(counts) < .08


@pytest.mark.parametrize('flag', ['random_downsample', 'random_noise', 'random_blur', 'random_jpeg'])
@pytest.mark.parametrize('source', [True, False])
def test_distortions_change_inputs_but_preserve_targets_and_masks(flag, source, tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    with AlignedDataset(folder, small(), source, 21) as clean:
        expected = clean.batch()
    with AlignedDataset(folder, small(**{flag: True}), source, 21) as distorted:
        actual = distorted.batch()
    assert not np.array_equal(actual[0], expected[0])
    assert_batch_equal(actual[1:], expected[1:])
    assert np.isfinite(actual[0]).all() and actual[0].min() >= 0 and actual[0].max() <= 1


@pytest.mark.parametrize('mode', CT_MODES[1:])
def test_color_modes_are_finite_reproducible_and_preserve_global_rng(mode):
    y, x = np.mgrid[:64, :64]
    image = np.stack((.1 + .7 * x / 64, .15 + .65 * y / 64, .3 + .1 * np.sin(x / 7)), -1).astype(np.float32)
    reference = image[..., ::-1].copy() * .7 + .2
    numpy_state, python_state = copy.deepcopy(np.random.get_state()), random.getstate()
    first = transfer_color(image.copy(), mode, reference, 137)
    second = transfer_color(image.copy(), mode, reference, 137)
    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape and first.dtype == np.float32 and np.isfinite(first).all()
    assert first.min() >= 0 and first.max() <= 1
    assert not np.array_equal(first, image)
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert random.getstate() == python_state


def test_source_color_transfer_requires_pair_and_checks_reference_identity(tmp_path):
    source_path = make_aligned(tmp_path / 'src', count=2)
    destination_path = make_aligned(tmp_path / 'dst', count=2, offset=61)
    config = small(ct_mode='rct')
    with AlignedDataset(source_path, config, True, 2) as source:
        with pytest.raises(ValueError, match='DST reference'):
            source.batch()
    with PairDataLoader(AlignedDataset(source_path, config, True, 2),
                        AlignedDataset(destination_path, config, False, 3)) as pair:
        first = pair.batch()
        state = json.loads(json.dumps(pair.state_dict()))
        expected = pair.batch()
        assert all(np.isfinite(value).all() for batch in first for value in batch)
    source = AlignedDataset(source_path, config, True, 99)
    destination = AlignedDataset(destination_path, config, False, 99)
    source.load_state_dict(state['src'])
    destination.load_state_dict(state['dst'])
    with PairDataLoader(source, destination) as resumed:
        assert_pair_equal(resumed.batch(), expected)
    changed = destination_path / '000.jpg'
    info = changed.stat()
    os.utime(changed, ns=(info.st_atime_ns, info.st_mtime_ns + 100_000_000))
    with AlignedDataset(source_path, config, True, 99) as source:
        source.load_state_dict(state['src'])
        with AlignedDataset(destination_path, config, False, 99) as destination:
            with pytest.raises(ValueError, match='DST color reference dataset changed'):
                PairDataLoader(source, destination)


def test_random_lab_color_changes_rgb_but_not_masks(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    with AlignedDataset(folder, small(), True, 21) as clean:
        expected = clean.batch()
    with AlignedDataset(folder, small(random_color=True), True, 21) as randomized:
        actual = randomized.batch()
    assert not np.array_equal(actual[0], expected[0])
    assert not np.array_equal(actual[1], expected[1])
    assert_batch_equal(actual[2:], expected[2:])


@pytest.mark.parametrize('mode', ['rct', 'mkl', 'idt'])
def test_color_transfer_handles_collapsed_distributions(mode):
    image = np.full((64, 64, 3), .2, np.float32)
    reference = np.full_like(image, .8)
    result = transfer_color(image, mode, reference, 137)
    assert np.isfinite(result).all()
    # IDT relaxes toward the target distribution over its twenty iterations.
    assert .5 < result.mean() < .85
    assert result.max() - result.min() < .001


def test_failed_pair_does_not_commit_src_state_and_closes_both_pools(tmp_path):
    source_path = make_aligned(tmp_path / 'src', count=2)
    destination_path = tmp_path / 'bad-dst'
    destination_path.mkdir()
    (destination_path / '000.jpg').write_bytes(b'bad aligned JPEG')
    pair = PairDataLoader(AlignedDataset(source_path, small(data_workers=1), True, 8),
                          AlignedDataset(destination_path, small(data_workers=1), False, 9))
    expected = pair.state_dict()
    before = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(ValueError, match='missing DFL aligned metadata'):
        pair.batch()
    assert pair.state_dict() == expected
    assert pair.source_data.closed and pair.destination_data.closed
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_prefetch_resume_matches_sync_and_closes_all_owned_workers(tmp_path):
    source_path = make_aligned(tmp_path / 'src', count=4)
    destination_path = make_aligned(tmp_path / 'dst', count=4, offset=49)
    options = dict(uniform_yaw=True, random_downsample=True, random_noise=True,
                   random_blur=True, random_jpeg=True, random_color=True,
                   ct_mode='rct', random_hsv_power=.15)
    sync = PairDataLoader(AlignedDataset(source_path, small(**options), True, 7),
                          AlignedDataset(destination_path, small(**options), False, 8))
    config = small(data_workers=2, **options)
    worker_pids = set()
    with sync, PairDataLoader(AlignedDataset(source_path, config, True, 7),
                             AlignedDataset(destination_path, config, False, 8)) as pair:
        assert_pair_equal(pair.batch(), sync.batch())
        worker_pids.update(pair.source_data.worker_pids + pair.destination_data.worker_pids)
        assert worker_pids
        assert len(pair.source_data._pending) == 2 and len(pair.destination_data._pending) == 2
        state = copy.deepcopy(pair.state_dict())
        expected = pair.batch()
        assert_pair_equal(expected, sync.batch())
        # Rewind while prefetched work exists; that work must not consume state.
        pair.load_state_dict(state)
        assert_pair_equal(pair.batch(), expected)
        worker_pids.update(pair.source_data.worker_pids + pair.destination_data.worker_pids)
    assert not worker_pids.intersection(process.pid for process in multiprocessing.active_children())
    with PairDataLoader(AlignedDataset(source_path, small(**options), True, 999),
                        AlignedDataset(destination_path, small(**options), False, 999)) as resumed:
        resumed.load_state_dict(state)
        assert_pair_equal(resumed.batch(), expected)
    pair.close()
    with pytest.raises(RuntimeError, match='closed'):
        pair.batch()


@pytest.mark.parametrize('extension', ['pak', 'zip'])
def test_packed_prefetch_matches_sync_and_restores_next_batch(extension, tmp_path):
    folder = make_aligned(tmp_path / 'person-data' / 'alice', count=3).parent
    pack_fixture(folder, extension)
    with AlignedDataset(folder, small(random_noise=True), True, 1) as sync:
        expected_first, expected_next = sync.batch(), sync.batch()
    with AlignedDataset(folder, small(random_noise=True, data_workers=1), True, 1) as parallel:
        assert_batch_equal(parallel.batch(), expected_first)
        state = parallel.state_dict()
        pids = set(parallel.worker_pids)
    assert not pids.intersection(process.pid for process in multiprocessing.active_children())
    with AlignedDataset(folder, small(random_noise=True, data_workers=1), True, 9) as resumed:
        resumed.load_state_dict(state)
        assert_batch_equal(resumed.batch(), expected_next)


def test_worker_read_failure_is_reported_and_not_silently_run_in_process(tmp_path):
    folder = tmp_path / 'corrupt'
    folder.mkdir()
    (folder / '000.jpg').write_bytes(b'not an aligned JPEG')
    before = {process.pid for process in multiprocessing.active_children()}
    with AlignedDataset(folder, small(data_workers=1), True, 0) as dataset:
        with pytest.raises(ValueError, match='missing DFL aligned metadata'):
            dataset.batch()
        assert dataset.closed
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_unknown_color_mode_is_never_downgraded():
    with pytest.raises(ValueError, match='Unsupported ME ct_mode'):
        transfer_color(np.zeros((64, 64, 3), np.float32), 'unsupported', None, 0)
