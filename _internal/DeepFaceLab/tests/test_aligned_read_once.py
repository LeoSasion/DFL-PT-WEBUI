"""One payload/one JPEG decode, using only synthetic loose and packed faces."""
import builtins
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from DFLIMG import DFLJPG
from facelib import FaceType
from me_backend.data import _load_inventory, decode_aligned, read_aligned
from me_backend.config import MEConfig
from samplelib import Sample
from tae.config import TrainConfig
from tae.data import TAEAlignedDataset, _pretraining_holdout
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


torch.set_num_threads(2)
cv2.setNumThreads(2)


def watch_sample_io(monkeypatch, paths, payloads):
    loose_reads, packed_reads, jpeg_decodes = [], [], []
    targets = {str(path) for path in paths}
    encoded = set(payloads)

    def track_open(original):
        def opened(path, *args, **kwargs):
            if str(path) in targets:
                loose_reads.append(str(path))
            return original(path, *args, **kwargs)
        return opened

    original_read = Sample.read_raw_file
    original_decode = cv2.imdecode

    def read(sample, *args, **kwargs):
        packed_reads.append(sample.filename)
        return original_read(sample, *args, **kwargs)

    def decode(buffer, flags):
        if np.asarray(buffer).tobytes() in encoded:
            jpeg_decodes.append(flags)
        return original_decode(buffer, flags)

    monkeypatch.setattr(builtins, 'open', track_open(builtins.open))
    monkeypatch.setattr(io, 'open', track_open(io.open))
    monkeypatch.setattr(Sample, 'read_raw_file', read)
    monkeypatch.setattr(cv2, 'imdecode', decode)
    return loose_reads, packed_reads, jpeg_decodes


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
def test_me_alignment_reads_and_decodes_each_jpeg_once(tmp_path, monkeypatch, extension):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    paths = sorted(folder.glob('*.jpg'))
    payloads = [path.read_bytes() for path in paths]
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                      d_mask_dims=16, batch_size=2)
    # Use the original DFLJPG file-backed decoder as the pixel baseline.
    expected = []
    for path in paths:
        dfl = DFLJPG.load(str(path))
        decoded = SimpleNamespace(name=str(path), image=dfl.get_img(),
            face_type=FaceType.fromString(dfl.get_face_type()), dfl=dfl, packed=None)
        expected.append(read_aligned(path, config, decoded=decoded))
    if extension:
        pack_fixture(folder, extension)
    _, references, _ = _load_inventory(folder)
    reads, packed_reads, decodes = watch_sample_io(monkeypatch, paths, payloads)
    for reference, baseline in zip(references, expected):
        for actual, wanted in zip(read_aligned(reference, config), baseline):
            np.testing.assert_array_equal(actual, wanted)
    assert len(packed_reads if extension else reads) == len(paths)
    assert len(decodes) == len(paths)
    assert set(decodes) == {cv2.IMREAD_COLOR if extension else cv2.IMREAD_UNCHANGED}


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
def test_dfm_calibration_hashes_and_aligns_one_payload_per_sample(tmp_path, monkeypatch, extension):
    from dfm_export.quantization import calibration_faces

    folder = make_aligned(tmp_path / 'aligned', count=8)
    paths = sorted(folder.glob('*.jpg'))
    payloads = [path.read_bytes() for path in paths]
    config = MEConfig(resolution=64)
    expected = [read_aligned(path, config)[0] for path in paths]
    if extension:
        pack_fixture(folder, extension)
    reads, packed_reads, decodes = watch_sample_io(monkeypatch, paths, payloads)
    calibrated = list(calibration_faces(folder, config, 8))
    assert len(calibrated) == len(paths)
    for (tensor, record), image, payload, path in zip(calibrated, expected, payloads, paths):
        np.testing.assert_array_equal(tensor.numpy(), image[None])
        assert record == {'name': path.name, 'sha256': hashlib.sha256(payload).hexdigest()}
    assert len(packed_reads if extension else reads) == len(paths)
    assert len(decodes) == len(paths)


def test_loose_inventory_preserves_fingerprint_with_one_metadata_stat(tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / 'aligned', count=3)
    nested = make_aligned(folder / 'person', count=2)
    paths = sorted([*folder.glob('*.jpg'), *nested.glob('*.jpg')])
    before = [(path.relative_to(folder).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
              for path in paths]
    expected = hashlib.sha256(repr(before).encode()).hexdigest()
    original_stat, stats = Path.stat, []

    def track_stat(path, *args, **kwargs):
        if path in paths:
            stats.append(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', track_stat)
    root, references, fingerprint = _load_inventory(folder)
    assert root == folder
    assert [item.path for item in references] == list(map(str, paths))
    assert fingerprint == expected
    # One is_file() check and one fingerprint stat per loose file. Previously
    # size and mtime each performed a separate fingerprint stat.
    assert len(stats) == 2 * len(paths)


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
def test_tae_startup_reuses_payload_for_hash_metadata_and_mask_validation(tmp_path, monkeypatch, extension):
    folder = make_aligned(tmp_path / 'aligned', count=3, size=256)
    paths = sorted(folder.glob('*.jpg'))
    for index, path in enumerate(paths):
        dfl = DFLJPG.load(str(path))
        dfl.get_dict()['source_id'] = f'explicit-group-{index % 2}'
        dfl.save()
    payloads = [path.read_bytes() for path in paths]
    hashes = [hashlib.sha256(payload).hexdigest() for payload in payloads]
    config = TrainConfig(resolution=256, face_type='f', inter_dims=8, hidden_dims=8)
    records = [(path.name, sha, f'explicit-group-{index % 2}')
               for index, (path, sha) in enumerate(zip(paths, hashes))]
    expected_fingerprint = hashlib.sha256(json.dumps(
        dict(records=records, policy=config.resume_contract(), version=1), sort_keys=True).encode()).hexdigest()
    if extension:
        pack_fixture(folder, extension)
    reads, packed_reads, decodes = watch_sample_io(monkeypatch, paths, payloads)
    with TAEAlignedDataset(folder, config, True, 7) as dataset:
        assert dataset.sample_hashes == hashes
        assert dataset.fingerprint == expected_fingerprint
        assert sorted(map(len, dataset.groups)) == [1, 2]
    assert len(packed_reads if extension else reads) == len(paths)
    assert len(decodes) == len(paths)


def test_decoded_snapshot_has_no_long_lived_image_cache(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=1, size=256)
    path = folder / '000.jpg'
    config = TrainConfig(resolution=256, face_type='f')
    decoded = decode_aligned(path)
    first = read_aligned(path, config, decoded=decoded)
    make_aligned(folder, count=1, size=256, offset=79)
    second = read_aligned(path, config)
    assert not np.array_equal(first[0], second[0])
    assert hashlib.sha256(decoded.payload).digest() != hashlib.sha256(path.read_bytes()).digest()
    # The explicitly supplied snapshot stays internally consistent with its bytes.
    for actual, expected in zip(read_aligned(path, config, decoded=decoded), first):
        np.testing.assert_array_equal(actual, expected)


def test_loose_grayscale_validation_does_not_silently_expand_channels(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=1)
    path = folder / '000.jpg'
    metadata = DFLJPG.load(str(path)).get_dict()
    assert cv2.imwrite(str(path), np.full((128, 128), 127, np.uint8))
    dfl = DFLJPG.load(str(path))
    dfl.set_dict(metadata)
    dfl.save()
    with pytest.raises(ValueError, match='invalid BGR image'):
        read_aligned(path, MEConfig(resolution=64))


def oversized_jpeg_payload(path):
    """Keep genuine DFL metadata but exceed OpenCV's JPEG pixel-count limit."""
    dfl = DFLJPG.load(str(path))
    for chunk in dfl.chunks:
        if chunk['name'] == 'SOF0':
            # 32769 squared exceeds the default 2**30 pixel limit; OpenCV
            # rejects the dimensions before allocating a decoded image.
            chunk['data'] = chunk['data'][:1] + b'\x80\x01\x80\x01' + chunk['data'][5:]
            return dfl.dump()
    raise AssertionError('Synthetic JPEG has no baseline SOF header')


def test_malformed_loose_jpeg_is_excluded_from_pretraining(tmp_path):
    folder = make_aligned(tmp_path / 'pool', count=3, size=256)
    paths = sorted(folder.glob('*.jpg'))
    held = next(f'group-{n}' for n in range(1000)
                if _pretraining_holdout('source-id:' + f'group-{n}'))
    trained = next(f'group-{n}' for n in range(1000)
                   if not _pretraining_holdout('source-id:' + f'group-{n}'))
    for index, path in enumerate(paths):
        dfl = DFLJPG.load(str(path))
        dfl.get_dict()['source_id'] = held if index == 0 else trained
        dfl.save()
    broken = paths[-1]
    payload = oversized_jpeg_payload(broken)
    broken.write_bytes(payload)
    assert DFLJPG.load(str(broken)).has_data()
    with pytest.raises(cv2.error, match='validateInputImageSize'):
        cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decode_aligned(broken).image is None
    with pytest.raises(ValueError, match='invalid BGR image'):
        read_aligned(broken, MEConfig(resolution=64))
    config = TrainConfig(resolution=256, face_type='f', inter_dims=8, hidden_dims=8)
    with pytest.raises(ValueError, match='expected a square aligned face'):
        TAEAlignedDataset(folder, config, True, 7)
    with TAEAlignedDataset(folder, config, True, 7, pretraining=True) as dataset:
        assert dataset.pretraining_receipt['trainingFaces'] == 1
        assert dataset.pretraining_receipt['validationFaces'] == 1
        assert dataset.pretraining_receipt['excludedFaces'] == 1
        excluded = [record for record in dataset.pretraining_manifest if record['split'] == 'excluded']
        assert len(excluded) == 1 and excluded[0]['file'] == broken.name
        assert 'expected a square aligned face' in excluded[0]['reason']


@pytest.mark.parametrize('extension', ['pak', 'zip'])
def test_packed_decoder_errors_retain_the_existing_exception(tmp_path, monkeypatch, extension):
    folder = make_aligned(tmp_path / 'aligned', count=1, size=256)
    payload = oversized_jpeg_payload(folder / '000.jpg')
    pack_fixture(folder, extension)
    _, references, _ = _load_inventory(folder)
    # Keep the archive's valid metadata/inventory while delivering a damaged
    # member payload, preserving the packed decoder's preexisting exception.
    monkeypatch.setattr(Sample, 'read_raw_file', lambda sample: payload)
    with pytest.raises(cv2.error, match='validateInputImageSize'):
        decode_aligned(references[0])
