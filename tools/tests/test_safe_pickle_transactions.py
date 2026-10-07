"""Data-only legacy pickle compatibility and real failure recovery checks."""
import hashlib
import json
from pathlib import Path
from pathlib import PureWindowsPath
import pickle
import shutil
import struct
import subprocess
import sys

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / '_internal' / 'DeepFaceLab'
sys.path.insert(0, str(BACKEND))

from core.safe_pickle import (DataFormatError, load_file, loads, validate_dfl_metadata,
                              validate_faceset_configs, validate_metadata_backup, DataRecord)
from core.safe_pickle import preserve_rejected_file
from core import faceset_transaction as tx
from DFLIMG import DFLJPG
from mainscripts import Sorter, Util
from samplelib import PackedFaceset
from samplelib.Sample import SampleType
from facelib import FaceType


def make_faces(directory, count=2):
    directory.mkdir()
    for index in range(count):
        path = directory / f'face-{index}.jpg'
        image = np.full((64, 64, 3), index * 23 + 100, np.uint8)
        assert cv2.imwrite(str(path), image)
        dfl = DFLJPG.load(path)
        points = np.arange(136, dtype=np.float32).reshape(68, 2) / 4
        dfl.set_dict({'face_type': 'whole_face', 'landmarks': points.tolist(),
                      'source_landmarks': points.tolist(), 'source_rect': [0, 0, 64, 64],
                      'source_filename': f'original-{index}.png',
                      'image_to_face_mat': [[1, 0, 0], [0, 1, 0]],
                      'native_landmarks98': {'points_aligned': np.arange(196).reshape(98, 2).tolist()},
                      'seg_ie_polys': {'polys': [{'type': 1, 'pts': [[8, 8], [56, 8], [56, 56]]}]}})
        dfl.set_xseg_mask(np.full((64, 64, 1), .5, np.float32))
        dfl.save()
        path.with_suffix('.jpg.landmarks.json').write_text(json.dumps({'aligned_sha256': tx.sha256(path)}))
    return directory


@pytest.mark.parametrize('protocol', range(6))
def test_numpy_historical_protocols_are_data_only_and_exact(protocol):
    data = {'landmarks': np.arange(136, dtype=np.float32).reshape(68, 2),
            'matrix': np.arange(6, dtype='>f8').reshape(2, 3),
            'fortran': np.asfortranarray(np.arange(12).reshape(3, 4)),
            'scalar': np.float32(.25), 'sample_type': SampleType.FACE,
            'face_type': FaceType.WHOLE_FACE}
    restored = loads(pickle.dumps(data, protocol=protocol))
    for key in ('landmarks', 'matrix', 'fortran'):
        np.testing.assert_array_equal(restored[key], data[key])
    assert restored['scalar'] == .25 and restored['sample_type'] == 1 and restored['face_type'] == 4


def test_globals_extensions_persistent_refs_and_extra_data_are_rejected_without_execution(tmp_path):
    marker = tmp_path / 'must-not-exist'
    class Forbidden:
        def __reduce__(self):
            return eval, (f"open({str(marker)!r}, 'w').write('bad')",)
    for payload in (pickle.dumps(Forbidden()), b'\x80\x04\x82\x01.', b'\x80\x04Pbad\n.',
                    pickle.dumps({}) + b'extra', b'\x80\x04\x97.'):
        with pytest.raises(DataFormatError): loads(payload)
    assert not marker.exists()


def test_session_records_and_paths_are_inert_and_other_profiles_reject_them():
    from merger.FrameInfo import FrameInfo
    value = FrameInfo(filepath=PureWindowsPath('C:/private/project/frame.png'),
                      landmarks_list=[np.zeros((68, 2), np.float32)])
    payload = pickle.dumps(value, protocol=4)
    result = loads(payload, profile='session')
    assert isinstance(result, DataRecord) and result.type_name == 'merger.FrameInfo.FrameInfo'
    assert result.state['filepath'] == 'C:\\private\\project\\frame.png'
    np.testing.assert_array_equal(result.state['landmarks_list'][0], value.landmarks_list[0])
    for profile in ('metadata', 'faceset', 'weights'):
        with pytest.raises(DataFormatError, match='Forbidden'): loads(payload, profile=profile)
    # Even in a session, no general instance construction is supported.
    with pytest.raises(DataFormatError, match='Forbidden'):
        loads(pickle.dumps(Exception('unknown object')), profile='session')


def test_bigint_claim_is_rejected_before_numeric_decode():
    payload = b'\x80\x04\x8b' + struct.pack('<i', 2**30)
    with pytest.raises(DataFormatError, match='Integer magnitude'): loads(payload, profile='weights')


def test_invalid_optional_settings_are_preserved_and_archive_collision_never_overwrites(tmp_path):
    path = tmp_path / 'settings.dat'; path.write_bytes(b'not a valid pickle')
    archive = preserve_rejected_file(path)
    assert archive.read_bytes() == path.read_bytes()
    assert preserve_rejected_file(path) == archive
    archive.write_bytes(b'newer unrelated content')
    with pytest.raises(DataFormatError, match='Cannot preserve'): preserve_rejected_file(path)
    assert archive.read_bytes() == b'newer unrelated content'
    assert path.read_bytes() == b'not a valid pickle'


def test_numpy_recipe_bounds_are_checked_before_numpy_allocation(monkeypatch):
    class HugeArray:
        def __reduce__(self):
            return np._core.multiarray._reconstruct, (np.ndarray, (0,), b'b'), (
                1, (100000000, 100000000), np.dtype('f8'), False, b'')
    payload = pickle.dumps(HugeArray(), protocol=4)
    def must_not_allocate(*args, **kwargs): raise AssertionError('Array allocation was reached')
    monkeypatch.setattr(np, 'frombuffer', must_not_allocate)
    with pytest.raises(DataFormatError, match='element budget'): loads(payload)
    with pytest.raises(DataFormatError, match='dtype'):
        loads(pickle.dumps(np.array([None], dtype=object)))


def test_cycles_depth_amplification_and_truncation_are_bounded():
    cycle = []; cycle.append(cycle)
    deep = []
    for _ in range(60): deep = [deep]
    amplified = [1]
    for _ in range(18): amplified = [amplified, amplified]
    for value in (cycle, deep, amplified):
        with pytest.raises(DataFormatError): loads(pickle.dumps(value, protocol=4))
    with pytest.raises(DataFormatError): loads(pickle.dumps({'data': [1, 2]})[:-1])
    with pytest.raises(DataFormatError): loads(b'\x80\x04' + b'X' + struct.pack('<I', 0xffffffff))


def test_schema_rejects_unbounded_canvas_bad_landmarks_and_mask_header():
    with pytest.raises(DataFormatError): validate_dfl_metadata({'landmarks': [[float('nan'), 0]]})
    with pytest.raises(DataFormatError): validate_dfl_metadata({'image_to_face_mat': [[1, 2]]})
    png = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', 2**30, 2**30)
    with pytest.raises(DataFormatError, match='canvas'): validate_dfl_metadata({'xseg_mask': png})
    with pytest.raises(DataFormatError): validate_metadata_backup({'../outside.jpg': ((64, 64, 3), {})})
    with pytest.raises(DataFormatError): validate_faceset_configs([{'filename': 'a.jpg', 'shape': [2**30, 2**30, 3]}])


def test_hostile_app15_and_pak_config_never_execute(tmp_path):
    image = make_faces(tmp_path / 'aligned', 1) / 'face-0.jpg'
    dfl = DFLJPG.load(image)
    payload = b'cos\nsystem\n(S\'echo forbidden\'\ntR.'
    chunk = next(chunk for chunk in dfl.chunks if chunk['name'] == 'APP15')
    chunk['data'] = payload
    # Serialise chunks directly, without a writer that normalises metadata.
    raw = b''
    for item in dfl.chunks:
        raw += struct.pack('BB', 255, item['m_h'])
        if item['data'] is not None: raw += struct.pack('>H', len(item['data']) + 2) + item['data']
        if item['ex_data'] is not None: raw += item['ex_data']
    image.write_bytes(raw)
    assert DFLJPG.load(image) is None
    archive = image.parent / 'faceset.pak'
    archive.write_bytes(struct.pack('<QQ', 1, len(payload)) + payload)
    with pytest.raises(DataFormatError, match='Forbidden'): PackedFaceset.load(image.parent)


def test_sort_preview_archives_keep_old_trash_and_exact_sidecars_then_recover(tmp_path):
    directory = make_faces(tmp_path / 'aligned')
    old = tmp_path / 'aligned_trash'; old.mkdir(); (old / 'old.jpg').write_bytes(b'old trash')
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    selected = [[directory / 'face-1.jpg']]; rejected = [[directory / 'face-0.jpg']]
    preview = Sorter.final_process(directory, selected, rejected, dry_run=True)
    assert preview['dry_run'] and preview['receipt_path'] is None
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    receipt = Sorter.final_process(directory, selected, rejected)
    assert receipt['state'] == 'committed' and (old / 'old.jpg').read_bytes() == b'old trash'
    assert (directory / '00000.jpg').read_bytes() == original['face-1.jpg']
    assert (directory / '00000.jpg.landmarks.json').read_bytes() == original['face-1.jpg.landmarks.json']
    assert (Path(receipt['archive_path']) / 'discarded' / 'face-0.jpg').read_bytes() == original['face-0.jpg']
    result = tx.recover_transaction(receipt['receipt_path'])
    assert result['state'] == 'rolled_back'
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    assert tx.recover_transaction(receipt['receipt_path'])['reused']


def test_sort_exact_small_target_counts_and_pose_coverage_without_front_bias():
    rows = [[f'{i}.jpg', i + 1, None, -1.2 + i * 2.4 / 129, -.4 + .8 * (i % 8) / 7] for i in range(130)]
    for count in (1, 2, 32, 64, 65, 129, 130, 131, 2000):
        selected, rejected = Sorter.select_best_samples(rows, count)
        assert len(selected) == min(count, 130)
        assert len(selected) + len(rejected) == 130
        assert len({row[0] for row in selected + rejected}) == 130
    selected, _ = Sorter.select_best_samples(rows, 2)
    assert selected[0][3] > .8 and selected[1][3] < -.8
    with pytest.raises(ValueError): Sorter.select_best_samples(rows, 0)


def test_target_collision_duplicate_source_and_newer_output_preserve_data(tmp_path):
    directory = tmp_path / 'aligned'; directory.mkdir()
    (directory / 'a.jpg').write_bytes(b'a'); (directory / 'untouched.jpg').write_bytes(b'untouched')
    with pytest.raises(tx.TransactionConflict):
        tx.execute_plan(directory, [{'source': 'a.jpg', 'target': 'untouched.jpg'}], operation='sort')
    with pytest.raises(tx.TransactionConflict):
        tx.execute_plan(directory, [{'source': 'a.jpg', 'target': 'x.jpg'}, {'source': 'a.jpg', 'target': 'y.jpg'}], operation='sort')
    receipt = tx.execute_plan(directory, [{'source': 'a.jpg', 'target': 'new.jpg'}], operation='sort')
    (directory / 'new.jpg').write_bytes(b'newer edited content')
    assert tx.recover_transaction(receipt['receipt_path'], dry_run=True)['conflicts']
    with pytest.raises(tx.TransactionConflict): tx.recover_transaction(receipt['receipt_path'])
    assert (directory / 'new.jpg').read_bytes() == b'newer edited content'
    assert (directory / 'untouched.jpg').read_bytes() == b'untouched'


def test_write_failure_before_commit_leaves_every_original(tmp_path, monkeypatch):
    directory = make_faces(tmp_path / 'aligned')
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    def no_space(*args, **kwargs): raise OSError('No space left on device')
    monkeypatch.setattr(tx.shutil, 'copyfile', no_space)
    with pytest.raises(OSError, match='No space'):
        Sorter.final_process(directory, [[directory / 'face-0.jpg']], [[directory / 'face-1.jpg']])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    assert not (directory / '.faceset-transaction.lock').exists()


def test_streamed_replacement_binds_sha_without_loading_payload_bytes(tmp_path, monkeypatch):
    directory = tmp_path / 'frames'; directory.mkdir()
    original = directory / 'a.png'; original.write_bytes(b'original frame')
    payload = tmp_path / 'denoised.png'; payload.write_bytes(b'processed frame')
    original_sha, output_sha = tx.sha256(original), tx.sha256(payload)
    def no_bulk_read(_path):
        raise AssertionError('streamed replacement must not call Path.read_bytes')
    with monkeypatch.context() as scope:
        scope.setattr(Path, 'read_bytes', no_bulk_read)
        receipt = tx.execute_plan(directory, [{'source': original.name, 'target': original.name,
            'source_sha256': original_sha, 'payload_path': payload, 'payload_sha256': output_sha}], operation='denoise')
    assert original.read_bytes() == payload.read_bytes() == b'processed frame'
    assert receipt['entries'][0]['source_sha256'] == original_sha
    assert receipt['entries'][0]['target_sha256'] == output_sha
    tx.recover_transaction(receipt['receipt_path'])
    assert original.read_bytes() == b'original frame' and payload.read_bytes() == b'processed frame'


@pytest.mark.parametrize('field', ['source_sha256', 'payload_sha256', 'both-payloads', 'directory-payload'])
def test_streamed_replacement_rejects_unbound_or_ambiguous_inputs(tmp_path, field):
    directory = tmp_path / 'frames'; directory.mkdir()
    original = directory / 'a.png'; original.write_bytes(b'original')
    payload = tmp_path / 'new.png'; payload.write_bytes(b'new')
    change = {'source': original.name, 'target': original.name, 'payload_path': payload}
    if field == 'both-payloads': change['payload'] = b'new'
    elif field == 'directory-payload': change['payload_path'] = tmp_path
    else: change[field] = '0' * 64
    with pytest.raises(tx.TransactionConflict):
        tx.execute_plan(directory, [change], operation='denoise')
    assert original.read_bytes() == b'original' and not (directory / '.faceset-transaction.lock').exists()


@pytest.mark.parametrize('phase', ['staging', 'prepared'])
def test_changed_streamed_payload_is_rejected_before_source_mutation(tmp_path, monkeypatch, phase):
    directory = tmp_path / 'frames'; directory.mkdir()
    original = directory / 'a.png'; original.write_bytes(b'original')
    payload = tmp_path / 'new.png'; payload.write_bytes(b'new')
    copyfile, journal = tx.shutil.copyfile, tx._journal
    if phase == 'staging':
        def change_after_copy(source, destination, *args, **kwargs):
            result = copyfile(source, destination, *args, **kwargs)
            if Path(source) == payload: payload.write_bytes(b'changed')
            return result
        monkeypatch.setattr(tx.shutil, 'copyfile', change_after_copy)
    else:
        def change_after_prepared(path, receipt):
            journal(path, receipt)
            if receipt['state'] == 'prepared': payload.write_bytes(b'changed')
        monkeypatch.setattr(tx, '_journal', change_after_prepared)
    with pytest.raises(tx.TransactionConflict, match='Replacement payload changed'):
        tx.execute_plan(directory, [{'source': original.name, 'target': original.name,
            'payload_path': payload, 'payload_sha256': tx.sha256(payload)}], operation='denoise')
    assert original.read_bytes() == b'original'
    assert not (directory / '.faceset-transaction.lock').exists()


def test_publish_failure_and_cancel_after_first_output_restore_whole_batch(tmp_path, monkeypatch):
    directory = make_faces(tmp_path / 'aligned')
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    real_publish = tx._exclusive_publish
    calls = 0
    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError('injected publish failure')
        return real_publish(source, target)
    monkeypatch.setattr(tx, '_exclusive_publish', fail_once)
    with pytest.raises(OSError, match='injected'):
        Sorter.final_process(directory, [[directory / 'face-1.jpg'], [directory / 'face-0.jpg']], [])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    monkeypatch.setattr(tx, '_exclusive_publish', real_publish)
    checks = 0
    def cancel():
        nonlocal checks
        checks += 1
        return checks == 5
    with pytest.raises(InterruptedError):
        tx.execute_plan(directory, [{'source': 'face-0.jpg', 'target': 'a.jpg'},
                                    {'source': 'face-1.jpg', 'target': 'b.jpg'}], operation='sort', cancel_check=cancel)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original


def test_actual_process_exit_during_publish_is_explicitly_recoverable(tmp_path):
    directory = tmp_path / 'aligned'; directory.mkdir()
    (directory / 'a.jpg').write_bytes(b'a'); (directory / 'b.jpg').write_bytes(b'b')
    code = """import os,sys
sys.path.insert(0, sys.argv[1])
from core import faceset_transaction as tx
real = tx._journal
def journal(path, receipt):
    real(path, receipt)
    if receipt['state'] == 'publishing': os._exit(77)
tx._journal = journal
tx.execute_plan(sys.argv[2], [{'source':'a.jpg','target':'b.jpg'}, {'source':'b.jpg','target':'a.jpg'}], operation='sort')
"""
    result = subprocess.run([sys.executable, '-c', code, str(BACKEND), str(directory)], timeout=20, capture_output=True)
    assert result.returncode == 77, result.stderr.decode()
    receipt = next((tmp_path / 'aligned_trash').glob('faceset-*/receipt.json'))
    assert not (directory / 'a.jpg').exists() and (directory / '.faceset-transaction.lock').exists()
    tx.recover_transaction(receipt)
    assert (directory / 'a.jpg').read_bytes() == b'a' and (directory / 'b.jpg').read_bytes() == b'b'
    assert not (directory / '.faceset-transaction.lock').exists()


def test_recovery_refuses_live_transaction_then_accepts_stopped_preparation(tmp_path):
    directory = tmp_path / 'aligned'; directory.mkdir()
    (directory / 'a.jpg').write_bytes(b'a')
    code = """import os,sys
sys.path.insert(0, sys.argv[1])
from core import faceset_transaction as tx
real = tx._journal
def journal(path, receipt):
    real(path, receipt)
    if receipt['state'] == 'preparing':
        print(path, flush=True)
        sys.stdin.readline()
        os._exit(77)
tx._journal = journal
tx.execute_plan(sys.argv[2], [{'source':'a.jpg','target':'x.jpg'}], operation='sort')
"""
    process = subprocess.Popen([sys.executable, '-c', code, str(BACKEND), str(directory)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        receipt = Path(process.stdout.readline().strip())
        assert receipt.is_file()
        with pytest.raises(tx.TransactionConflict, match='may still be running'):
            tx.recover_transaction(receipt)
    finally:
        process.communicate('\n', timeout=20)
    assert process.returncode == 77
    result = tx.recover_transaction(receipt)
    assert result['state'] == 'rolled_back' and (directory / 'a.jpg').read_bytes() == b'a'
    assert not (directory / '.faceset-transaction.lock').exists()


def test_tampered_backup_or_receipt_cannot_modify_published_files(tmp_path):
    directory = tmp_path / 'aligned'; directory.mkdir()
    (directory / 'a.jpg').write_bytes(b'a')
    receipt = tx.execute_plan(directory, [{'source': 'a.jpg', 'target': 'x.jpg'}], operation='sort')
    path = Path(receipt['receipt_path'])
    backup = path.parent / receipt['entries'][0]['backup']
    backup.write_bytes(b'changed')
    with pytest.raises(tx.TransactionConflict, match='backup changed'): tx.recover_transaction(path)
    assert (directory / 'x.jpg').read_bytes() == b'a'
    backup.write_bytes(b'a')
    data = json.loads(path.read_text()); data['entries'][0]['target'] = '../outside.jpg'
    path.write_text(json.dumps(data))
    with pytest.raises(tx.TransactionConflict, match='Unsafe'): tx.recover_transaction(path)
    assert (directory / 'x.jpg').read_bytes() == b'a'


def test_metadata_manifest_tampering_stops_before_any_image_write(tmp_path):
    directory = make_faces(tmp_path / 'aligned')
    Util.save_faceset_metadata_folder(directory)
    manifest = directory / 'meta.dat.manifest.json'
    value = json.loads(manifest.read_text()); value['metadata_sha256'] = '0' * 64
    manifest.write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match='SHA'): Util.restore_faceset_metadata_folder(directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_restore_preserves_68_98_polygons_mask_and_unique_meta_then_rollback(tmp_path):
    directory = make_faces(tmp_path / 'aligned')
    metadata = {p.name: DFLJPG.load(p).get_dict() for p in directory.glob('*.jpg')}
    Util.save_faceset_metadata_folder(directory)
    saved_meta = (directory / 'meta.dat').read_bytes()
    for path in directory.glob('*.jpg'):
        assert cv2.imwrite(str(path), np.full((32, 32, 3), 17, np.uint8))
    edited = {p.name: p.read_bytes() for p in directory.iterdir()}
    preview = Util.restore_faceset_metadata_folder(directory, dry_run=True)
    assert preview['resized'] and {p.name: p.read_bytes() for p in directory.iterdir()} == edited
    receipt = Util.restore_faceset_metadata_folder(directory)
    assert receipt['state'] == 'committed' and (directory / 'meta.dat').read_bytes() == saved_meta
    for path in directory.glob('*.jpg'):
        restored = DFLJPG.load(path)
        assert restored.get_shape() == (64, 64, 3)
        assert restored.get_dict() == metadata[path.name]
        assert not path.with_suffix('.jpg.landmarks.json').exists()
    tx.recover_transaction(receipt['receipt_path'])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == edited


def test_missing_image_and_bad_backup_do_not_partially_restore(tmp_path):
    directory = make_faces(tmp_path / 'aligned')
    Util.save_faceset_metadata_folder(directory)
    (directory / 'face-1.jpg').unlink()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match='图片缺失'): Util.restore_faceset_metadata_folder(directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    (directory / 'meta.dat').write_bytes(b'cos\nsystem\n(S\'echo forbidden\'\ntR.')
    with pytest.raises(ValueError, match='Forbidden'): Util.restore_faceset_metadata_folder(directory)
    assert (directory / 'face-0.jpg').read_bytes() == before['face-0.jpg']


def test_second_metadata_save_retains_first_unique_backup(tmp_path):
    directory = make_faces(tmp_path / 'aligned', 1)
    Util.save_faceset_metadata_folder(directory)
    before = (directory / 'meta.dat').read_bytes()
    dfl = DFLJPG.load(directory / 'face-0.jpg'); dfl.set_source_filename('new.png'); dfl.save()
    Util.save_faceset_metadata_folder(directory)
    receipt = next((tmp_path / 'aligned_history').glob('faceset-*/receipt.json'))
    backups = json.loads(receipt.read_text())['entries']
    entry = next(entry for entry in backups if entry['source'] == 'meta.dat')
    assert (receipt.parent / entry['backup']).read_bytes() == before
    assert (directory / 'meta.dat').read_bytes() != before


def test_verified_generic_xseg_and_fan_hashes_and_arrays_are_unchanged():
    for path, count in ((REPO / '_internal/model_generic_xseg/XSeg_256.pth', 222),
                        (BACKEND / 'facelib/2DFAN.npy', 945)):
        before = tx.sha256(path)
        mapping = load_file(path, profile='weights')
        assert len(mapping) == count and all(isinstance(array, np.ndarray) for array in mapping.values())
        assert all(np.isfinite(array).all() for array in mapping.values())
        assert tx.sha256(path) == before


def test_actual_production_metadata_copy_roundtrips_and_recovers_without_source_writes(tmp_path):
    source = REPO / 'workspace/.vision-evaluation/production-landmarks-20261006-v2/tufa-whole_face/sample0_0.jpg'
    if not source.is_file(): pytest.skip('Local production evidence is not shipped as user media')
    before = tx.sha256(source)
    directory = tmp_path / 'aligned'; directory.mkdir()
    shutil.copyfile(source, directory / source.name)
    assert DFLJPG.load(directory / source.name).has_data()
    receipt = Sorter.final_process(directory, [[directory / source.name]], [])
    assert tx.sha256(directory / '00000.jpg') == before
    tx.recover_transaction(receipt['receipt_path'])
    Util.save_faceset_metadata_folder(directory)
    receipt = Util.restore_faceset_metadata_folder(directory)
    np.testing.assert_array_equal(DFLJPG.load(directory / source.name).get_landmarks(), DFLJPG.load(source).get_landmarks())
    assert DFLJPG.load(directory / source.name).get_dict()['native_landmarks98'] == DFLJPG.load(source).get_dict()['native_landmarks98']
    assert tx.sha256(source) == before


def test_actual_production_faces_run_legacy_selector_and_preview_without_changing_sources(tmp_path, monkeypatch):
    source = REPO / 'workspace/.vision-evaluation/production-landmarks-20261006-v2/tufa-whole_face'
    paths = list(source.glob('sample*_0.jpg'))
    if len(paths) < 3: pytest.skip('Local production media is not shipped as a test fixture')
    directory = tmp_path / 'aligned'; directory.mkdir()
    hashes = {path.name: tx.sha256(path) for path in paths}
    for path in paths: shutil.copyfile(path, directory / path.name)
    monkeypatch.setattr(Sorter.multiprocessing, 'cpu_count', lambda: 1)
    receipt = Sorter.main(directory, sort_by_method='final-by-size', target_count=2, dry_run=True)
    assert receipt['details']['selected_count'] == 2 and receipt['details']['archived_count'] == len(paths) - 2
    assert len(receipt['details']['scores']) == len(paths)
    assert all(row['metric'] == 'source_box_area' for row in receipt['details']['scores'])
    assert {path.name: tx.sha256(path) for path in paths} == hashes
    assert {path.name: tx.sha256(path) for path in directory.glob('*.jpg')} == hashes
