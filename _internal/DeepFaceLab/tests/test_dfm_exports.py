"""Actual ONNX bundle, export isolation and precision boundary regressions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from dfm_export.exporter import DFMNetwork, export_dfm, file_sha256
from dfm_export.quantization import normalize_me_layers, precision_plan
from dfm_export import publication
from dfm_export.publication import (PublicationRecoveryError, publish_bundle,
                                    recover_publication, transaction_journal_path)
from me_backend.config import MEConfig
from me_backend.engine import MEEngine

torch.set_num_threads(2)


def small(archi='liae-ud'):
    return MEConfig(archi=archi, resolution=64, ae_dims=32, e_dims=16,
                    d_dims=16, d_mask_dims=16, batch_size=1)


def test_external_bundle_relocates_and_runs_dynamic_batch(tmp_path):
    directory = tmp_path / 'model'
    engine = MEEngine(small(), 'cpu')
    engine.save(directory / 'me.pt')
    checkpoint_sha = file_sha256(directory / 'me.pt')
    path = export_dfm(directory, external_threshold=1, export_id='test-job')
    receipt = json.loads(path.with_suffix('.dfm.export.json').read_text())
    assert receipt['external_data'] and receipt['export_id'] == 'test-job'
    assert checkpoint_sha == file_sha256(directory / 'me.pt')
    relocated = tmp_path / 'moved'
    relocated.mkdir()
    for item in receipt['files']:
        shutil.copyfile(directory / item['name'], relocated / item['name'])
        assert file_sha256(relocated / item['name']) == item['sha256']
    data = np.random.default_rng(4).random((2, 64, 64, 3), dtype=np.float32)
    session = ort.InferenceSession(str(relocated / path.name), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        expected = DFMNetwork(engine.network)(torch.from_numpy(data))
    for actual, ref in zip(session.run(None, {'in_face:0': data}), expected):
        np.testing.assert_allclose(actual, ref.numpy(), atol=2e-6, rtol=2e-4)
    # Missing a sidecar must fail instead of returning a runnable partial model.
    del session
    (relocated / next(x['name'] for x in receipt['files'] if x['name'].endswith('.data'))).unlink()
    with pytest.raises(Exception):
        ort.InferenceSession(str(relocated / path.name), providers=['CPUExecutionProvider'])


def test_failed_export_preserves_previous_model_and_checkpoint(tmp_path):
    engine = MEEngine(small(), 'cpu')
    engine.save(tmp_path / 'me.pt')
    checkpoint_sha = file_sha256(tmp_path / 'me.pt')
    output = tmp_path / 'model.dfm'
    output.write_bytes(b'previous export')
    with patch('onnx.checker.check_model', side_effect=RuntimeError('validation failure')):
        with pytest.raises(RuntimeError, match='validation failure'):
            export_dfm(tmp_path, external_threshold=1)
    assert output.read_bytes() == b'previous export'
    assert file_sha256(tmp_path / 'me.pt') == checkpoint_sha
    assert not list(tmp_path.glob('.dfm-export-*'))
    assert not list(tmp_path.glob('*.data'))


@pytest.mark.parametrize('archi', ['liae', 'liae-ud', 'df-ud', 'liae-utc'])
def test_private_quantization_layer_conversion_preserves_network(archi):
    engine = MEEngine(small(archi), 'cpu')
    module = DFMNetwork(engine.network).eval()
    data = torch.rand(2, 64, 64, 3)
    with torch.no_grad():
        expected = module(data)
        normalize_me_layers(module)
        actual = module(data)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    plan = precision_plan(module)
    assert any(value == 'nvfp4' for value in plan.values())
    assert any(value == 'fp8' for value in plan.values())
    assert all(value == 'fp16' for name, value in plan.items() if 'out_conv' in name or 'upscalem' in name)


def test_extreme_export_rejects_cpu_before_publication(tmp_path):
    MEEngine(small(), 'cpu').save(tmp_path / 'me.pt')
    with pytest.raises(ValueError, match='CUDA GPU'):
        export_dfm(tmp_path, mode='extreme', device='cpu')
    assert not (tmp_path / 'model_extreme.dfm').exists()


def test_me_bf16_is_explicit_and_mutually_exclusive():
    assert MEConfig.from_dict({'use_bf16': True}).use_bf16
    assert not MEConfig.from_dict({}).use_bf16
    with pytest.raises(ValueError, match='mutually exclusive'):
        MEConfig(use_fp16=True, use_bf16=True)
    engine = MEEngine(MEConfig(**{**small().to_dict(), 'use_bf16': True}), 'cpu')
    assert all(np.isfinite(value).all() for value in engine.predict(np.zeros((1, 3, 64, 64), np.float32)))
    with pytest.raises(ValueError, match='requires CUDA'):
        engine.train_step(None, None)


def publication_fixture(tmp_path, *, previous=True, stage_name='stage'):
    """Small inert bundle: no GPU, model loading, training or user media."""
    output = tmp_path / 'model.dfm'
    fixed = ['model.dfm.engine', 'model.dfm.layers.json', 'model.dfm', 'model.dfm.export.json']
    if previous:
        for name in fixed:
            (tmp_path / name).write_bytes(('old-' + name).encode())
    preserved = {'me.pt': b'checkpoint', 'unrelated.txt': b'user-file',
                 'model.dfm.' + '0' * 32 + '.data': b'old-external-weights'}
    for name, data in preserved.items():
        (tmp_path / name).write_bytes(data)
    stage = tmp_path / stage_name
    stage.mkdir()
    names = ['model.dfm.' + '1' * 32 + '.data', *fixed]
    files = []
    for name in names:
        source = stage / name
        source.write_bytes(('new-' + name).encode())
        files.append(source)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    return output, files, before


def assert_publication_rolled_back(output, before):
    parent = output.parent
    actual = {path.name: path.read_bytes() for path in parent.iterdir()
              if path.is_file() and not path.name.endswith('.publish.lock')}
    assert actual == before
    assert not transaction_journal_path(output).exists()
    assert not list(parent.glob('.dfm-publish-*'))


@pytest.mark.parametrize('previous', [True, False])
@pytest.mark.parametrize('failed_index', range(5))
@pytest.mark.parametrize('after_replace', [False, True])
def test_bundle_each_publication_failure_rolls_back(tmp_path, previous, failed_index, after_replace):
    output, files, before = publication_fixture(tmp_path, previous=previous)
    replace = os.replace
    failed = False

    def fault(source, target):
        nonlocal failed
        if Path(source) == files[failed_index] and not failed:
            failed = True
            if after_replace:
                replace(source, target)
            raise PermissionError('synthetic occupied publication target')
        return replace(source, target)

    with patch.object(publication.os, 'replace', side_effect=fault):
        with pytest.raises(PermissionError, match='synthetic occupied'):
            publish_bundle(output, files)
    assert failed
    assert_publication_rolled_back(output, before)
    # Retrying is an ordinary new export, with freshly generated staged files.
    for source in files:
        source.write_bytes(('new-' + source.name).encode())
    publish_bundle(output, files)
    for source in files:
        assert (tmp_path / source.name).read_bytes() == ('new-' + source.name).encode()
    assert (tmp_path / 'me.pt').read_bytes() == before['me.pt']
    assert not transaction_journal_path(output).exists()


def test_recovery_failure_retains_backups_then_retry_recovers(tmp_path):
    output, files, before = publication_fixture(tmp_path)
    replace = os.replace

    def fault(source, target):
        source = Path(source)
        if source == files[-1] or source.name.startswith('restore-'):
            raise PermissionError('synthetic file is still open')
        return replace(source, target)

    with patch.object(publication.os, 'replace', side_effect=fault):
        with pytest.raises(PublicationRecoveryError, match='recovery is pending'):
            publish_bundle(output, files)
    record = json.loads(transaction_journal_path(output).read_text())
    assert record['state'] == 'prepared'
    backup = tmp_path / record['backup_directory']
    for index, entry in enumerate(record['files']):
        if entry['old'] is not None:
            assert (backup / f'backup-{index}').read_bytes() == before[entry['name']]
    assert recover_publication(output)
    assert_publication_rolled_back(output, before)
    assert not recover_publication(output)


@pytest.mark.parametrize('last_file', [False, True])
def test_process_interruption_recovers_and_os_releases_lock(tmp_path, last_file):
    output, files, before = publication_fixture(tmp_path)
    code = "\n".join([
        'import os, sys',
        f'sys.path.insert(0, {str(Path(publication.__file__).parents[1])!r})',
        'from pathlib import Path',
        'from dfm_export import publication',
        'replace = os.replace',
        f'crash_at = {str(files[-1] if last_file else files[1])!r}',
        'def crash(source, target):',
        '    replace(source, target)',
        '    if str(source) == crash_at: os._exit(73)',
        'publication.os.replace = crash',
        f'publication.publish_bundle(Path({str(output)!r}), {[str(path) for path in files]!r})',
    ])
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, timeout=30)
    assert result.returncode == 73, result.stderr.decode(errors='replace')
    assert transaction_journal_path(output).exists()
    assert recover_publication(output)
    assert_publication_rolled_back(output, before)


def test_live_publication_lock_is_not_removed(tmp_path):
    output, files, _ = publication_fixture(tmp_path)
    with publication._publication_lock(output):
        with pytest.raises(PublicationRecoveryError, match='publication is active'):
            publish_bundle(output, files)
        with pytest.raises(PublicationRecoveryError, match='publication is active'):
            recover_publication(output)
    publish_bundle(output, files)
    assert not transaction_journal_path(output).exists()


class PublicationInterruption(BaseException):
    pass


def interrupt_publication(output, files):
    replace = os.replace

    def interrupt(source, target):
        replace(source, target)
        if Path(source) == files[-2]:
            raise PublicationInterruption()

    with patch.object(publication.os, 'replace', side_effect=interrupt):
        with pytest.raises(PublicationInterruption):
            publish_bundle(output, files)


@pytest.mark.parametrize('change', ['target', 'backup', 'journal_path', 'unexpected_backup_file'])
def test_recovery_conflicts_preserve_user_files_and_journal(tmp_path, change):
    output, files, before = publication_fixture(tmp_path)
    interrupt_publication(output, files)
    journal = transaction_journal_path(output)
    record = json.loads(journal.read_text())
    backup = tmp_path / record['backup_directory']
    if change == 'target':
        output.write_bytes(b'external-replacement')
    elif change == 'backup':
        (backup / 'backup-3').write_bytes(b'external-backup-replacement')
    elif change == 'journal_path':
        record['files'][0]['name'] = '../unrelated.txt'
        journal.write_text(json.dumps(record))
    else:
        (backup / 'user-file.txt').write_bytes(b'preserve this too')
    with pytest.raises(PublicationRecoveryError):
        recover_publication(output)
    assert journal.exists()
    assert backup.exists()
    assert (tmp_path / 'unrelated.txt').read_bytes() == before['unrelated.txt']
    assert (tmp_path / 'me.pt').read_bytes() == before['me.pt']
    if change == 'target':
        assert output.read_bytes() == b'external-replacement'
    if change == 'unexpected_backup_file':
        assert (backup / 'user-file.txt').read_bytes() == b'preserve this too'


def test_committed_cleanup_interruption_keeps_new_bundle(tmp_path):
    output, files, before = publication_fixture(tmp_path)
    with patch.object(publication, '_cleanup', side_effect=PublicationInterruption):
        with pytest.raises(PublicationInterruption):
            publish_bundle(output, files)
    assert json.loads(transaction_journal_path(output).read_text())['state'] == 'committed'
    assert recover_publication(output)
    for source in files:
        assert (tmp_path / source.name).read_bytes() == ('new-' + source.name).encode()
    assert (tmp_path / 'me.pt').read_bytes() == before['me.pt']
    assert not transaction_journal_path(output).exists()


def test_export_recovers_before_model_loading(tmp_path):
    output, files, before = publication_fixture(tmp_path)
    interrupt_publication(output, files)
    with patch('dfm_export.exporter.load_network', side_effect=ValueError('synthetic model load failure')):
        with pytest.raises(ValueError, match='synthetic model load failure'):
            export_dfm(tmp_path)
    assert_publication_rolled_back(output, before)


@pytest.mark.parametrize('journal_write', [1, 2])
@pytest.mark.parametrize('after_replace', [False, True])
def test_journal_write_failures_leave_a_complete_generation(tmp_path, journal_write, after_replace):
    output, files, before = publication_fixture(tmp_path)
    replace = os.replace
    writes = 0

    def fault(source, target):
        nonlocal writes
        if Path(target) == transaction_journal_path(output):
            writes += 1
            if writes == journal_write:
                if after_replace:
                    replace(source, target)
                raise PermissionError('synthetic journal replacement failure')
        return replace(source, target)

    with patch.object(publication.os, 'replace', side_effect=fault):
        with pytest.raises(PermissionError, match='synthetic journal'):
            publish_bundle(output, files)
    if journal_write == 2 and after_replace:
        # The durable commit was completed even though its caller lost the result.
        for source in files:
            assert (tmp_path / source.name).read_bytes() == ('new-' + source.name).encode()
        assert not transaction_journal_path(output).exists()
    else:
        assert_publication_rolled_back(output, before)


@pytest.mark.parametrize('after_directory_removed', [False, True])
def test_partial_cleanup_can_finish_without_backup_files(tmp_path, after_directory_removed):
    output, files, before = publication_fixture(tmp_path)
    operation = Path.rmdir if after_directory_removed else Path.unlink
    method = 'rmdir' if after_directory_removed else 'unlink'
    interrupted = False

    def interrupt(path, *args, **kwargs):
        nonlocal interrupted
        result = operation(path, *args, **kwargs)
        if not interrupted and (path.name.startswith('.dfm-publish-') if after_directory_removed
                                else path.name.startswith('backup-')):
            interrupted = True
            raise PublicationInterruption()
        return result

    with patch.object(Path, method, interrupt):
        with pytest.raises(PublicationInterruption):
            publish_bundle(output, files)
    assert interrupted
    assert transaction_journal_path(output).exists()
    assert recover_publication(output)
    assert not transaction_journal_path(output).exists()
    assert not list(tmp_path.glob('.dfm-publish-*'))
    assert (tmp_path / 'me.pt').read_bytes() == before['me.pt']


@pytest.mark.skipif(os.name != 'nt', reason='Windows file handle sharing contract')
def test_windows_open_graph_rolls_back_other_bundle_files(tmp_path):
    output, files, before = publication_fixture(tmp_path)
    with output.open('rb'):
        with pytest.raises(PermissionError):
            publish_bundle(output, files)
    assert_publication_rolled_back(output, before)


def test_committed_backup_modification_is_preserved(tmp_path):
    output, files, _ = publication_fixture(tmp_path)
    with patch.object(publication, '_cleanup', side_effect=PublicationInterruption):
        with pytest.raises(PublicationInterruption):
            publish_bundle(output, files)
    journal = transaction_journal_path(output)
    backup = tmp_path / json.loads(journal.read_text())['backup_directory'] / 'backup-1'
    backup.write_bytes(b'externally modified backup')
    with pytest.raises(PublicationRecoveryError, match='changed before cleanup'):
        recover_publication(output)
    assert journal.exists()
    assert backup.read_bytes() == b'externally modified backup'
