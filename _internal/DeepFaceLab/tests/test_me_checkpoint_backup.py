import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from me_backend.checkpoint_backup import list_backups, restore_backup
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.web_bridge import TrainingHeartbeat
from tests.me_fixtures import make_aligned


ROOT = Path(__file__).resolve().parents[1]


def _cli(*arguments, env=None):
    result = subprocess.run([sys.executable, str(ROOT / 'me.py'), *map(str, arguments)],
                            capture_output=True, text=True, encoding='utf-8', timeout=120,
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', **(env or {})})
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_automatic_generations_restore_and_heartbeat(tmp_path):
    src, dst = make_aligned(tmp_path / 'src'), make_aligned(tmp_path / 'dst', offset=12)
    model = tmp_path / 'model'
    heartbeat = tmp_path / 'heartbeat.json'
    config = MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16,
                      batch_size=1)
    common = ['--src', src, '--dst', dst, '--model', model, '--device', 'cpu', '--threads', '2',
              '--save-every', '1', '--backup-every', '1', '--backup-keep', '2']
    _cli('web-train', *common, '--name', 'model', '--config-json', json.dumps(config.to_dict()),
         '--steps', '3', env={'DFL_WEB_HEARTBEAT_FILE': str(heartbeat)})
    status = json.loads(heartbeat.read_text(encoding='utf-8'))
    assert status['phase'] == 'finished' and status['iteration'] == 3
    assert status['pid'] > 0 and status['at'] and status['progressAt'] and status['startedAt']
    backups = list_backups(model, verify=True)
    assert [entry['iteration'] for entry in backups] == [3, 2]
    assert all(entry['kind'] == 'automatic' for entry in backups)
    assert [item['iteration'] for item in json.loads(_cli('list-backups', '--model', model).stdout)['backups']] == [3, 2]
    _cli('web-train', *common, '--name', 'model', '--resume', '--steps', '1')
    assert MEEngine.load(model / 'me.pt', 'cpu').iteration == 4
    selected = next(entry for entry in list_backups(model, verify=True) if entry['iteration'] == 3)
    result = json.loads(_cli('restore-backup', '--model', model, '--backup-id', selected['id']).stdout)
    assert result['restoredIteration'] == 3
    assert result['rollbackId'].startswith('manual/')
    assert result['historyStatus'] == 'backed-up'
    assert MEEngine.load(model / 'me.pt', 'cpu').iteration == 3
    assert json.loads((model / 'metadata.json').read_text(encoding='utf-8'))['iteration'] == 3
    assert [json.loads(line)['iteration'] for line in (model / 'loss-history.jsonl').read_text().splitlines()] == [1, 2, 3]
    assert next(entry for entry in list_backups(model, verify=True) if entry['id'] == result['rollbackId'])['iteration'] == 4
    legacy = model / 'backups' / 'iter-00000003-1234567890123456789'
    legacy.mkdir()
    for filename in ('me.pt', 'metadata.json'):
        shutil.copy2(model / filename, legacy / filename)
    _cli('web-train', *common, '--name', 'model', '--resume', '--steps', '1')
    assert [json.loads(line)['iteration'] for line in (model / 'loss-history.jsonl').read_text().splitlines()] == [1, 2, 3, 4]
    legacy_id = 'legacy/iter-00000003-1234567890123456789'
    assert any(entry['id'] == legacy_id for entry in list_backups(model, verify=True))
    legacy_result = restore_backup(model, legacy_id)
    assert legacy_result['restoredIteration'] == 3
    assert legacy_result['historyStatus'] == 'reconstructed-from-current'
    assert [json.loads(line)['iteration'] for line in (model / 'loss-history.jsonl').read_text().splitlines()] == [1, 2, 3]


def test_corrupt_backup_is_rejected_without_touching_current_checkpoint(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    engine = MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                               d_mask_dims=16, batch_size=1), 'cpu')
    engine.save(model / 'me.pt')
    (model / 'metadata.json').write_text(json.dumps({'modelClass': 'ME', 'iteration': 0}), encoding='utf-8')
    from me_backend.checkpoint_backup import create_backup
    backup = create_backup(model, 0)
    current = (model / 'me.pt').read_bytes()
    with (Path(backup['path']) / 'me.pt').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore_backup(model, backup['id'])
    assert (model / 'me.pt').read_bytes() == current


def test_verified_generation_recovers_corrupt_primary_and_preserves_raw_bytes(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    engine = MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                               d_mask_dims=16, batch_size=1), 'cpu')
    engine.save(model / 'me.pt')
    (model / 'metadata.json').write_text(json.dumps({'modelClass':'ME', 'iteration':0,
        'config':engine.config.to_dict()}), encoding='utf-8')
    from me_backend.checkpoint_backup import create_backup
    backup = create_backup(model, 0)
    (model / 'me.pt').write_bytes(b'broken-checkpoint')
    (model / 'metadata.json').write_text('{broken', encoding='utf-8')
    result = restore_backup(model, backup['id'])
    assert result['restoredIteration'] == 0 and result['rollbackId'] is None
    assert MEEngine.load(model / 'me.pt', 'cpu').iteration == 0
    quarantine = Path(result['quarantinedPrimary'])
    assert (quarantine / 'me.pt').read_bytes() == b'broken-checkpoint'
    assert (quarantine / 'metadata.json').read_text(encoding='utf-8') == '{broken'


def test_transient_heartbeat_write_error_does_not_stop_training(tmp_path, monkeypatch):
    from me_backend import web_bridge
    original = web_bridge.atomic_json
    attempts = 0

    def flaky(target, value):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError('temporary Windows file sharing conflict')
        return original(target, value)

    monkeypatch.setattr(web_bridge, 'atomic_json', flaky)
    target = tmp_path / 'heartbeat.json'
    heartbeat = TrainingHeartbeat(target, 0)
    heartbeat.start()
    heartbeat.update(phase='training', iteration=1, progressed=True)
    heartbeat.close('finished')
    state = json.loads(target.read_text(encoding='utf-8'))
    assert state['iteration'] == 1 and state['phase'] == 'finished'
    assert attempts >= 2
