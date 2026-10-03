"""Verified, bounded backup generations for a single ME model directory."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time


BACKUP_ID = re.compile(r'^(automatic|manual|legacy)/iter-(\d{8,12})-(\d{10,20})$')
BACKUP_FILES = ('me.pt', 'metadata.json')
HISTORY_FILE = 'loss-history.jsonl'


def _sha256(target):
    digest = hashlib.sha256()
    with Path(target).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(target, value):
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _durable_copy(source, destination):
    with Path(source).open('rb') as reader, Path(destination).open('wb') as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _backup_root(model_directory):
    return Path(model_directory).resolve() / 'backups'


def _copy_history_prefix(source, destination, iteration, require_exact=False):
    """Copy a consecutive history through iteration without copying newer records."""
    expected = 1
    with Path(source).open('rb') as reader, Path(destination).open('wb') as writer:
        for line in reader:
            if expected > iteration:
                if require_exact:
                    raise ValueError('ME backup history extends past its checkpoint')
                break
            if len(line) > 1_000_000 or not line.endswith(b'\n'):
                raise ValueError('ME loss history has an incomplete record')
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError('ME loss history has an invalid record') from error
            if type(record.get('iteration')) is not int or record['iteration'] != expected:
                raise ValueError('ME loss history does not match its checkpoint')
            writer.write(line)
            expected += 1
        if expected != iteration + 1:
            raise ValueError('ME loss history ends before its checkpoint')
        writer.flush()
        os.fsync(writer.fileno())


def _verify_backup_directory(backup, expected_iteration=None):
    metadata = json.loads((backup / 'metadata.json').read_text(encoding='utf-8'))
    if backup.parent.name == 'backups':
        # Backups made by the previous Web bridge have no checksum manifest.
        # Recompute their digests and validate the checkpoint before restore.
        manifest = dict(schemaVersion=1, kind='legacy', iteration=metadata.get('iteration'),
                        createdAt=datetime.fromtimestamp(backup.stat().st_mtime, timezone.utc).isoformat().replace('+00:00', 'Z'),
                        sha256={name: _sha256(backup / name) for name in (*BACKUP_FILES, HISTORY_FILE)
                                if (backup / name).is_file()})
    else:
        manifest = json.loads((backup / 'backup.json').read_text(encoding='utf-8'))
        if manifest.get('schemaVersion') != 1 or manifest.get('kind') != backup.parent.name:
            raise ValueError('ME backup manifest is invalid')
    iteration = manifest.get('iteration')
    if type(iteration) is not int or iteration < 0 or metadata.get('iteration') != iteration:
        raise ValueError('ME backup iteration does not match its metadata')
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError('ME backup ID and iteration differ')
    for name in (*BACKUP_FILES, *([HISTORY_FILE] if HISTORY_FILE in manifest.get('sha256', {}) else [])):
        if not (backup / name).is_file() or not re.fullmatch(r'[a-f0-9]{64}', manifest.get('sha256', {}).get(name, '')):
            raise ValueError(f'ME backup is missing {name} or its digest')
        if _sha256(backup / name) != manifest['sha256'][name]:
            raise ValueError(f'ME backup {name} failed checksum verification')
    return manifest


def list_backups(model_directory, verify=False):
    root = _backup_root(model_directory)
    backups = []
    for kind in ('automatic', 'manual', 'legacy'):
        group = root if kind == 'legacy' else root / kind
        if not group.is_dir():
            continue
        for directory in group.iterdir():
            if not directory.is_dir():
                continue
            backup_id = f'{kind}/{directory.name}'
            match = BACKUP_ID.fullmatch(backup_id)
            if not match:
                continue
            try:
                manifest = _verify_backup_directory(directory, int(match[2])) if verify or kind == 'legacy' else json.loads(
                    (directory / 'backup.json').read_text(encoding='utf-8'))
                if manifest.get('iteration') != int(match[2]):
                    continue
                backups.append({**manifest, 'id': backup_id})
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
    return sorted(backups, key=lambda item: (item['iteration'], item['id']), reverse=True)


def create_backup(model_directory, iteration, kind='automatic', keep=3):
    """Copy a completed checkpoint atomically, then prune only older automatic generations."""
    if kind not in ('automatic', 'manual') or type(iteration) is not int or iteration < 0:
        raise ValueError('Invalid ME backup kind or iteration')
    if type(keep) is not int or not 2 <= keep <= 32:
        raise ValueError('ME backup retention must keep 2..32 generations')
    model_directory = Path(model_directory).resolve()
    metadata = json.loads((model_directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('iteration') != iteration or metadata.get('modelClass') != 'ME':
        raise ValueError('ME checkpoint metadata does not match the backup iteration')
    group = _backup_root(model_directory) / kind
    group.mkdir(parents=True, exist_ok=True)
    backup_id = f'iter-{iteration:08d}-{time.time_ns()}'
    destination = group / backup_id
    staging = Path(tempfile.mkdtemp(prefix=f'.{backup_id}-', dir=group))
    try:
        sha256 = {}
        for name in BACKUP_FILES:
            _durable_copy(model_directory / name, staging / name)
            sha256[name] = _sha256(staging / name)
        if (model_directory / HISTORY_FILE).is_file():
            _copy_history_prefix(model_directory / HISTORY_FILE, staging / HISTORY_FILE,
                                 iteration, require_exact=True)
            sha256[HISTORY_FILE] = _sha256(staging / HISTORY_FILE)
        manifest = dict(schemaVersion=1, kind=kind, iteration=iteration,
                        createdAt=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        sha256=sha256)
        _atomic_json(staging / 'backup.json', manifest)
        _verify_backup_directory(staging, iteration)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if kind == 'automatic':
        generations = [item for item in list_backups(model_directory, verify=True) if item['kind'] == 'automatic']
        for old in generations[keep:]:
            old_path = group / old['id'].split('/', 1)[1]
            shutil.rmtree(old_path)
    return dict(manifest, id=f'{kind}/{backup_id}', path=str(destination))


def restore_backup(model_directory, backup_id):
    """Restore one explicitly selected, verified generation; retain a rollback copy."""
    match = BACKUP_ID.fullmatch(str(backup_id))
    if not match:
        raise ValueError('ME backup ID is invalid')
    model_directory = Path(model_directory).resolve()
    group = _backup_root(model_directory) if match[1] == 'legacy' else _backup_root(model_directory) / match[1]
    source = group / f'iter-{match[2]}-{match[3]}'
    if not source.is_dir():
        raise FileNotFoundError('ME backup does not exist')
    manifest = _verify_backup_directory(source, int(match[2]))
    from .engine import MEEngine
    restored = MEEngine.load(source / 'me.pt', 'cpu')
    if restored.iteration != manifest['iteration']:
        raise ValueError('ME backup checkpoint iteration does not match its metadata')
    restored_metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    if restored_metadata.get('modelClass') != 'ME' or restored_metadata.get('config') != restored.config.to_dict():
        raise ValueError('ME backup metadata does not match its checkpoint')
    current_checkpoint = model_directory / 'me.pt'
    current_metadata = model_directory / 'metadata.json'
    if not model_directory.is_dir():
        raise FileNotFoundError('ME model directory does not exist')
    rollback = None
    quarantine = None
    try:
        current = json.loads(current_metadata.read_text(encoding='utf-8'))
        if not current_checkpoint.is_file() or current.get('modelClass') != 'ME':
            raise ValueError('Current ME checkpoint is incomplete')
        current_engine = MEEngine.load(current_checkpoint, 'cpu')
        if current_engine.iteration != current.get('iteration') or current_engine.config.to_dict() != current.get('config'):
            raise ValueError('Current ME checkpoint and metadata differ')
        rollback = create_backup(model_directory, current['iteration'], kind='manual')
    except Exception:
        # A broken primary checkpoint must not make a verified older generation
        # unrestorable. Preserve its raw bytes before replacing either file.
        unsafe = _backup_root(model_directory) / 'unsafe'
        unsafe.mkdir(parents=True, exist_ok=True)
        quarantine = unsafe / f'restore-{time.time_ns()}'
        staging_unsafe = Path(tempfile.mkdtemp(prefix='.restore-unsafe-', dir=unsafe))
        try:
            for name in (*BACKUP_FILES, HISTORY_FILE):
                if (model_directory / name).is_file():
                    _durable_copy(model_directory / name, staging_unsafe / name)
            _atomic_json(staging_unsafe / 'recovery.json', dict(createdAt=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        reason='Unverified primary bytes preserved before restoring a verified ME backup'))
            os.replace(staging_unsafe, quarantine)
        finally:
            if staging_unsafe.exists():
                shutil.rmtree(staging_unsafe)
    pending = Path(tempfile.mkdtemp(prefix='.restore-', dir=model_directory))
    history_status = 'absent'
    try:
        for name in BACKUP_FILES:
            _durable_copy(source / name, pending / name)
        if (source / HISTORY_FILE).is_file():
            _copy_history_prefix(source / HISTORY_FILE, pending / HISTORY_FILE,
                                 manifest['iteration'], require_exact=True)
            history_status = 'backed-up'
        elif (model_directory / HISTORY_FILE).is_file():
            _copy_history_prefix(model_directory / HISTORY_FILE, pending / HISTORY_FILE,
                                 manifest['iteration'])
            history_status = 'reconstructed-from-current'
        if _sha256(pending / 'me.pt') != manifest['sha256']['me.pt']:
            raise ValueError('ME staged checkpoint checksum changed')
        if history_status == 'backed-up' and _sha256(pending / HISTORY_FILE) != manifest['sha256'][HISTORY_FILE]:
            raise ValueError('ME staged history checksum changed')
        os.replace(pending / 'me.pt', current_checkpoint)
        os.replace(pending / 'metadata.json', current_metadata)
        if history_status != 'absent':
            os.replace(pending / HISTORY_FILE, model_directory / HISTORY_FILE)
    except BaseException:
        rollback_directory = _backup_root(model_directory) / rollback['id'] if rollback else quarantine
        for name in (*BACKUP_FILES, HISTORY_FILE):
            if (rollback_directory / name).exists():
                temporary = model_directory / f'.{name}.rollback-{os.getpid()}'
                _durable_copy(rollback_directory / name, temporary)
                os.replace(temporary, model_directory / name)
        raise
    finally:
        shutil.rmtree(pending)
    return dict(restoredIteration=restored.iteration, backupId=backup_id,
                rollbackId=rollback['id'] if rollback else None,
                quarantinedPrimary=str(quarantine) if quarantine else None,
                historyStatus=history_status)
