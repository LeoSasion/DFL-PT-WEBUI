"""Recoverable file batches for legacy faceset sorting and metadata restore.

All originals and all new bytes are staged and verified before a source is
removed. Independent batches never empty a previous trash/history directory.
Each committed output is recorded by SHA-256; recovery refuses edited output.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from core.process_identity import process_identity as _process_identity


class TransactionConflict(ValueError):
    pass




def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _plain_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise TransactionConflict('Unsafe transaction filename')
    return name


def _regular(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise TransactionConflict(f'Transaction requires an ordinary file: {path}')
    return path


def _journal(path, receipt):
    pending = path.with_suffix('.json.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _exclusive_publish(source, target):
    # Atomic no-clobber publication on the same filesystem. The transient
    # staging name is removed immediately: no persistent hardlink aliases to
    # the original or the recovery backup are created.
    os.link(source, target)
    source.unlink()


def _paths(receipt_path, receipt):
    receipt_path = Path(receipt_path).absolute()
    batch = receipt_path.parent
    if (receipt_path.name != 'receipt.json' or receipt_path.is_symlink()
            or not re.fullmatch(r'faceset-[0-9a-f]{32}', batch.name)
            or receipt.get('id') != batch.name or batch.resolve() != batch
            or receipt.get('schema') != 1):
        raise TransactionConflict('Invalid transaction receipt location/schema')
    name = _plain_name(receipt.get('input_name'))
    if batch.parent.name not in (name + '_trash', name + '_history'):
        raise TransactionConflict('Receipt history does not match its faceset')
    directory = batch.parent.parent / name
    if not directory.is_dir() or directory.resolve() != directory:
        raise TransactionConflict('Faceset path changed or contains a symlink')
    for name in ('originals', 'staging', 'discarded'):
        path = batch / name
        if not path.is_dir() or path.resolve() != path:
            raise TransactionConflict('Transaction subdirectory changed or contains a symlink')
    sources, targets = set(), set()
    entries = receipt.get('entries')
    if not isinstance(entries, list) or len(entries) > 500000:
        raise TransactionConflict('Invalid transaction entries')
    for index, entry in enumerate(entries):
        source_name = _plain_name(entry.get('source'))
        if source_name.casefold() in sources:
            raise TransactionConflict('Duplicate transaction source')
        sources.add(source_name.casefold())
        if entry.get('backup') != f'originals/{index:06d}.bin':
            raise TransactionConflict('Invalid original backup location')
        for key in ('source_sha256', 'target_sha256'):
            if entry.get(key) is not None and not re.fullmatch(r'[0-9a-f]{64}', entry[key]):
                raise TransactionConflict('Invalid receipt digest')
        target = entry.get('target')
        if target is not None:
            location = entry.get('location')
            _plain_name(target)
            if location not in ('input', 'discarded'):
                raise TransactionConflict('Invalid transaction target location')
            target_key = (location, target.casefold())
            if target_key in targets: raise TransactionConflict('Duplicate transaction target')
            targets.add(target_key)
    return directory, batch


def _target(directory, batch, entry):
    if entry['target'] is None: return None
    return (directory if entry['location'] == 'input' else batch / 'discarded') / entry['target']


def _recovery_conflicts(directory, batch, receipt):
    conflicts = []
    original_hashes = {entry['source'].casefold(): entry['source_sha256'] for entry in receipt['entries']}
    outputs = {str(_target(directory, batch, entry)).casefold(): entry['target_sha256']
               for entry in receipt['entries'] if entry['target'] is not None}
    for entry in receipt['entries']:
        backup = batch / entry['backup']
        if not backup.is_file() or backup.resolve() != backup or sha256(backup) != entry['source_sha256']:
            conflicts.append(f'Original backup changed/missing: {entry["source"]}')
        source = directory / entry['source']
        if source.exists():
            try: digest = sha256(_regular(source))
            except (OSError, ValueError): digest = None
            if digest not in (entry['source_sha256'], outputs.get(str(source).casefold())):
                conflicts.append(f'Source has newer/unknown data: {entry["source"]}')
        target = _target(directory, batch, entry)
        if target is not None and target.exists():
            try: digest = sha256(_regular(target))
            except (OSError, ValueError): digest = None
            allowed = [entry['target_sha256']]
            if entry['location'] == 'input': allowed.append(original_hashes.get(target.name.casefold()))
            if digest not in allowed:
                conflicts.append(f'Published output changed: {entry["target"]}')
    return conflicts


def recover_transaction(receipt_path, *, dry_run=False):
    receipt_path = _regular(Path(receipt_path).absolute())
    if receipt_path.stat().st_size > 128 * 1024 * 1024:
        raise TransactionConflict('Receipt is too large')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    directory, batch = _paths(receipt_path, receipt)
    if receipt['state'] == 'rolled_back':
        return {**receipt, 'receipt_path': str(receipt_path), 'reused': True}
    if receipt['state'] not in ('preparing', 'failed_before_commit', 'prepared', 'removing', 'publishing', 'committed', 'rollback_failed'):
        raise TransactionConflict('Batch has no complete original backup for recovery')
    if receipt['state'] != 'committed':
        owner_pid = receipt.get('owner_pid')
        if type(owner_pid) is not int or not 0 < owner_pid <= 2**32 - 1:
            raise TransactionConflict('Incomplete transaction owner identity')
        identity = _process_identity(owner_pid)
        if owner_pid != os.getpid() and identity is not None and identity in (receipt.get('owner_identity'), 'unknown'):
            raise TransactionConflict('Transaction process may still be running; stop it before recovery')
    if receipt['state'] in ('preparing', 'failed_before_commit'):
        # No source deletion is possible until the durable prepared/removing
        # state. A stopped preparation is recoverable without complete backups
        # only when every original still matches its recorded bytes.
        conflicts = []
        for entry in receipt['entries']:
            source = directory / entry['source']
            if not source.is_file() or sha256(_regular(source)) != entry['source_sha256']:
                conflicts.append(f'Original changed during aborted preparation: {entry["source"]}')
        if dry_run: return {**receipt, 'receipt_path': str(receipt_path), 'dry_run': True, 'conflicts': conflicts}
        if conflicts: raise TransactionConflict('; '.join(conflicts))
        receipt['state'] = 'rolled_back'; _journal(receipt_path, receipt)
        lock = directory / '.faceset-transaction.lock'
        if lock.exists():
            if lock.is_symlink() or lock.read_text(encoding='ascii') != receipt['id']:
                raise TransactionConflict('Another transaction owns the mutation lock')
            lock.unlink()
        return {**receipt, 'receipt_path': str(receipt_path)}
    conflicts = _recovery_conflicts(directory, batch, receipt)
    if dry_run:
        return {**receipt, 'receipt_path': str(receipt_path), 'dry_run': True, 'conflicts': conflicts}
    if conflicts:
        raise TransactionConflict('; '.join(conflicts))
    lock = directory / '.faceset-transaction.lock'
    owns_lock = False
    if lock.exists():
        if lock.is_symlink() or lock.read_text(encoding='ascii') != receipt['id']:
            raise TransactionConflict('Another faceset transaction owns the mutation lock')
    else:
        with lock.open('x', encoding='ascii') as stream: stream.write(receipt['id'])
        owns_lock = True
    try:
        for entry in receipt['entries']:
            target = _target(directory, batch, entry)
            if target is not None and target.exists() and sha256(target) == entry['target_sha256']:
                # Recheck immediately before each deletion. An interrupted
                # recovery can always restart from the retained backups.
                _regular(target).unlink()
        for index, entry in enumerate(receipt['entries']):
            source = directory / entry['source']
            if source.exists():
                if sha256(_regular(source)) != entry['source_sha256']:
                    raise TransactionConflict(f'Source changed during recovery: {source.name}')
                continue
            pending = batch / 'staging' / f'recover-{index:06d}.bin'
            pending.parent.mkdir(exist_ok=True)
            shutil.copyfile(_regular(batch / entry['backup']), pending)
            if sha256(pending) != entry['source_sha256']:
                raise TransactionConflict('Recovery copy verification failed')
            _exclusive_publish(pending, source)
        receipt['state'] = 'rolled_back'
        _journal(receipt_path, receipt)
        if lock.exists() and lock.read_text(encoding='ascii') == receipt['id']: lock.unlink()
    except BaseException as error:
        receipt['state'] = 'rollback_failed'
        receipt['error'] = str(error)
        _journal(receipt_path, receipt)
        # Retain the lock and all backups for explicit recovery.
        raise
    return {**receipt, 'receipt_path': str(receipt_path)}


def execute_plan(input_path, changes, *, operation, dry_run=False, cancel_check=None, details=None):
    """changes: source name, target name/None, location input/discarded,
    optional payload bytes or ordinary payload_path for a streamed replacement.
    Both forms bind their exact bytes before staging; only one may be supplied.
    Payload omitted means an unchanged-byte rename.
    Every involved source must be a direct ordinary file in input_path.
    """
    directory = Path(input_path).absolute()
    if not directory.is_dir() or directory.resolve() != directory:
        raise TransactionConflict('Faceset must be an ordinary resolved directory')
    if not isinstance(changes, list) or len(changes) > 500000:
        raise TransactionConflict('Invalid batch change count')
    identifier = 'faceset-' + uuid.uuid4().hex
    history = directory.parent / (directory.name + ('_trash' if operation == 'sort' else '_history'))
    if history.exists() and (not history.is_dir() or history.resolve() != history):
        raise TransactionConflict('History directory contains a symlink or is not a directory')
    batch = history / identifier
    receipt_path = batch / 'receipt.json'
    receipt = {'schema': 1, 'id': identifier, 'input_name': directory.name,
               'operation': operation, 'state': 'preview', 'entries': [],
               'owner_pid': os.getpid(), 'owner_identity': _process_identity(os.getpid())}
    if details is not None:
        encoded = json.dumps(details, allow_nan=False)
        if len(encoded.encode('utf-8')) > 1024 * 1024:
            raise TransactionConflict('Transaction details exceed the supported size')
        receipt['details'] = json.loads(encoded)
    source_names = set()
    target_names = set()
    payloads = []
    for index, change in enumerate(changes):
        source = _plain_name(change['source'])
        if source.casefold() in source_names: raise TransactionConflict('Duplicate batch source')
        source_names.add(source.casefold())
        source_path = _regular(directory / source)
        payload = change.get('payload')
        if payload is not None and not isinstance(payload, bytes):
            raise TransactionConflict('Replacement payload must be bytes')
        payload_path = change.get('payload_path')
        if payload_path is not None:
            if payload is not None or not isinstance(payload_path, (str, os.PathLike)):
                raise TransactionConflict('Supply payload bytes or one ordinary payload path')
            payload_path = _regular(Path(payload_path).absolute())
        replacement_digest = (sha256(payload_path) if payload_path is not None else
                              hashlib.sha256(payload).hexdigest() if payload is not None else None)
        expected_payload = change.get('payload_sha256')
        if expected_payload is not None and (not isinstance(expected_payload, str)
                or not re.fullmatch(r'[0-9a-f]{64}', expected_payload)
                or replacement_digest != expected_payload):
            raise TransactionConflict('Replacement payload differs from its expected SHA256')
        target = change.get('target')
        location = change.get('location', 'input')
        if target is not None:
            _plain_name(target)
            if location not in ('input', 'discarded'): raise TransactionConflict('Invalid batch target')
            target_key = (location, target.casefold())
            if target_key in target_names: raise TransactionConflict('Duplicate batch target')
            target_names.add(target_key)
        digest = sha256(source_path)
        expected_source = change.get('source_sha256')
        if expected_source is not None and (not isinstance(expected_source, str)
                or not re.fullmatch(r'[0-9a-f]{64}', expected_source) or digest != expected_source):
            raise TransactionConflict('Source differs from its expected SHA256')
        receipt['entries'].append({'source': source, 'source_sha256': digest,
                                  'target': target, 'location': location,
                                  'target_sha256': replacement_digest if replacement_digest is not None else digest,
                                  'backup': f'originals/{index:06d}.bin', 'published': False})
        payloads.append((payload, payload_path))
    # Include non-participating files, case-insensitively even on a Unix QA
    # host, because production filesystems and historical names are Windows.
    existing = {path.name.casefold(): path for path in directory.iterdir()}
    for location, target in target_names:
        if location == 'input' and target in existing and target not in source_names:
            raise TransactionConflict(f'Target collides with an untouched file: {existing[target].name}')
    if dry_run or not changes:
        return {**receipt, 'dry_run': dry_run, 'receipt_path': None}
    lock = directory / '.faceset-transaction.lock'
    with lock.open('x', encoding='ascii') as stream: stream.write(identifier)
    mutated = False
    try:
        history.mkdir(exist_ok=True)
        batch.mkdir()
        (batch / 'originals').mkdir()
        (batch / 'staging').mkdir()
        (batch / 'discarded').mkdir()
        receipt['state'] = 'preparing'; _journal(receipt_path, receipt)
        for index, (entry, (payload, payload_path)) in enumerate(zip(receipt['entries'], payloads)):
            if cancel_check and cancel_check(): raise InterruptedError('Faceset transaction cancelled')
            source = _regular(directory / entry['source'])
            if sha256(source) != entry['source_sha256']:
                raise TransactionConflict(f'Source changed after preview: {source.name}')
            backup = batch / entry['backup']
            shutil.copyfile(source, backup)
            if sha256(backup) != entry['source_sha256']:
                raise TransactionConflict('Original backup verification failed')
            if entry['target'] is not None:
                staged = batch / 'staging' / f'{index:06d}.bin'
                if payload_path is not None:
                    if sha256(_regular(payload_path)) != entry['target_sha256']:
                        raise TransactionConflict('Replacement payload changed after preview')
                    shutil.copyfile(payload_path, staged)
                    if sha256(_regular(payload_path)) != entry['target_sha256']:
                        raise TransactionConflict('Replacement payload changed while staging')
                elif payload is None: shutil.copyfile(source, staged)
                else: staged.write_bytes(payload)
                if sha256(staged) != entry['target_sha256']:
                    raise TransactionConflict('Staged output verification failed')
        receipt['state'] = 'prepared'; _journal(receipt_path, receipt)
        # Check the full batch again before the first source mutation.
        for index, (entry, (_payload, payload_path)) in enumerate(zip(receipt['entries'], payloads)):
            if sha256(_regular(directory / entry['source'])) != entry['source_sha256']:
                raise TransactionConflict('Source changed before batch commit')
            if entry['target'] is not None:
                staged = _regular(batch / 'staging' / f'{index:06d}.bin')
                if sha256(staged) != entry['target_sha256']:
                    raise TransactionConflict('Staged output changed before batch commit')
                if payload_path is not None and sha256(_regular(payload_path)) != entry['target_sha256']:
                    raise TransactionConflict('Replacement payload changed before batch commit')
        if cancel_check and cancel_check(): raise InterruptedError('Faceset transaction cancelled')
        receipt['state'] = 'removing'; _journal(receipt_path, receipt)
        for entry in receipt['entries']:
            source = _regular(directory / entry['source'])
            if sha256(source) != entry['source_sha256']:
                raise TransactionConflict('Source changed during batch commit')
            source.unlink(); mutated = True
        receipt['state'] = 'publishing'; _journal(receipt_path, receipt)
        for index, entry in enumerate(receipt['entries']):
            if cancel_check and cancel_check(): raise InterruptedError('Faceset transaction cancelled')
            target = _target(directory, batch, entry)
            if target is not None:
                staged = _regular(batch / 'staging' / f'{index:06d}.bin')
                if sha256(staged) != entry['target_sha256']:
                    raise TransactionConflict('Staged output changed before publication')
                _exclusive_publish(staged, target)
                if sha256(_regular(target)) != entry['target_sha256']:
                    raise TransactionConflict('Published output verification failed')
                entry['published'] = True
                _journal(receipt_path, receipt)
        receipt['state'] = 'committed'; _journal(receipt_path, receipt)
    except BaseException as error:
        receipt['error'] = str(error)
        if mutated:
            _journal(receipt_path, receipt)
            try:
                recover_transaction(receipt_path)
            except BaseException as recovery_error:
                raise TransactionConflict(f'Batch failed; originals retained at {receipt_path}; recovery failed: {recovery_error}') from error
        else:
            receipt['state'] = 'failed_before_commit'
            if batch.exists(): _journal(receipt_path, receipt)
        raise
    finally:
        if (lock.exists() and lock.read_text(encoding='ascii') == identifier
                and receipt['state'] != 'rollback_failed'):
            # If recovery failed it updated its own receipt; read that state
            # before releasing this process's mutation lock.
            saved_state = json.loads(receipt_path.read_text(encoding='utf-8'))['state'] if receipt_path.exists() else receipt['state']
            if saved_state != 'rollback_failed': lock.unlink()
    return {**receipt, 'receipt_path': str(receipt_path), 'archive_path': str(batch)}
