"""Recoverable publication of the fixed-name DFM bundle.

The journal is the pending marker for readers. A filesystem cannot atomically
replace several fixed filenames, so readers must reject a bundle while it is
present. Backups survive process termination and are never consumed by rollback.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid
from core.file_hash import file_sha256


class PublicationRecoveryError(RuntimeError):
    """Keep the journal/backups and retry after resolving a file conflict."""


def transaction_journal_path(output):
    output = Path(output)
    return output.with_name(output.name + '.publish.json')


def _plain(path, *, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise PublicationRecoveryError('Publication path is not a plain local file/directory: ' + str(path))
    return info


def _fingerprint(path):
    try:
        info = _plain(path)
    except FileNotFoundError:
        return None
    return {'bytes': info.st_size, 'sha256': file_sha256(path)}


def _sync_directory(directory):
    # Windows does not expose directory fsync through Python. Each copied file
    # and journal is flushed before its atomic rename; POSIX also syncs entries.
    if os.name != 'nt':
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _copy_durable(source, target):
    _plain(source)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 16 * 1024**2)
        outgoing.flush()
        os.fsync(outgoing.fileno())


@contextmanager
def _publication_lock(output):
    lock = output.with_name(output.name + '.publish.lock')
    try:
        _plain(lock)
    except FileNotFoundError:
        pass
    # Keep the lock file: unlinking it could let another process lock a different
    # inode. The OS releases the byte lock automatically when a process exits.
    with lock.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise PublicationRecoveryError('Another DFM publication is active: ' + str(output)) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _allowed_name(output, name):
    fixed = {output.name, output.name + '.export.json', output.name + '.engine',
             output.name + '.layers.json'}
    return (isinstance(name, str) and (name in fixed or
            re.fullmatch(re.escape(output.name) + r'\.[0-9a-f]{32}\.data', name) is not None))


def _valid_fingerprint(value):
    return (isinstance(value, dict) and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and value['bytes'] >= 0
            and isinstance(value['sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None)


def _read_journal(output):
    journal = transaction_journal_path(output)
    try:
        info = _plain(journal)
    except FileNotFoundError:
        return None
    try:
        if info.st_size > 1024**2:
            raise ValueError('oversized journal')
        record = json.loads(journal.read_text(encoding='utf-8'))
        if (record['version'] != 1 or record['output'] != output.name
                or record['state'] not in ('prepared', 'committed', 'rolled_back')
                or not re.fullmatch(r'\.dfm-publish-[0-9a-f]{32}', record['backup_directory'])):
            raise ValueError('invalid transaction header')
        entries = record['files']
        if not isinstance(entries, list) or not 2 <= len(entries) <= 1024:
            raise ValueError('invalid transaction files')
        names = [entry['name'] for entry in entries]
        if (not all(isinstance(name, str) for name in names)
                or len({name.casefold() for name in names}) != len(names)
                or output.name not in names or names[-1] != output.name + '.export.json'):
            raise ValueError('invalid transaction names')
        for entry in entries:
            if (not _allowed_name(output, entry['name']) or not _valid_fingerprint(entry['new'])
                    or not (entry['old'] is None or _valid_fingerprint(entry['old']))):
                raise ValueError('invalid transaction entry')
        try:
            _plain(output.parent / record['backup_directory'], directory=True)
        except FileNotFoundError:
            if record['state'] == 'prepared':
                raise
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise PublicationRecoveryError('Invalid DFM publication journal; no files changed: ' + str(journal)) from error
    return record


def _write_journal(output, record):
    journal = transaction_journal_path(output)
    backup = output.parent / record['backup_directory']
    pending = backup / 'journal.next'
    if pending.exists():
        _plain(pending)
        pending.unlink()
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, journal)
    _sync_directory(output.parent)


def _cleanup(output, record):
    backup = output.parent / record['backup_directory']
    try:
        _plain(backup, directory=True)
    except FileNotFoundError:
        transaction_journal_path(output).unlink()
        _sync_directory(output.parent)
        return
    allowed = {'journal.next'} | {f'{kind}-{index}' for index in range(len(record['files']))
                                for kind in ('backup', 'restore')}
    children = list(backup.iterdir())
    # Never recursively delete a directory, or remove files not owned by this
    # transaction. A conflicting file leaves a visible pending marker for repair.
    if any(child.name not in allowed for child in children):
        raise PublicationRecoveryError('Unexpected file in DFM recovery directory: ' + str(backup))
    for child in children:
        _plain(child)
        if child.name.startswith(('backup-', 'restore-')):
            index = int(child.name.split('-')[1])
            if _fingerprint(child) != record['files'][index]['old']:
                raise PublicationRecoveryError('DFM recovery file changed before cleanup: ' + str(child))
    for child in children:
        child.unlink()
    # Remove the marker last. An interrupted cleanup can resume with fewer backup
    # files because committed/rolled_back journals no longer require rollback.
    backup.rmdir()
    transaction_journal_path(output).unlink()
    _sync_directory(output.parent)


def _recover_locked(output):
    record = _read_journal(output)
    if record is None:
        return False
    entries = record['files']
    backup = output.parent / record['backup_directory']
    if record['state'] == 'prepared':
        # Validate the complete recovery plan before changing any public file.
        for index, entry in enumerate(entries):
            current = _fingerprint(output.parent / entry['name'])
            if current not in (entry['old'], entry['new']):
                raise PublicationRecoveryError('DFM publication target changed externally: ' + entry['name'])
            if entry['old'] is not None and _fingerprint(backup / f'backup-{index}') != entry['old']:
                raise PublicationRecoveryError('DFM publication backup is missing or changed: ' + entry['name'])
        for index in reversed(range(len(entries))):
            entry = entries[index]
            target = output.parent / entry['name']
            current = _fingerprint(target)
            if current == entry['old']:
                continue
            if current != entry['new']:
                raise PublicationRecoveryError('DFM publication target changed during recovery: ' + entry['name'])
            if entry['old'] is None:
                target.unlink()
            else:
                restore = backup / f'restore-{index}'
                if restore.exists():
                    _plain(restore)
                    restore.unlink()
                _copy_durable(backup / f'backup-{index}', restore)
                if _fingerprint(restore) != entry['old']:
                    raise PublicationRecoveryError('DFM backup copy failed verification: ' + entry['name'])
                if _fingerprint(target) != entry['new']:
                    raise PublicationRecoveryError('DFM publication target changed during backup copy: ' + entry['name'])
                os.replace(restore, target)
        _sync_directory(output.parent)
        record['state'] = 'rolled_back'
        _write_journal(output, record)
    expected = 'new' if record['state'] == 'committed' else 'old'
    if any(_fingerprint(output.parent / entry['name']) != entry[expected] for entry in entries):
        raise PublicationRecoveryError('DFM bundle changed before publication cleanup; recovery files retained')
    _cleanup(output, record)
    return True


def recover_publication(output):
    """Recover an interrupted publication, or fail without overwriting conflicts."""
    output = Path(output).absolute()
    with _publication_lock(output):
        return _recover_locked(output)


def publish_bundle(output, files):
    """Publish validated sibling staging files, keeping the receipt last.

    Ordinary failures roll back immediately. Process interruption leaves a
    pending journal that recover_publication (also called by the next export)
    can safely replay. Committed bundles survive interrupted cleanup.
    """
    output = Path(output).absolute()
    files = [Path(source) for source in files]
    names = [source.name for source in files]
    if (len({name.casefold() for name in names}) != len(names)
            or not all(_allowed_name(output, name) for name in names)
            or output.name not in names or not names or names[-1] != output.name + '.export.json'):
        raise ValueError('Invalid DFM publication bundle names/order')
    with _publication_lock(output):
        _recover_locked(output)
        entries = []
        for source in files:
            target = output.parent / source.name
            if source.absolute() == target:
                raise ValueError('DFM publication requires separate staging files')
            new, old = _fingerprint(source), _fingerprint(target)
            if new is None:
                raise FileNotFoundError(source)
            # Windows FlushFileBuffers requires a writable handle. These are
            # private staged outputs, opened without changing their bytes.
            with source.open('r+b') as stream:
                os.fsync(stream.fileno())
            if source.name.endswith('.data') and old is not None:
                raise PublicationRecoveryError('Unique DFM sidecar already exists: ' + source.name)
            entries.append({'name': source.name, 'old': old, 'new': new})
        backup = output.parent / ('.dfm-publish-' + uuid.uuid4().hex)
        backup.mkdir()
        record = {'version': 1, 'output': output.name, 'state': 'prepared',
                  'backup_directory': backup.name, 'files': entries}
        try:
            for index, entry in enumerate(entries):
                if entry['old'] is not None:
                    _copy_durable(output.parent / entry['name'], backup / f'backup-{index}')
                    if _fingerprint(backup / f'backup-{index}') != entry['old']:
                        raise PublicationRecoveryError('Existing DFM bundle changed during backup')
            _sync_directory(backup)
            _write_journal(output, record)
            for source, entry in zip(files, entries):
                target = output.parent / entry['name']
                if _fingerprint(target) != entry['old'] or _fingerprint(source) != entry['new']:
                    raise PublicationRecoveryError('DFM publication input changed before replacement')
                os.replace(source, target)
            if any(_fingerprint(output.parent / entry['name']) != entry['new'] for entry in entries):
                raise PublicationRecoveryError('DFM bundle changed during publication')
            _sync_directory(output.parent)
            record['state'] = 'committed'
            _write_journal(output, record)
        except Exception as error:
            if transaction_journal_path(output).exists():
                try:
                    _recover_locked(output)
                except Exception as recovery_error:
                    raise PublicationRecoveryError(
                        'DFM publication failed and recovery is pending; close file users and retry export. '
                        'Journal: ' + str(transaction_journal_path(output)) + '; ' + str(recovery_error)
                    ) from error
            else:
                # No journal means no public target has been replaced. Clean only
                # the backup files that this invocation created.
                for index in range(len(entries)):
                    candidate = backup / f'backup-{index}'
                    if candidate.exists():
                        _plain(candidate)
                        candidate.unlink()
                pending = backup / 'journal.next'
                if pending.exists():
                    _plain(pending)
                    pending.unlink()
                backup.rmdir()
            raise
        _cleanup(output, record)
