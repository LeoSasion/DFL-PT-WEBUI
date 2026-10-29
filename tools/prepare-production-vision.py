"""Install pinned project resources, verify or build a separate review resource pack.

Never writes a workspace. Existing destinations are immutable; differing files
fail. Public release admission is independent of local installation permission.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile
import os
import uuid
import io
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(root, relative):
    if Path(root).is_symlink() or (hasattr(Path(root), 'is_junction') and Path(root).is_junction()):
        raise ValueError('Linked resource root is unsupported')
    root = Path(root).resolve()
    item = PurePosixPath(relative)
    if item.is_absolute() or not item.parts or any(p in ('.', '..') for p in relative.split('/')) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe resource path: ' + relative)
    target = root.joinpath(*item.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError('Resource escaped installation root')
    for ancestor in (target, *target.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, 'is_junction') and ancestor.is_junction()):
            raise ValueError('Linked resource paths are unsupported')
    return target


def check(path, record):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == record['sizeBytes'] and digest(path) == record['sha256']


def selected(manifest, profiles):
    return [group for group in manifest['groups'] if group['profile'] in profiles]


def write_resource_report(project, report):
    path = safe(project, '.launcher-install/vision/resources.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('resources.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def copy_owned_resource(source, target, written, name):
    value = hashlib.sha256()
    with target.open('xb') as output:
        try:
            while block := source.read(1024 * 1024):
                output.write(block)
                value.update(block)
        finally:
            written[name] = value.hexdigest()


@contextmanager
def resource_staging(destination, group_id):
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=group_id + '.staging-', dir=destination.parent))
    written = {}
    try:
        yield staging, written
    finally:
        if staging.exists() and not staging.is_symlink() and not staging.is_junction():
            for name, value in written.items():
                try: path = safe(staging, name)
                except ValueError: continue
                if path.is_file() and digest(path) == value: path.unlink()
            # Walk only plain directories. Unknown files/links and changed
            # bytes are retained; cleanup never follows a replacement junction.
            directories = []; pending = [staging]
            while pending:
                parent = pending.pop(); directories.append(parent)
                for path in parent.iterdir():
                    if path.is_dir() and not path.is_symlink() and not path.is_junction(): pending.append(path)
            for path in reversed(directories):
                try: path.rmdir()
                except OSError: pass


def cached_resource(project, group, record, cache, allow_download):
    managed = safe(project, '.launcher-install/vision/files/' + record['sha256'])
    manual = safe(project, '.launcher-install/vision/manual/' + group['id'] + '/' + record['path'])
    legacy = safe(cache, record['sourceCache']) if cache and record.get('sourceCache') else None
    for candidate in [manual, legacy, managed]:
        if candidate and check(candidate, record):
            return candidate
    if manual.exists():
        raise ValueError('Manual file checksum differs; preserved: ' + str(manual))
    if not allow_download or not record.get('url'):
        raise ValueError('Offline resource missing: ' + group['id'] + '/' + record['path'])
    managed.parent.mkdir(parents=True, exist_ok=True)
    partial = managed.with_name(managed.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with urllib.request.urlopen(record['url'], timeout=30) as response, partial.open('xb') as output:
            count = 0
            while True:
                block = response.read(min(1024 * 1024, record['sizeBytes'] + 1 - count))
                if not block:
                    break
                count += len(block)
                if count > record['sizeBytes']:
                    raise ValueError('Resource download exceeds pinned size')
                output.write(block)
        if record.get('textLineEndings') == 'crlf':
            # Reviewed source inventories include Windows checkout line endings.
            # Normalize only explicitly declared UTF-8 source/license files;
            # the resulting bytes must still match the original pinned hash.
            text = partial.read_text(encoding='utf-8')
            partial.write_bytes(text.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))
        if not check(partial, record):
            raise ValueError('Resource checksum differs: ' + group['id'] + '/' + record['path'])
        os.replace(partial, managed)
        return managed
    finally:
        if partial.exists():
            partial.unlink()


def install(project, manifest, groups, cache, allow_download):
    report = {'schemaVersion': 1, 'complete': False, 'resources': [], 'failed': []}
    def save():
        write_resource_report(project, report)
    save()
    results = []
    for group in groups:
        if group.get('localInstallationAllowed') is not True:
            raise ValueError('Local resource installation is not admitted: ' + group['id'])
        destination = safe(project, group['installRoot'])
        if destination.exists():
            if not all(check(safe(destination, r['path']), r) for r in group['files']):
                raise ValueError('Existing resource group differs; preserved: ' + str(destination))
            results.append({'id': group['id'], 'status': 'verified-existing'})
            continue
        with resource_staging(destination, group['id']) as (staging, written):
            for record in group['files']:
                output = safe(staging, record['path'])
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if 'content' in record:
                        copy_owned_resource(io.BytesIO(record['content'].encode('utf-8')), output, written, record['path'])
                    elif record.get('repositoryFile'):
                        source = safe(ROOT, record['repositoryFile'])
                        if not check(source, record):
                            raise ValueError('Checked-in resource/license differs: ' + record['repositoryFile'])
                        with source.open('rb') as stream: copy_owned_resource(stream, output, written, record['path'])
                    else:
                        source = cached_resource(project, group, record, cache, allow_download)
                        with source.open('rb') as stream: copy_owned_resource(stream, output, written, record['path'])
                    if not check(output, record):
                        raise ValueError('Resource checksum differs: ' + group['id'] + '/' + record['path'])
                except Exception as error:
                    report['failed'].append({'group': group['id'], 'path': record['path'], 'error': str(error),
                        'manual': bool(record.get('manualDownload')) or not bool(record.get('url') or record.get('repositoryFile') or 'content' in record),
                        'url': group.get('source', {}).get('official_weight_url', group.get('source', {}).get('weightUrl', '')),
                        'savePath': '.launcher-install/vision/manual/' + group['id'] + '/' + record['path'],
                        'sha256': record['sha256'], 'sizeBytes': record['sizeBytes']})
                    save()
            if all(check(safe(staging, r['path']), r) for r in group['files']):
                staging.rename(destination)
                results.append({'id': group['id'], 'status': 'installed', 'fileCount': len(group['files'])})
    report.update(resources=results, complete=not report['failed'])
    save()
    print(json.dumps({'ok': report['complete'], **report}, ensure_ascii=False))
    if report['failed']:
        raise ValueError('Offline resource missing or download failed; see .launcher-install/vision/resources.json. Supply the listed manual files and retry; verified downloads are cached.')


def pack(project, manifest, groups, output, public):
    if public and any(group.get('publicDistributionAllowed') is not True for group in groups):
        raise ValueError('Public distribution requires separate admission for every selected model/source/weight license')
    if output.exists():
        raise ValueError('Use a fresh resource-pack destination')
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for group in groups:
            for record in group['files']:
                source = safe(safe(project, group['installRoot']), record['path'])
                if not check(source, record):
                    raise ValueError('Resource must be verified before packing: ' + str(source))
                name = group['installRoot'] + '/' + record['path']
                archive.write(source, name)
                inventory.append({'path': name, 'sha256': record['sha256'], 'sizeBytes': record['sizeBytes']})
        archive.writestr('RESOURCE-PACK.json', json.dumps({'schemaVersion': 1, 'publicDistribution': public,
            'manifestSha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'groups': [g['id'] for g in groups], 'files': inventory}, indent=2))
    verify_pack(output)


def verify_pack(path, manifest=None):
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo('RESOURCE-PACK.json').file_size > 4 * 1024 * 1024:
            raise ValueError('Resource pack metadata exceeds limit')
        record = json.loads(archive.read('RESOURCE-PACK.json'))
        if (not isinstance(record, dict) or type(record.get('schemaVersion')) is not int
                or record['schemaVersion'] != 1 or type(record.get('publicDistribution')) is not bool
                or not isinstance(record.get('groups'), list) or not record['groups']
                or not all(isinstance(g, str) for g in record['groups']) or len(set(record['groups'])) != len(record['groups'])
                or not isinstance(record.get('files'), list) or not 0 < len(record['files']) <= 50000):
            raise ValueError('Unsupported or invalid resource pack schema')
        for item in record['files']:
            if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                    or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in item['sha256'])
                    or type(item.get('sizeBytes')) is not int or item['sizeBytes'] < 0):
                raise ValueError('Invalid resource pack file record')
        expected = {r['path']: r for r in record['files']}
        if manifest is not None:
            if manifest.get('schemaVersion') != 1:
                raise ValueError('Resource pack differs from the fixed project manifest')
            groups = [g for g in manifest['groups'] if g['id'] in record['groups']]
            pinned = {g['installRoot'] + '/' + r['path']: {'path': g['installRoot'] + '/' + r['path'],
                'sha256': r['sha256'], 'sizeBytes': r['sizeBytes']} for g in groups for r in g['files']}
            if pinned != expected or len(groups) != len(record['groups']):
                raise ValueError('Resource pack does not match the admitted model groups')
        names = archive.namelist()
        if len(expected) != len(record['files']) or len(names) != len(expected) + 1 or len({n.casefold() for n in names}) != len(names) or set(names) != set(expected) | {'RESOURCE-PACK.json'}:
            raise ValueError('Resource pack inventory mismatch')
        for name, item in expected.items():
            safe(ROOT, name)
            if archive.getinfo(name).file_size != item['sizeBytes']:
                raise ValueError('Resource pack file size mismatch: ' + name)
            with archive.open(name) as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            if value != item['sha256'] or archive.getinfo(name).file_size != item['sizeBytes']:
                raise ValueError('Resource pack file hash mismatch: ' + name)
    print(json.dumps({'ok': True, 'pack': str(path), 'sha256': digest(path), 'fileCount': len(expected),
                      'publicDistribution': record['publicDistribution']}))


def install_pack(project, manifest, groups, pack_path):
    if any(g.get('localInstallationAllowed') is not True for g in groups):
        raise ValueError('Local resource installation is not admitted')
    verify_pack(pack_path, manifest)
    report = {'schemaVersion': 1, 'complete': False, 'resources': [], 'failed': []}
    with zipfile.ZipFile(pack_path) as archive:
        record = json.loads(archive.read('RESOURCE-PACK.json'))
        if not {g['id'] for g in groups} <= set(record['groups']):
            raise ValueError('Resource pack does not contain the requested profile')
        write_resource_report(project, report)
        try:
            for group in groups:
                destination = safe(project, group['installRoot'])
                if destination.exists():
                    if not all(check(safe(destination, r['path']), r) for r in group['files']):
                        raise ValueError('Existing resource group differs; preserved: ' + str(destination))
                    report['resources'].append({'id': group['id'], 'status': 'verified-existing'})
                    continue
                with resource_staging(destination, group['id']) as (staging, written):
                    for item in group['files']:
                        target = safe(staging, item['path'])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(group['installRoot'] + '/' + item['path']) as source:
                            copy_owned_resource(source, target, written, item['path'])
                        if not check(target, item):
                            raise ValueError('Resource changed during extraction')
                    staging.rename(destination)
                report['resources'].append({'id': group['id'], 'status': 'installed', 'fileCount': len(group['files'])})
            report['complete'] = True
        except Exception:
            report['message'] = '辅助资源包安装未完成，请检查资源包和日志后重试；已验证文件保留。'
            raise
        finally:
            write_resource_report(project, report)
    print(json.dumps({'ok': True, 'installedGroups': [g['id'] for g in groups]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('install', 'install-pack', 'verify', 'pack', 'verify-pack'))
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--profiles', nargs='+', choices=('production', 'restoration', 'scene'), default=['production'])
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--public-distribution', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release/vision-assets.json').read_text(encoding='utf-8'))
    if manifest.get('schemaVersion') != 1:
        raise ValueError('Unsupported production resource manifest')
    groups = selected(manifest, args.profiles)
    if args.command == 'install':
        install(args.project_root.resolve(), manifest, groups, args.cache_root, args.allow_download)
    elif args.command == 'verify':
        for group in groups:
            destination = safe(args.project_root.resolve(), group['installRoot'])
            if not all(check(safe(destination, r['path']), r) for r in group['files']):
                raise ValueError('Missing or changed resources: ' + group['id'])
        print(json.dumps({'ok': True, 'verifiedGroups': [g['id'] for g in groups]}))
    elif args.command == 'install-pack':
        install_pack(args.project_root.resolve(), manifest, groups, args.output.resolve())
    elif args.command == 'pack':
        pack(args.project_root.resolve(), manifest, groups, args.output.resolve(), args.public_distribution)
    else:
        verify_pack(args.output.resolve(), manifest)
