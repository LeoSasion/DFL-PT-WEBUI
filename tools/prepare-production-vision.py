"""Install pinned project resources, verify or build a separate review resource pack.

Never writes a workspace. Existing destinations are immutable; differing files
fail. Public release admission is independent of local installation permission.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(root, relative):
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


def install(project, manifest, groups, cache, allow_download):
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=group['id'] + '.staging-', dir=destination.parent))
        try:
            for record in group['files']:
                output = safe(staging, record['path'])
                output.parent.mkdir(parents=True, exist_ok=True)
                if 'content' in record:
                    output.write_bytes(record['content'].encode('utf-8'))
                else:
                    source = safe(cache, record['sourceCache']) if cache and record.get('sourceCache') else None
                    if source and check(source, record):
                        shutil.copyfile(source, output)
                    elif record.get('repositoryFile'):
                        source = safe(ROOT, record['repositoryFile'])
                        if not check(source, record):
                            raise ValueError('Checked-in resource/license differs: ' + record['repositoryFile'])
                        shutil.copyfile(source, output)
                    elif allow_download and record.get('url'):
                        with urllib.request.urlopen(record['url'], timeout=60) as response, output.open('xb') as stream:
                            shutil.copyfileobj(response, stream)
                    else:
                        raise ValueError('Offline resource missing: ' + group['id'] + '/' + record['path'])
                if not check(output, record):
                    raise ValueError('Resource checksum differs: ' + group['id'] + '/' + record['path'])
            staging.rename(destination)
            results.append({'id': group['id'], 'status': 'installed', 'fileCount': len(group['files'])})
        except Exception:
            # A failed batch is diagnostic evidence; it never becomes active.
            raise
    print(json.dumps({'ok': True, 'resources': results}))


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
        record = json.loads(archive.read('RESOURCE-PACK.json'))
        expected = {r['path']: r for r in record['files']}
        if manifest is not None:
            if record.get('manifestSha256') != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest():
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
    verify_pack(pack_path, manifest)
    with zipfile.ZipFile(pack_path) as archive:
        record = json.loads(archive.read('RESOURCE-PACK.json'))
        if not {g['id'] for g in groups} <= set(record['groups']):
            raise ValueError('Resource pack does not contain the requested profile')
        for group in groups:
            destination = safe(project, group['installRoot'])
            if destination.exists():
                if not all(check(safe(destination, r['path']), r) for r in group['files']):
                    raise ValueError('Existing resource group differs; preserved: ' + str(destination))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=group['id'] + '.staging-', dir=destination.parent))
            for item in group['files']:
                target = safe(staging, item['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(group['installRoot'] + '/' + item['path']) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                if not check(target, item):
                    raise ValueError('Resource changed during extraction')
            staging.rename(destination)
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
