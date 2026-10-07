"""Windows CPython 3.12 dependency graph, hash locks, wheelhouse and Python SBOM.

Generate is an explicit maintainer action against the tested local interpreter.
Install never resolves dependencies: every selected wheel must match the lock.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILES = {'production': 'requirements.txt', 'restoration': 'requirements-restoration.txt',
                 'validation': 'requirements-validation.txt', 'evaluation': 'tools/requirements-vision-evaluation.txt',
                 'scene': 'requirements-scene.txt'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def direct(path, visited=None):
    visited = set() if visited is None else visited
    path = Path(path).resolve()
    if path in visited:
        return []
    visited.add(path)
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith('-r '):
            result.extend(direct(path.parent / line[3:].strip(), visited))
        elif line and not line.startswith(('#', '--')):
            result.append(line)
    return result


def graph(profile):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    result, queue = {}, ['torch==2.9.1+cu128'] + direct(ROOT / PROFILE_FILES[profile])
    while queue:
        requirement = Requirement(queue.pop())
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        key = canonicalize_name(requirement.name)
        distribution = metadata.distribution(requirement.name)
        if distribution.version not in requirement.specifier:
            raise ValueError(f'Tested installation does not satisfy {requirement}')
        if key in result:
            continue
        result[key] = {'name': distribution.metadata['Name'], 'version': distribution.version,
                       'requires': list(distribution.requires or []),
                       'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License', 'UNSPECIFIED'),
                       'homepage': distribution.metadata.get('Home-page'),
                       'profiles': [profile]}
        queue.extend(distribution.requires or [])
    if 'h5py' in result or ('onnxruntime' in result and profile in ('production', 'restoration')):
        raise ValueError('Production dependency graph unexpectedly contains a validation-only package')
    cv = [name for name in result if name.startswith('opencv-python')]
    if cv != ['opencv-python']:
        raise ValueError('Desktop runtime must contain exactly the GUI OpenCV wheel')
    return result


def run(args):
    if importlib.util.find_spec('pip') is not None:
        command = [sys.executable, '-I', '-m', 'pip', '--isolated', *args]
    else:
        # Production portable environments omit installer distributions.
        # Python's pinned base carries ensurepip's official wheel: use it
        # only in the isolated child, preserving the environment's pip state.
        code = ("import ensurepip,pathlib,runpy,sys; "
                "wheel=pathlib.Path(ensurepip.__file__).parent/'_bundled'/('pip-'+ensurepip.version()+'-py3-none-any.whl'); "
                "assert wheel.is_file(),'Project-local bundled pip wheel is missing'; "
                "sys.path.insert(0,str(wheel)); sys.argv=['pip','--isolated']+sys.argv[1:]; "
                "runpy.run_module('pip',run_name='__main__')")
        command = [sys.executable, '-I', '-c', code, *args]
    subprocess.run(command, check=True)


def generate(destination, download, selected_profiles=None):
    from packaging.utils import parse_wheel_filename, canonicalize_name
    destination.mkdir(parents=True, exist_ok=True)
    profiles = {profile: graph(profile) for profile in (selected_profiles or PROFILE_FILES)}
    existing_sbom = ROOT / 'release/python-sbom.json'
    if selected_profiles and existing_sbom.exists():
        existing = json.loads(existing_sbom.read_text(encoding='utf-8'))
        for profile, names in existing['profiles'].items():
            if profile not in profiles:
                profiles[profile] = {name: existing['packages'][name] for name in names}
    combined = {}
    for profile, records in profiles.items():
        for key, record in records.items():
            if key not in combined:
                combined[key] = {**record, 'profiles': []}
            combined[key]['profiles'].append(profile)
    wheelhouse = destination / 'wheelhouse'
    wheelhouse.mkdir(exist_ok=True)
    if download:
        for cuda in ('torch', 'torchvision'):
            record = combined[cuda]
            run(['download', '--no-deps', '--only-binary=:all:', '--index-url', 'https://download.pytorch.org/whl/cu128',
                 '--dest', str(wheelhouse), record['name'] + '==' + record['version']])
        ordinary = destination / 'resolved-maintainer-constraints.txt'
        ordinary.write_text('\n'.join(r['name'] + '==' + r['version'] for k, r in sorted(combined.items()) if k not in ('torch', 'torchvision')) + '\n', encoding='utf-8')
        # Build old pure Python source distributions into pinned, locally hashed
        # wheels using the already tested build tooling; consumers never build.
        run(['wheel', '--no-deps', '--no-build-isolation', '--index-url', 'https://pypi.org/simple',
             '--wheel-dir', str(wheelhouse), '-r', str(ordinary)])
    wheels = {}
    for path in wheelhouse.glob('*.whl'):
        name, version, _, tags = parse_wheel_filename(path.name)
        if name in wheels:
            raise ValueError('Duplicate wheel distribution: ' + name)
        wheels[name] = {'file': path.name, 'version': str(version), 'sha256': digest(path),
                        'sizeBytes': path.stat().st_size, 'tags': sorted(str(tag) for tag in tags)}
    missing = sorted(set(combined) - set(wheels))
    if missing:
        raise ValueError('Wheelhouse is incomplete: ' + ', '.join(missing))
    for key, record in combined.items():
        if wheels[key]['version'] != record['version']:
            raise ValueError('Wheel version differs from tested package: ' + key)
        record['wheel'] = wheels[key]
    ROOT.joinpath('release/python-locks').mkdir(exist_ok=True)
    for profile, records in profiles.items():
        path = ROOT / f'release/python-locks/{profile}-win-cp312.txt'
        path.write_text('# Generated from the tested Windows CPython 3.12 graph. No network resolution at installation.\n'
                        + '\n'.join(combined[key]['name'] + '==' + combined[key]['version'] + ' --hash=sha256:' + wheels[key]['sha256'] for key in sorted(records)) + '\n', encoding='utf-8')
    payload = {'schemaVersion': 1, 'target': 'win_amd64-cp312', 'python': platform.python_version(),
               'profiles': {key: sorted(value) for key, value in profiles.items()}, 'packages': combined}
    (ROOT / 'release/python-sbom.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (wheelhouse / 'python-sbom.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'profiles': {k: len(v) for k, v in profiles.items()}, 'wheels': len(wheels), 'wheelhouse': str(wheelhouse)}))


def verify(wheelhouse, profile):
    payload = json.loads((ROOT / 'release/python-sbom.json').read_text(encoding='utf-8'))
    if payload.get('schemaVersion') != 1 or payload.get('target') != 'win_amd64-cp312':
        raise ValueError('Unsupported runtime lock target')
    if sys.version_info[:2] != (3, 12) or sys.platform != 'win32':
        raise ValueError('Locked wheelhouse requires Windows x64 CPython 3.12')
    for key in payload['profiles'][profile]:
        item = payload['packages'][key]['wheel']
        path = wheelhouse / item['file']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item['sizeBytes'] or digest(path) != item['sha256']:
            raise ValueError('Missing or changed locked wheel: ' + item['file'])
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('generate', 'verify', 'install'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--profile', choices=tuple(PROFILE_FILES), default='production')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--profiles', nargs='+', choices=tuple(PROFILE_FILES))
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args.directory.resolve(), args.download, args.profiles)
    else:
        verify(args.directory.resolve(), args.profile)
        if args.command == 'install':
            run(['install', '--no-index', '--find-links', str(args.directory.resolve()), '--require-hashes', '--only-binary=:all:',
                 '-r', str(ROOT / f'release/python-locks/{args.profile}-win-cp312.txt')])
            run(['check'])
        print('Locked wheelhouse verified: ' + args.profile)
