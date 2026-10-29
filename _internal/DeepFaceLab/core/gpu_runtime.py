"""Official Windows Torch variants, hash locks and selected-device capability checks."""
import argparse
import base64
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def compiled_architectures(torch_module):
    # get_arch_list() returns [] without a CUDA driver, even for a valid CUDA wheel.
    query = getattr(torch_module._C, '_cuda_getArchFlags', None)
    flags = query() if query is not None else None
    return flags.split() if isinstance(flags, str) else list(torch_module.cuda.get_arch_list())


def architecture_supported(capability, architectures):
    major, minor = capability
    for value in architectures:
        match = re.fullmatch(r'(sm|compute)_(\d+)', value)
        if not match:
            continue
        number = int(match[2])
        target = (number // 10, number % 10)
        if match[1] == 'sm' and target[0] == major and target[1] <= minor:
            return True
        if match[1] == 'compute' and target <= (major, minor):
            return True
    return False


def validate_cuda_devices(torch_module, indexes):
    architectures = compiled_architectures(torch_module)
    if not architectures:
        raise RuntimeError('The installed PyTorch wheel has no identifiable CUDA kernels')
    for index in indexes:
        capability = torch_module.cuda.get_device_capability(index)
        cuda = tuple(int(part) for part in torch_module.version.cuda.split('.')[:2])
        stack_supported = not (capability[0] >= 10 and cuda < (12, 8))
        if capability[0] < 7:
            stack_supported = stack_supported and cuda < (13, 0) and 0 < torch_module.backends.cudnn.version() <= 91002
        if not stack_supported or not architecture_supported(capability, architectures):
            name = torch_module.cuda.get_device_name(index)
            variant = 'cu126 (GTX 10 / Pascal)' if capability[0] < 7 else 'cu128 (Blackwell)' if capability[0] >= 10 else 'a matching official CUDA build'
            raise RuntimeError(f'GPU {index} {name} (SM {capability[0]}.{capability[1]}) is not supported by '
                               f'PyTorch {torch_module.__version__}; select the official {variant} runtime. '
                               'Replacing cuDNN DLLs alone cannot add missing PyTorch kernels.')


def load_manifest(root):
    value = json.loads((Path(root) / 'release/torch-runtimes.json').read_text(encoding='utf-8'))
    if value.get('schemaVersion') != 1 or set(value.get('runtimes', {})) != {'cu126', 'cu128'}:
        raise ValueError('Unsupported official Torch runtime manifest')
    return value


def derive_sbom(base, manifest, runtime):
    """Overlay two complete wheel records; never reuse another CUDA build's hash."""
    spec = manifest['runtimes'][runtime]
    result = copy.deepcopy(base)
    if result.get('schemaVersion') != 1 or result.get('target') != 'win_amd64-cp312':
        raise ValueError('Expected the fixed Windows CPython 3.12 SBOM')
    for name in ['torch', 'torchvision']:
        original = result['packages'][name]
        item = copy.deepcopy(spec.get('packages', {}).get(name, original))
        expected = spec[name]
        wheel = item['wheel']
        if (item['version'] != expected or wheel['version'] != expected
                or not re.fullmatch(r'[0-9a-f]{64}', wheel['sha256'])
                or wheel['file'] != f'{name}-{expected}-cp312-cp312-win_amd64.whl'
                or wheel['sizeBytes'] <= 0 or 'cp312-cp312-win_amd64' not in wheel['tags']):
            raise ValueError('Invalid official wheel record for ' + name)
        item['profiles'] = original['profiles']
        result['packages'][name] = item
    result['torchRuntime'] = runtime
    result['cudaVersion'] = spec['cuda']
    return result


def runtime_sbom(root, runtime='cu128'):
    root = Path(root)
    base = json.loads((root / 'release/python-sbom.json').read_text(encoding='utf-8'))
    return derive_sbom(base, load_manifest(root), runtime)


def installed_runtime(root, manifest):
    site = Path(root) / '.venv/Lib/site-packages'
    distributions = [d for d in metadata.distributions(path=[str(site)])
                     if d.metadata.get('Name', '').lower() == 'torch']
    if not distributions:
        return None
    if len(distributions) != 1:
        raise ValueError('Multiple Torch installations found; the existing environment will not be replaced')
    version = distributions[0].version
    for name, spec in manifest['runtimes'].items():
        if version == spec['torch']:
            return name
    raise ValueError(f'Unsupported existing Torch {version}; the installer will not replace an existing environment')


def detect_capabilities():
    try:
        result = subprocess.run(['nvidia-smi.exe', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            return []
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows or any(re.fullmatch(r'\d+\.\d+', line) is None for line in rows):
            return []
        return [tuple(map(int, line.split('.'))) for line in rows]
    except (OSError, subprocess.TimeoutExpired):
        return []


def select_runtime(requested='auto', installed=None, capabilities=()):
    if requested not in {'auto', 'cu126', 'cu128'}:
        raise ValueError('Unknown CUDA runtime: ' + requested)
    if installed is not None:
        if installed not in {'cu126', 'cu128'}:
            raise ValueError('Unsupported existing CUDA runtime')
        if requested != 'auto' and requested != installed:
            raise ValueError(f'Existing {installed} environment will not be replaced with {requested}; '
                             'use a separate source installation for the other GPU family')
        return installed
    if requested != 'auto':
        return requested
    old = any(cc[0] < 7 for cc in capabilities)
    blackwell = any(cc[0] >= 10 for cc in capabilities)
    if old and blackwell:
        raise ValueError('Pascal and Blackwell require different official Torch builds; choose cu126 or cu128 '
                         'explicitly and select only GPUs supported by that environment')
    return 'cu126' if old else 'cu128'


def render_profile(root, runtime, profile, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).absolute()
    if not output_dir.is_relative_to(root) or output_dir.resolve() != output_dir:
        raise ValueError('Resolved runtime locks must stay inside the project without directory links')
    sbom = runtime_sbom(root, runtime)
    if profile not in sbom['profiles']:
        raise ValueError('Unknown functional runtime profile: ' + profile)
    if output_dir in {root / 'release', root / 'release/python-locks'}:
        raise ValueError('Resolved locks must not overwrite the source release baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / (profile + '-win-cp312.txt')
    rows = [f"{sbom['packages'][name]['name']}=={sbom['packages'][name]['version']} "
            f"--hash=sha256:{sbom['packages'][name]['wheel']['sha256']}" for name in sorted(sbom['profiles'][profile])]
    lock.write_text('# Resolved official ' + runtime + ' Windows CPython 3.12 profile.\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    sbom_path = output_dir / 'python-sbom.json'
    sbom_path.write_text(json.dumps(sbom, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    spec = load_manifest(root)['runtimes'][runtime]
    return dict(cudaRuntime=runtime, indexUrl=spec['indexUrl'], requirements=str(lock), sbom=str(sbom_path))


def runtime_package_prefix(root, environment=None):
    root = Path(root).resolve()
    prefix = Path(environment).absolute() if environment is not None else root / '.venv'
    managed = root / '.runtime/torch-variants'
    if (prefix.resolve() != prefix or
            (prefix != root / '.venv' and (prefix == managed or not prefix.is_relative_to(managed)))):
        raise ValueError('Torch packages must stay in the active venv or a managed runtime prefix without links')
    return prefix


def validate_torch_runtime(root, torch_module, torchvision_module, cuda_runtime='auto', package_environment=None):
    """Share version and binary rules with isolated, unactivated runtime probes."""
    torch, torchvision = torch_module, torchvision_module
    prefix = runtime_package_prefix(root, package_environment)
    for module in [torch, torchvision]:
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise ValueError('Runtime module is outside the project: ' + module.__name__)
    manifest = load_manifest(root)
    matching = [name for name, spec in manifest['runtimes'].items()
                if (torch.__version__, torchvision.__version__, torch.version.cuda) ==
                   (spec['torch'], spec['torchvision'], spec['cuda'])]
    if len(matching) != 1 or (cuda_runtime != 'auto' and matching[0] != cuda_runtime):
        raise ValueError(f'Unsupported Torch/TorchVision/CUDA combination: {torch.__version__}, {torchvision.__version__}, {torch.version.cuda}')
    runtime = matching[0]
    arches = compiled_architectures(torch)
    expected = manifest['runtimes'][runtime]['architectures']
    if not set(expected) <= set(arches):
        raise ValueError('Torch kernels do not match the pinned official ' + runtime + ' build')
    if torch.backends.cudnn.version() != manifest['runtimes'][runtime]['cudnnVersion']:
        raise ValueError('cuDNN differs from the selected official Torch build; do not replace bundled DLLs independently')
    if runtime == 'cu126' and not (0 < torch.backends.cudnn.version() <= 91002):
        raise ValueError('The Pascal runtime requires its official cuDNN build with Pascal support')
    return dict(cudaRuntime=runtime, python=sys.version.split()[0], torch=torch.__version__,
                torchvision=torchvision.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), architectures=arches)


def validate_runtime(root, requirements=None, cuda_runtime='auto', maintenance_transaction=None):
    root = Path(root).resolve()
    from .torch_runtime_switch import assert_runtime_available
    if maintenance_transaction is None:assert_runtime_available(root)
    else:assert_runtime_available(root,maintenance_transaction)
    import torch
    import torchvision
    import cv2, numpy, scipy  # Native dependency imports are part of runtime acceptance.
    site = runtime_package_prefix(root)
    if (sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != site
            or Path(sys.base_prefix).resolve() != root / '_internal/python_base'):
        raise ValueError('Python must use the project-local Python 3.12 base and .venv')
    for module in [cv2, numpy, scipy]:
        if not Path(module.__file__).resolve().is_relative_to(site):
            raise ValueError('Runtime module is outside the project: ' + module.__name__)
    result = validate_torch_runtime(root, torch, torchvision, cuda_runtime)
    if requirements is not None:
        for line in Path(requirements).read_text(encoding='utf-8').splitlines():
            match = re.match(r'^([\w.-]+)==([^ #]+)', line.strip())
            if match:
                distribution = metadata.distribution(match[1])
                if (distribution.version != match[2] or
                        not Path(distribution.locate_file('')).resolve().is_relative_to(site)):
                    raise ValueError('Runtime dependency differs from its selected lock: ' + match[1])
    for name, version in [('ultralytics', '8.4.142'), ('timm', '0.4.12')]:
        if metadata.version(name) != version:
            raise ValueError('Unexpected production vision dependency: ' + name)
    return result


def verify_runtime_records(root, names=('torch', 'torchvision'), environment=None):
    """Explicit release check; keep multi-GB hashing out of normal startup."""
    prefix = runtime_package_prefix(root, environment)
    counts = {}
    for name in names:
        if environment is None:
            distribution = metadata.distribution(name)
        else:
            distributions = [d for d in metadata.distributions(path=[str(prefix / 'Lib/site-packages')])
                             if d.metadata.get('Name', '').lower() == name.lower()]
            if len(distributions) != 1:
                raise ValueError('Expected one staged wheel distribution: ' + name)
            distribution = distributions[0]
        if not distribution.files:
            raise ValueError('Missing installed wheel RECORD: ' + name)
        count = 0
        for entry in distribution.files:
            if entry.hash is None:
                continue
            path = Path(distribution.locate_file(entry))
            if not path.resolve().is_relative_to(prefix) or not path.is_file():
                raise ValueError('Installed wheel file is missing or outside the environment: ' + str(path))
            with path.open('rb') as stream:
                value = base64.urlsafe_b64encode(hashlib.file_digest(stream, entry.hash.mode).digest()).decode().rstrip('=')
            if value != entry.hash.value:
                raise ValueError('Installed wheel file differs from its RECORD: ' + str(path))
            count += 1
        counts[name] = count
    return counts


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['resolve', 'render-lock', 'validate'])
    parser.add_argument('--cuda-runtime', choices=['auto', 'cu126', 'cu128'], default='auto')
    parser.add_argument('--profile', default='production')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--maintenance-transaction',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == 'validate':
        result = validate_runtime(root, args.requirements, args.cuda_runtime,args.maintenance_transaction)
        print(f"DFL_PT_RUNTIME_OK|{result['python']}|{result['torch']}|{result['cuda']}")
    else:
        if args.maintenance_transaction:parser.error('Private maintenance tokens apply only to validation')
        if args.output_dir is None:
            parser.error('--output-dir is required')
        if args.command == 'render-lock':
            if args.cuda_runtime == 'auto':
                parser.error('render-lock requires an explicit --cuda-runtime')
            runtime = args.cuda_runtime
        else:
            installed = installed_runtime(root, load_manifest(root))
            runtime = select_runtime(args.cuda_runtime, installed, [] if installed else detect_capabilities())
        print(json.dumps(render_profile(root, runtime, args.profile, args.output_dir)))
