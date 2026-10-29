"""CUDA variants share one dependency graph without sharing incompatible wheel hashes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.gpu_runtime import (architecture_supported, derive_sbom, load_manifest,
                              render_profile, runtime_package_prefix, runtime_sbom, select_runtime,
                              validate_cuda_devices, validate_torch_runtime, verify_runtime_records)


@pytest.mark.parametrize('output,exit_code,expected', [
    ('6.1\n12.0\n', 0, [(6, 1), (12, 0)]),
    ('6.1\nnot available\n', 0, []),
    ('6.1\n', 1, []),
])
def test_driver_detection_tolerates_non_utf8_diagnostics_without_losing_valid_devices(monkeypatch, output, exit_code, expected):
    from core import gpu_runtime
    run = gpu_runtime.subprocess.run
    diagnostics = []
    script = ('import sys;sys.stdout.buffer.write(' + repr(output.encode()) + ');'
              'sys.stderr.buffer.write(bytes([0xce,0xd2]));sys.exit(' + str(exit_code) + ')')
    def driver_query(arguments, **kwargs):
        result = run([sys.executable, '-I', '-B', '-c', script], **kwargs)
        diagnostics.append(result.stderr)
        return result
    monkeypatch.setattr(gpu_runtime.subprocess, 'run', driver_query)
    assert gpu_runtime.detect_capabilities() == expected
    assert diagnostics and isinstance(diagnostics[0], str) and '\ufffd' in diagnostics[0]


@pytest.mark.parametrize('capability,arches,supported', [
    ((6, 1), ['sm_60'], True), ((6, 0), ['sm_61'], False),
    ((6, 1), ['sm_70', 'sm_120'], False), ((8, 9), ['sm_86'], True),
    ((12, 0), ['sm_90'], False), ((12, 0), ['compute_90'], True),
    ((12, 0), ['sm_120'], True), ((6, 1), ['compute_70'], False),
])
def test_native_minor_compatibility_and_ptx_forward_compatibility(capability, arches, supported):
    assert architecture_supported(capability, arches) is supported


def test_forward_ptx_alone_does_not_make_cuda126_libraries_support_blackwell():
    torch = SimpleNamespace(__version__='2.9.1+cu126', version=SimpleNamespace(cuda='12.6'),
        _C=SimpleNamespace(_cuda_getArchFlags=lambda:'compute_90'),
        cuda=SimpleNamespace(get_device_capability=lambda index:(12,0), get_device_name=lambda index:'synthetic Blackwell'))
    with pytest.raises(RuntimeError, match='cu128'):
        validate_cuda_devices(torch, [0])


def test_pascal_requires_a_cudnn_build_that_still_supports_its_architecture():
    torch = SimpleNamespace(__version__='2.9.1+cu126', version=SimpleNamespace(cuda='12.6'),
        _C=SimpleNamespace(_cuda_getArchFlags=lambda:'sm_61'),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:91100)),
        cuda=SimpleNamespace(get_device_capability=lambda index:(6,1), get_device_name=lambda index:'synthetic Pascal'))
    with pytest.raises(RuntimeError, match='cu126'):
        validate_cuda_devices(torch, [0])


@pytest.mark.parametrize('requested,installed,capabilities,result', [
    ('auto', None, [], 'cu128'), ('auto', None, [(6, 1)], 'cu126'),
    ('auto', None, [(12, 0)], 'cu128'), ('auto', None, [(6, 1), (8, 6)], 'cu126'),
    ('auto', 'cu128', [(6, 1)], 'cu128'), ('auto', 'cu126', [(12, 0)], 'cu126'),
    ('cu126', None, [(6, 1), (12, 0)], 'cu126'), ('cu128', 'cu128', [], 'cu128'),
])
def test_selection_preserves_existing_environments(requested, installed, capabilities, result):
    assert select_runtime(requested, installed, capabilities) == result


def test_selection_refuses_silent_replacement_and_ambiguous_mixed_hardware():
    with pytest.raises(ValueError, match='will not be replaced'):
        select_runtime('cu126', 'cu128', [(6, 1)])
    with pytest.raises(ValueError, match='different official Torch builds'):
        select_runtime('auto', None, [(6, 1), (12, 0)])


def test_real_legacy_profile_changes_only_two_complete_wheel_records():
    base = json.loads((ROOT / 'release/python-sbom.json').read_text(encoding='utf-8'))
    before = copy.deepcopy(base)
    legacy = runtime_sbom(ROOT, 'cu126')
    modern = runtime_sbom(ROOT, 'cu128')
    assert base == before and legacy['profiles'] == modern['profiles']
    assert set(legacy['packages']) == set(modern['packages'])
    assert {name for name in base['packages'] if legacy['packages'][name] != base['packages'][name]} == {'torch', 'torchvision'}
    for name in ['torch', 'torchvision']:
        assert legacy['packages'][name]['version'].endswith('+cu126')
        assert legacy['packages'][name]['wheel']['sha256'] != modern['packages'][name]['wheel']['sha256']
        assert '+cu126-cp312-cp312-win_amd64.whl' in legacy['packages'][name]['wheel']['file']


def test_overlay_active_windows_dependencies_match_the_shared_graph():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    base = runtime_sbom(ROOT, 'cu128')
    legacy = runtime_sbom(ROOT, 'cu126')
    def active(item):
        return {str(req) for line in item['requires']
                if not (req := Requirement(line)).marker or req.marker.evaluate({'extra': ''})}
    for name in ['torch', 'torchvision']:
        assert active(base['packages'][name]) == active(legacy['packages'][name])
        for requirement in active(legacy['packages'][name]):
            req = Requirement(requirement)
            key = canonicalize_name(req.name)
            assert legacy['packages'][key]['version'] in req.specifier


def project_fixture(tmp_path):
    root = tmp_path / 'project'
    (root / 'release').mkdir(parents=True)
    for name in ['python-sbom.json', 'torch-runtimes.json']:
        (root / 'release' / name).write_bytes((ROOT / 'release' / name).read_bytes())
    return root


@pytest.mark.parametrize('profile', ['production','restoration','validation','evaluation','scene'])
def test_each_functional_lock_uses_the_selected_runtime_hashes(tmp_path, profile):
    root = project_fixture(tmp_path)
    result = render_profile(root, 'cu126', profile, root / 'selection')
    sbom = json.loads(Path(result['sbom']).read_text(encoding='utf-8'))
    lines = Path(result['requirements']).read_text(encoding='utf-8').splitlines()
    for name in sbom['profiles'][profile]:
        item = sbom['packages'][name]
        assert f"{item['name']}=={item['version']} --hash=sha256:{item['wheel']['sha256']}" in lines
    assert result['indexUrl'] == 'https://download.pytorch.org/whl/cu126'
    assert (root / 'release/python-sbom.json').read_bytes() == (ROOT / 'release/python-sbom.json').read_bytes()


def test_render_refuses_an_output_outside_the_project(tmp_path):
    root = project_fixture(tmp_path)
    with pytest.raises(ValueError, match='stay inside the project'):
        render_profile(root, 'cu126', 'production', tmp_path / 'outside')


@pytest.mark.parametrize('directory', ['release', 'release/python-locks'])
def test_resolved_locks_cannot_overwrite_the_modern_source_baseline(tmp_path, directory):
    root = project_fixture(tmp_path)
    original = (root / 'release/python-sbom.json').read_bytes()
    with pytest.raises(ValueError, match='source release baseline'):
        render_profile(root, 'cu126', 'production', root / directory)
    assert (root / 'release/python-sbom.json').read_bytes() == original


def test_overlay_cannot_reuse_a_cu128_record_for_cu126():
    base = json.loads((ROOT / 'release/python-sbom.json').read_text(encoding='utf-8'))
    manifest = load_manifest(ROOT)
    manifest['runtimes']['cu126']['packages']['torch'] = base['packages']['torch']
    with pytest.raises(ValueError, match='Invalid official wheel record'):
        derive_sbom(base, manifest, 'cu126')


def test_offline_verification_rejects_a_wrong_cuda_wheel_even_under_legacy_filename(tmp_path, monkeypatch):
    root = project_fixture(tmp_path)
    base = json.loads((root / 'release/python-sbom.json').read_text(encoding='utf-8'))
    manifest = load_manifest(root)
    base['profiles'] = {'production': ['torch','torchvision']}
    base['packages'] = {name: base['packages'][name] for name in ['torch','torchvision']}
    wheelhouse = root / 'wheels'
    wheelhouse.mkdir()
    for name in ['torch','torchvision']:
        item = manifest['runtimes']['cu126']['packages'][name]['wheel']
        data = ('synthetic legacy '+name).encode()
        item.update(sizeBytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (wheelhouse / item['file']).write_bytes(data)
    (root / 'release/python-sbom.json').write_text(json.dumps(base), encoding='utf-8')
    (root / 'release/torch-runtimes.json').write_text(json.dumps(manifest), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('cuda_variant_dependencies', ROOT / 'tools/runtime-dependencies.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, 'ROOT', root)
    assert tool.verify(wheelhouse, 'production', 'cu126')['torchRuntime'] == 'cu126'
    path = wheelhouse / manifest['runtimes']['cu126']['packages']['torch']['wheel']['file']
    path.write_bytes(b'synthetic modern wheel in the wrong profile')
    with pytest.raises(ValueError, match='Missing or changed locked wheel'):
        tool.verify(wheelhouse, 'production', 'cu126')


def test_legacy_generation_cannot_write_into_the_source_release(tmp_path, monkeypatch):
    root = project_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location('protected_cuda_generation', ROOT / 'tools/runtime-dependencies.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, 'ROOT', root)
    original = (root / 'release/python-sbom.json').read_bytes()
    with pytest.raises(ValueError, match='source release baseline'):
        tool.generate(root / 'release', False, ['production'], 'cu126')
    assert (root / 'release/python-sbom.json').read_bytes() == original


def test_release_integrity_rejects_a_replaced_cuda_library(tmp_path, monkeypatch):
    import base64
    from importlib.metadata import FileHash, PackagePath
    from core import gpu_runtime
    root = tmp_path / 'project'
    library = root / '.venv/Lib/site-packages/torch/lib/cudnn.dll'
    library.parent.mkdir(parents=True)
    data = b'synthetic official compatible library'
    library.write_bytes(data)
    entry = PackagePath('torch/lib/cudnn.dll')
    entry.hash = FileHash('sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='))
    distribution = SimpleNamespace(files=[entry], locate_file=lambda item: library)
    monkeypatch.setattr(gpu_runtime.metadata, 'distribution', lambda name: distribution)
    assert verify_runtime_records(root, ['torch']) == {'torch': 1}
    library.write_bytes(b'a CUDA library from an incompatible build')
    with pytest.raises(ValueError, match='differs from its RECORD'):
        verify_runtime_records(root, ['torch'])


def test_unactivated_candidate_uses_the_same_binary_rules_without_loading_active_torch(tmp_path):
    root = project_fixture(tmp_path)
    prefix = root / '.runtime/torch-variants/staging-test'
    site = prefix / 'Lib/site-packages'
    for name in ['torch', 'torchvision']:
        path = site / name / '__init__.py'
        path.parent.mkdir(parents=True)
        path.write_text('# synthetic module location', encoding='utf-8')
    spec = load_manifest(root)['runtimes']['cu126']
    torch = SimpleNamespace(__name__='torch', __version__=spec['torch'], __file__=str(site/'torch/__init__.py'),
        version=SimpleNamespace(cuda='12.6'), _C=SimpleNamespace(_cuda_getArchFlags=lambda:' '.join(spec['architectures'])),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:91002)))
    vision = SimpleNamespace(__name__='torchvision', __version__=spec['torchvision'], __file__=str(site/'torchvision/__init__.py'))
    assert validate_torch_runtime(root, torch, vision, 'cu126', prefix)['cudaRuntime']=='cu126'
    vision.__version__='0.24.1+cu128'
    with pytest.raises(ValueError, match='combination'):
        validate_torch_runtime(root, torch, vision, 'cu126', prefix)


@pytest.mark.parametrize('suffix', ['.runtime/unmanaged', 'workspace/model', 'release'])
def test_runtime_candidate_prefix_cannot_point_at_other_project_content(tmp_path, suffix):
    root = project_fixture(tmp_path)
    with pytest.raises(ValueError, match='managed runtime prefix'):
        runtime_package_prefix(root, root/suffix)


def test_staged_record_hashes_are_scoped_to_candidate_including_its_console_entry(tmp_path, monkeypatch):
    import base64
    import csv
    from core import gpu_runtime
    root=project_fixture(tmp_path)
    prefix=root/'.runtime/torch-variants/staging-test'
    site=prefix/'Lib/site-packages'
    metadata=site/'torch-2.9.1+cu126.dist-info'
    metadata.mkdir(parents=True)
    (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: torch\nVersion: 2.9.1+cu126\n',encoding='utf-8')
    files=[(site/'torch/marker.py',b'candidate module'),(prefix/'Scripts/torchrun.exe',b'candidate console launcher')]
    rows=[]
    for path,data in files:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        relative=path.relative_to(site).as_posix() if path.is_relative_to(site) else '../../Scripts/'+path.name
        rows.append([relative,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
    with (metadata/'RECORD').open('w',encoding='utf-8',newline='') as stream:
        csv.writer(stream).writerows(rows)
    monkeypatch.setattr(gpu_runtime.metadata,'distribution',lambda name:pytest.fail('read active distribution instead of candidate'))
    assert verify_runtime_records(root,['torch'],environment=prefix)=={'torch':2}
    files[1][0].write_bytes(b'changed console launcher')
    with pytest.raises(ValueError,match='differs from its RECORD'):
        verify_runtime_records(root,['torch'],environment=prefix)
