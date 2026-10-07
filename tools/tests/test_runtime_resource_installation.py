import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RESOURCES = module('production_resources', 'tools/prepare-production-vision.py')
PROJECT = module('active_project', 'launcher/resolve-active-project.py')


def fixture():
    text = 'immutable official test bytes\n'
    item = {'path': 'model.bin', 'content': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'sizeBytes': len(text.encode())}
    group = {'id': 'fixture', 'profile': 'production', 'installRoot': '_internal/vision_models/production/fixture',
             'localInstallationAllowed': True, 'publicDistributionAllowed': False, 'files': [item]}
    return {'schemaVersion': 1, 'groups': [group]}, group


def test_install_is_immutable_and_repeatable(tmp_path):
    manifest, group = fixture()
    RESOURCES.install(tmp_path, manifest, [group], None, False)
    RESOURCES.install(tmp_path, manifest, [group], None, False)
    target = tmp_path / group['installRoot'] / 'model.bin'
    target.write_bytes(b'user-modification')
    with pytest.raises(ValueError, match='preserved'):
        RESOURCES.install(tmp_path, manifest, [group], None, False)
    assert target.read_bytes() == b'user-modification'


def test_failed_group_never_publishes(tmp_path):
    manifest, group = fixture()
    group['files'].append({'path': 'missing.bin', 'sha256': '0'*64, 'sizeBytes': 12})
    with pytest.raises(ValueError, match='Offline resource missing'):
        RESOURCES.install(tmp_path, manifest, [group], None, False)
    assert not (tmp_path / group['installRoot']).exists()
    assert list((tmp_path / '_internal/vision_models/production').glob('fixture.staging-*'))


def test_public_admission_cannot_be_bypassed_by_local_install(tmp_path):
    manifest, group = fixture()
    RESOURCES.install(tmp_path, manifest, [group], None, False)
    with pytest.raises(ValueError, match='Public distribution'):
        RESOURCES.pack(tmp_path, manifest, [group], tmp_path / 'public.zip', True)
    assert not (tmp_path / 'public.zip').exists()


def test_resource_pack_bound_to_fixed_manifest_and_roundtrips(tmp_path):
    manifest, group = fixture()
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    RESOURCES.install(source, manifest, [group], None, False)
    output = tmp_path / 'pack.zip'
    RESOURCES.pack(source, manifest, [group], output, False)
    RESOURCES.install_pack(dest, manifest, [group], output)
    assert (dest / group['installRoot'] / 'model.bin').read_text() == group['files'][0]['content']
    wrong = dict(manifest, schemaVersion=2)
    with pytest.raises(ValueError, match='fixed project manifest'):
        RESOURCES.verify_pack(output, wrong)


def test_resource_paths_reject_escape_and_zip_duplicate(tmp_path):
    for name in ('../evil', '/evil', 'x/../evil', 'x\\evil', 'C:/evil'):
        with pytest.raises(ValueError):
            RESOURCES.safe(tmp_path, name)
    manifest, group = fixture()
    RESOURCES.install(tmp_path, manifest, [group], None, False)
    output = tmp_path / 'pack.zip'
    RESOURCES.pack(tmp_path, manifest, [group], output, False)
    with pytest.warns(UserWarning, match='Duplicate name'):
        with zipfile.ZipFile(output, 'a') as archive:
            archive.writestr('RESOURCE-PACK.json', archive.read('RESOURCE-PACK.json'))
    with pytest.raises(ValueError, match='inventory mismatch'):
        RESOURCES.verify_pack(output, manifest)


def test_active_projects_follow_same_ids_without_cross_project_writes(tmp_path):
    registry = tmp_path / 'webui/.runtime/projects.json'
    registry.parent.mkdir(parents=True)
    projects = [{'id': 'default', 'name': 'default'}, {'id': 'case-two', 'name': 'two'}]
    for active in ('default', 'case-two'):
        registry.write_text(json.dumps({'activeId': active, 'projects': projects}))
        value = PROJECT.resolve(tmp_path)
        assert value['id'] == active
        assert Path(value['workspace']) == tmp_path / ('workspace' if active == 'default' else 'workspaces/case-two')
        assert not Path(value['workspace']).exists()
    for active in ('../../outside', 'unknown'):
        registry.write_text(json.dumps({'activeId': active, 'projects': projects}))
        with pytest.raises(ValueError):
            PROJECT.resolve(tmp_path)


def test_production_graph_has_one_cv_wheel_and_excludes_research_tools():
    sbom = json.loads((ROOT / 'release/python-sbom.json').read_text(encoding='utf-8'))
    production = set(sbom['profiles']['production'])
    assert {'torch', 'torchvision', 'ultralytics', 'timm', 'onnx', 'pyside6'} <= production
    assert not production & {'h5py', 'onnxruntime', 'basicsr', 'gfpgan', 'pytest', 'tb-nightly'}
    assert [name for name in production if name.startswith('opencv-python')] == ['opencv-python']
    for profile, packages in sbom['profiles'].items():
        lock = (ROOT / f'release/python-locks/{profile}-win-cp312.txt').read_text()
        assert lock.count('--hash=sha256:') == len(packages)
        for key in packages:
            assert sbom['packages'][key]['wheel']['sha256'] in lock
