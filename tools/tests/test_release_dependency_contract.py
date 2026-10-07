import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('release_dependency_contract', ROOT / 'tools/build-release.py')
RELEASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RELEASE
spec.loader.exec_module(RELEASE)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(RELEASE, 'REPO', tmp_path)
    directory = tmp_path / 'launcher/bin'
    directory.mkdir(parents=True)
    binary = b'original native compiled artifact'
    source = b'current active project contract'
    (directory / 'DFL-PT-WEBUI.Launcher.exe').write_bytes(binary)
    record = {'schemaVersion': 1, 'executable': {'file': 'DFL-PT-WEBUI.Launcher.exe',
        'sha256': hashlib.sha256(binary).hexdigest(), 'bytes': len(binary)},
        'sources': [{'path': 'launcher/host/DflEnvironment.cs', 'sha256': hashlib.sha256(source).hexdigest(),
                     'bytes': len(source)}]}
    (directory / 'launcher-build.json').write_text(json.dumps(record), encoding='utf-8')
    entry = SimpleNamespace(blob='snapshot-blob', content=lambda: io.BytesIO(source))
    return SimpleNamespace(entries={'launcher/host/DflEnvironment.cs': entry}), directory


def test_stale_native_environment_code_cannot_ship_with_new_source(tmp_path, monkeypatch):
    plan, directory = fixture(tmp_path, monkeypatch)
    RELEASE.verify_launcher_build(plan)
    plan.entries['launcher/host/DflEnvironment.cs'].content = lambda: io.BytesIO(b'changed project context contract')
    with pytest.raises(RELEASE.ReleaseError, match='different source'):
        RELEASE.verify_launcher_build(plan)


def test_mixed_windows_line_endings_require_exact_recorded_build_bytes(tmp_path, monkeypatch):
    plan, directory = fixture(tmp_path, monkeypatch)
    mixed = b'class ActiveProject {\r\n    public string Workspace;\n}\r\n'
    record_path = directory / 'launcher-build.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record['sources'][0].update(bytes=len(mixed), sha256=hashlib.sha256(mixed).hexdigest())
    record_path.write_text(json.dumps(record), encoding='utf-8')
    working = tmp_path / 'launcher/host/DflEnvironment.cs'
    working.parent.mkdir(parents=True)
    working.write_bytes(mixed)
    entry = plan.entries['launcher/host/DflEnvironment.cs']
    entry.content = lambda: io.BytesIO(mixed.replace(b'\r\n', b'\n'))
    RELEASE.verify_launcher_build(plan)
    working.write_bytes(mixed.replace(b'Workspace', b'OldSource'))
    with pytest.raises(RELEASE.ReleaseError, match='different source'):
        RELEASE.verify_launcher_build(plan)


def test_changed_binary_or_missing_source_cannot_satisfy_build_record(tmp_path, monkeypatch):
    plan, directory = fixture(tmp_path, monkeypatch)
    plan.entries.clear()
    with pytest.raises(RELEASE.ReleaseError, match='absent from the release snapshot'):
        RELEASE.verify_launcher_build(plan)
    (directory / 'DFL-PT-WEBUI.Launcher.exe').write_bytes(b'other native executable')
    with pytest.raises(RELEASE.ReleaseError, match='differs from its build record'):
        RELEASE.verify_launcher_build(plan)


def test_git_text_line_endings_are_allowed_without_ignoring_code_changes(tmp_path, monkeypatch):
    plan, directory = fixture(tmp_path, monkeypatch)
    windows_source = b'class ActiveProject {\r\n    public string Workspace;\r\n}\r\n'
    record_path = directory / 'launcher-build.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record['sources'][0].update(bytes=len(windows_source), sha256=hashlib.sha256(windows_source).hexdigest())
    record_path.write_text(json.dumps(record), encoding='utf-8')
    git_source = windows_source.replace(b'\r\n', b'\n')
    entry = plan.entries['launcher/host/DflEnvironment.cs']
    entry.content = lambda: io.BytesIO(git_source)
    RELEASE.verify_launcher_build(plan)
    entry.content = lambda: io.BytesIO(git_source.replace(b'Workspace', b'OldSource'))
    with pytest.raises(RELEASE.ReleaseError, match='different source'):
        RELEASE.verify_launcher_build(plan)


def runtime_fixture(tmp_path, monkeypatch, files):
    monkeypatch.setattr(RELEASE, 'REPO', tmp_path)
    site_packages = tmp_path / '.venv/Lib/site-packages'
    site_packages.mkdir(parents=True)
    module = site_packages / 'cv2/__init__.py'
    module.parent.mkdir()
    module.write_bytes(b'wheel-owned import module')
    distribution = SimpleNamespace(version='4.12.0.88', files=files,
        locate_file=lambda name: site_packages / name)
    monkeypatch.setattr(RELEASE.importlib.metadata, 'distribution', lambda name: distribution)
    config = {'runtimeExclude': ['**/tests/**', '**/*.pyc'], 'portableExclude': [],
              'pythonRuntimeInclude': ['numpy/_core/tests/_natype.py']}
    plan = RELEASE.Plan('portable', config, 'fixture-commit', 'bundled')
    item = {'name': 'opencv-python', 'version': distribution.version}
    return plan, item, distribution, site_packages


@pytest.mark.parametrize('missing', ['cv2/cv2.pyd', 'timm/models/vision_transformer.py'])
def test_locked_runtime_missing_import_files_fail_before_archiving(tmp_path, monkeypatch, missing):
    plan, item, _, _ = runtime_fixture(tmp_path, monkeypatch, ['cv2/__init__.py', missing])
    with pytest.raises(RELEASE.ReleaseError, match='runtime file is missing'):
        RELEASE.add_python_runtime_distribution(plan, item)


def test_explicitly_excluded_record_files_and_non_import_wheel_paths_can_be_absent(tmp_path, monkeypatch):
    files = ['cv2/__init__.py', 'cv2/tests/test_runtime.py', 'cv2/__pycache__/cached.pyc',
             '../../Scripts/tool.exe', '../../share/man/man1/tool.1']
    plan, item, _, _ = runtime_fixture(tmp_path, monkeypatch, files)
    RELEASE.add_python_runtime_distribution(plan, item)
    assert list(plan.entries) == ['.venv/Lib/site-packages/cv2/__init__.py']


def test_runtime_allowlist_cannot_silently_omit_a_missing_module_under_tests(tmp_path, monkeypatch):
    plan, item, _, _ = runtime_fixture(tmp_path, monkeypatch, ['cv2/__init__.py', 'numpy/_core/tests/_natype.py'])
    with pytest.raises(RELEASE.ReleaseError, match='runtime file is missing'):
        RELEASE.add_python_runtime_distribution(plan, item)


def test_same_version_distribution_from_another_environment_is_rejected(tmp_path, monkeypatch):
    plan, item, distribution, _ = runtime_fixture(tmp_path, monkeypatch, ['cv2/__init__.py'])
    distribution.locate_file = lambda name: tmp_path / 'foreign-runtime/site-packages' / name
    with pytest.raises(RELEASE.ReleaseError, match='outside the project runtime'):
        RELEASE.add_python_runtime_distribution(plan, item)


def test_record_entry_outside_the_project_environment_is_rejected(tmp_path, monkeypatch):
    plan, item, _, _ = runtime_fixture(tmp_path, monkeypatch, ['cv2/__init__.py', '../../../outside.py'])
    with pytest.raises(RELEASE.ReleaseError, match='escapes the project runtime'):
        RELEASE.add_python_runtime_distribution(plan, item)


@pytest.mark.parametrize('files', [None, []])
def test_locked_distribution_requires_a_wheel_file_inventory(tmp_path, monkeypatch, files):
    plan, item, _, _ = runtime_fixture(tmp_path, monkeypatch, files)
    with pytest.raises(RELEASE.ReleaseError, match='no file inventory'):
        RELEASE.add_python_runtime_distribution(plan, item)
