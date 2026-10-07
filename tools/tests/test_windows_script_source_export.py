import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('windows_script_source_release',ROOT/'tools/build-release.py')
RELEASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=RELEASE;spec.loader.exec_module(RELEASE)

def test_git_batch_source_exports_cmd_compatible_crlf_with_unchanged_canonical_fingerprint(monkeypatch):
    raw='@echo off\nchcp 65001 >nul\r\necho 中文\nexit /b 0'.encode('utf-8')
    calls=[]
    def blob(*args):calls.append(args);return raw
    monkeypatch.setattr(RELEASE,'git',blob)
    entry=RELEASE.git_source_entry('legacy-cli/menu.bat',len(raw),'pinned-blob')
    with entry.content() as stream:exported=stream.read()
    assert entry.blob=='pinned-blob' and entry.size==len(exported)
    assert exported==raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    assert hashlib.sha256(exported.replace(b'\r\n',b'\n')).digest()==hashlib.sha256(raw.replace(b'\r\n',b'\n')).digest()
    assert calls==[('cat-file','blob','pinned-blob')]

@pytest.mark.parametrize('raw',[b'@echo\x00off\n',b'@echo \xff\n'])
def test_non_utf8_or_nul_batch_source_is_rejected(monkeypatch,raw):
    monkeypatch.setattr(RELEASE,'git',lambda *args:raw)
    with pytest.raises(RELEASE.ReleaseError,match='UTF-8 text without NUL'):
        RELEASE.git_source_entry('install.cmd',len(raw),'pinned-blob')

def test_non_batch_source_and_runtime_files_are_never_rewritten(monkeypatch,tmp_path):
    monkeypatch.setattr(RELEASE,'git',lambda *args:(_ for _ in ()).throw(AssertionError('Unexpected Git blob read')))
    entry=RELEASE.git_source_entry('tools/script.py',17,'source-blob')
    assert entry.data is None and entry.blob=='source-blob' and entry.size==17
    monkeypatch.setattr(RELEASE,'REPO',tmp_path)
    path=tmp_path/'runtime/npm.cmd';path.parent.mkdir();raw=b'@echo off\nexit /b\n';path.write_bytes(raw)
    plan=RELEASE.Plan('portable',{},'commit','bundled');plan.file('runtime/npm.cmd')
    with plan.entries['runtime/npm.cmd'].content() as stream:assert stream.read()==raw
