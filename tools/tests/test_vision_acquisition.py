import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('vision_acquisition',ROOT/'tools/prepare-production-vision.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)

def record(name,data=b'valid',**extra):
    return dict(path=name,sizeBytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**extra)

def fixture():
    group=dict(id='synthetic',profile='production',installRoot='_internal/vision_models/synthetic',
        localInstallationAllowed=True,source={'official_weight_url':'https://example.org/manual'},
        files=[record('first.bin',b'first',url='https://example.org/first'),record('manual.bin'),record('last.bin',b'last',url='https://example.org/last')])
    return {'schemaVersion':1,'groups':[group]},group

def test_download_all_available_then_manual_retry_is_network_free(tmp_path,monkeypatch):
    manifest,group=fixture();calls=[]
    def fetch(url,**kwargs):calls.append(url);return io.BytesIO(url.rsplit('/',1)[1].encode())
    monkeypatch.setattr(tool.urllib.request,'urlopen',fetch)
    with pytest.raises(ValueError):tool.install(tmp_path,manifest,[group],None,True)
    assert len(calls)==2
    report=json.loads((tmp_path/'.launcher-install/vision/resources.json').read_text())
    assert report['failed'][0]['manual'] and report['failed'][0]['path']=='manual.bin'
    manual=tmp_path/report['failed'][0]['savePath'];manual.parent.mkdir(parents=True);manual.write_bytes(b'valid')
    monkeypatch.setattr(tool.urllib.request,'urlopen',lambda *a,**k:pytest.fail('unexpected repeated download'))
    tool.install(tmp_path,manifest,[group],None,True)
    assert json.loads((tmp_path/'.launcher-install/vision/resources.json').read_text())['complete']
    assert not list((tmp_path/'_internal/vision_models').glob('*.staging-*'))

@pytest.mark.parametrize('body',[b'invalid',b'longer-than-pinned-size',b''])
def test_bad_downloads_are_never_cached_or_published(tmp_path,monkeypatch,body):
    manifest,group=fixture();monkeypatch.setattr(tool.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(body))
    with pytest.raises(ValueError):tool.install(tmp_path,manifest,[group],None,True)
    assert not (tmp_path/group['installRoot']).exists()
    assert not list((tmp_path/'.launcher-install/vision/files').glob('*.partial'))
    assert not list((tmp_path/'.launcher-install/vision/files').iterdir())

def test_wrong_manual_file_is_preserved(tmp_path,monkeypatch):
    manifest,group=fixture();manual=tmp_path/'.launcher-install/vision/manual/synthetic/manual.bin'
    manual.parent.mkdir(parents=True);manual.write_bytes(b'user wrong file')
    monkeypatch.setattr(tool.urllib.request,'urlopen',lambda url,**k:io.BytesIO(url.rsplit('/',1)[1].encode()))
    with pytest.raises(ValueError):tool.install(tmp_path,manifest,[group],None,True)
    assert manual.read_bytes()==b'user wrong file'

def test_acquisition_metadata_change_keeps_reviewed_pack_compatible(tmp_path):
    item=record('file',content='valid');g=dict(id='synthetic',profile='production',installRoot='_internal/test',localInstallationAllowed=True,files=[item])
    m={'schemaVersion':1,'groups':[g]};tool.install(tmp_path/'from',m,[g],None,False)
    pack=tmp_path/'pack.zip';tool.pack(tmp_path/'from',m,[g],pack,False)
    item['url']='https://example.org/new-acquisition';tool.verify_pack(pack,m)
    item['sha256']='0'*64
    with pytest.raises(ValueError):tool.verify_pack(pack,m)

def test_current_manual_inventory_has_only_tufa_without_a_direct_url():
    m=json.loads((ROOT/'release/vision-assets.json').read_text())
    manual=[(g['id'],r['path']) for g in m['groups'] if g['profile']=='production' for r in g['files'] if not any(k in r for k in ['url','content','repositoryFile'])]
    assert manual==[('tufa','weights/model.pth')]
    t=next(g for g in m['groups'] if g['id']=='tufa');r=t['files'][0]
    assert tool.check(ROOT/r['repositoryFile'],r)
    assert all(r['path'].endswith('.py') or r['path'].endswith('/LICENSE')
        for g in m['groups'] for r in g['files'] if r.get('textLineEndings'))


def test_explicit_source_line_endings_preserve_reviewed_hash(tmp_path,monkeypatch):
    r=record('source.py',b'a\r\nb\r\n',url='https://example.org/source',textLineEndings='crlf')
    g=dict(id='source',installRoot='_internal/source',localInstallationAllowed=True,files=[r])
    monkeypatch.setattr(tool.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'a\nb\n'))
    tool.install(tmp_path,{},[g],None,True)
    assert (tmp_path/g['installRoot']/r['path']).read_bytes()==b'a\r\nb\r\n'


def test_interrupted_response_cleans_only_partial_and_retains_previous_cache(tmp_path,monkeypatch):
    manifest,group=fixture();calls=[]
    class Broken(io.BytesIO):
        def read(self,size=-1):
            if self.tell():raise TimeoutError('synthetic interrupted response')
            return super().read(1)
    def fetch(url,**kwargs):
        calls.append(url)
        return io.BytesIO(b'first') if url.endswith('/first') else Broken(b'last')
    monkeypatch.setattr(tool.urllib.request,'urlopen',fetch)
    with pytest.raises(ValueError):tool.install(tmp_path,manifest,[group],None,True)
    cache=tmp_path/'.launcher-install/vision/files'
    assert [p.name for p in cache.iterdir()]==[hashlib.sha256(b'first').hexdigest()]
    report=json.loads((tmp_path/'.launcher-install/vision/resources.json').read_text())
    assert {r['path'] for r in report['failed']}=={'manual.bin','last.bin'}


def test_old_pack_clears_stale_pending_report_without_downloading(tmp_path,monkeypatch):
    manifest,group=fixture();group['files']=[record('file',content='valid')]
    tool.install(tmp_path/'source',manifest,[group],None,False)
    pack=tmp_path/'old.zip';tool.pack(tmp_path/'source',manifest,[group],pack,False)
    pending=tmp_path/'target/.launcher-install/vision/resources.json';pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps(dict(schemaVersion=1,complete=False,failed=[])))
    monkeypatch.setattr(tool.urllib.request,'urlopen',lambda *a,**k:pytest.fail('pack reuse must be offline'))
    tool.install_pack(tmp_path/'target',manifest,[group],pack)
    assert json.loads(pending.read_text())['complete']


def test_interrupted_pack_copy_removes_only_owned_partial(tmp_path,monkeypatch):
    manifest,group=fixture();group['files']=[record('file',content='valid')]
    tool.install(tmp_path/'source',manifest,[group],None,False)
    pack=tmp_path/'old.zip';tool.pack(tmp_path/'source',manifest,[group],pack,False)
    real_open=tool.zipfile.ZipFile.open
    class Interrupted(io.BytesIO):
        def read(self,size=-1):
            if self.tell():raise OSError('synthetic interrupted archive')
            return super().read(1)
    calls=[]
    def read_once_then_fail(archive,name,*a,**kw):
        calls.append(name)
        if name==group['installRoot']+'/file' and calls.count(name)==2:return Interrupted(b'valid')
        return real_open(archive,name,*a,**kw)
    monkeypatch.setattr(tool.zipfile.ZipFile,'open',read_once_then_fail)
    with pytest.raises(OSError,match='synthetic'):tool.install_pack(tmp_path/'target',manifest,[group],pack)
    assert not list((tmp_path/'target').rglob('*.staging-*'))
    assert not (tmp_path/'target'/group['installRoot']).exists()


@pytest.mark.parametrize('metadata', [[], {'schemaVersion': 2},
    {'schemaVersion': True, 'publicDistribution': False, 'groups': ['synthetic'], 'files': []},
    {'schemaVersion': 1, 'publicDistribution': False, 'groups': ['synthetic'], 'files': [None]}])
def test_invalid_pack_schema_is_rejected_before_installation(tmp_path, metadata):
    pack=tmp_path/'invalid.zip'
    with tool.zipfile.ZipFile(pack,'w') as archive:
        archive.writestr('RESOURCE-PACK.json',json.dumps(metadata))
    manifest,group=fixture()
    with pytest.raises(ValueError):tool.install_pack(tmp_path/'target',manifest,[group],pack)
    assert not (tmp_path/'target').exists()


def test_staging_cleanup_retains_unknown_and_modified_files(tmp_path):
    with tool.resource_staging(tmp_path/'group','synthetic') as (staging,written):
        tool.copy_owned_resource(io.BytesIO(b'owned'),staging/'owned',written,'owned')
        tool.copy_owned_resource(io.BytesIO(b'initial'),staging/'changed',written,'changed')
        (staging/'changed').write_bytes(b'modified outside the copy')
        (staging/'unknown').write_bytes(b'unknown file')
    assert not (staging/'owned').exists()
    assert (staging/'changed').read_bytes()==b'modified outside the copy'
    assert (staging/'unknown').read_bytes()==b'unknown file'
