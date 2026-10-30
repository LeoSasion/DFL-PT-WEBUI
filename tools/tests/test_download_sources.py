import hashlib
import io
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'_internal/DeepFaceLab'))
from core import download_sources as sources

@pytest.fixture(autouse=True)
def clean_routes(monkeypatch):
    monkeypatch.delenv('DFL_MIRROR',raising=False)
    sources._cooldowns.clear()

class Response(io.BytesIO):
    def __init__(self,data,url):super().__init__(data);self.url=url

@pytest.mark.parametrize('first_failure',['timeout','hash','truncated','oversize'])
def test_first_failure_switches_once_and_publishes_only_pinned_bytes(tmp_path,first_failure):
    url='https://nodejs.org/dist/v24.19.0/test.zip';calls=[]
    def fetch(u,**kw):
        calls.append(u)
        if len(calls)==1:
            if first_failure=='timeout':raise TimeoutError('synthetic timeout')
            return Response({'hash':b'bad','truncated':b'o','oversize':b'oversized'}[first_failure],u)
        return Response(b'yes',u)
    partial=tmp_path/'file.partial'
    chosen=sources.download(url,partial,hashlib.sha256(b'yes').hexdigest(),3,opener=fetch)
    assert len(calls)==2 and chosen.startswith('https://registry.npmmirror.com/')
    assert partial.read_bytes()==b'yes'

def test_all_failed_routes_clean_the_owned_partial(tmp_path):
    p=tmp_path/'file.partial'
    with pytest.raises(ValueError,match='All pinned'):
        sources.download('https://github.com/a/b/releases/download/v1/file',p,'0'*64,3,
            opener=lambda u,**k:Response(b'bad',u))
    assert not p.exists()

def test_existing_partial_is_preserved(tmp_path):
    p=tmp_path/'file.partial';p.write_bytes(b'unknown')
    with pytest.raises(FileExistsError):sources.download('https://github.com/a/b',p,'0'*64)
    assert p.read_bytes()==b'unknown'

@pytest.mark.parametrize('url',['https://user:secret@github.com/a','https://github.com/a?token=private','https://github.com/a#private'])
def test_private_or_signed_urls_are_never_sent_to_relays(url):
    assert sources.candidates(url)==[url]

def test_failed_host_cools_down_without_penalizing_other_hosts(tmp_path):
    def fetch(u,**k):
        if u.startswith('https://nodejs.org/'):raise TimeoutError()
        return Response(b'yes',u)
    sources.download('https://nodejs.org/dist/a',tmp_path/'file',hashlib.sha256(b'yes').hexdigest(),3,opener=fetch)
    assert sources.candidates('https://nodejs.org/dist/b')[0].startswith('https://registry.npmmirror.com/')
    assert sources.candidates('https://github.com/a/b')[0].startswith('https://github.com/')

def test_modes_keep_cuda_variants_and_hashes_independent(monkeypatch):
    for runtime in ['cu126','cu128']:
        assert all(runtime in torch for _,torch in sources.pip_routes(runtime))
    monkeypatch.setenv('DFL_MIRROR','official')
    assert len(sources.pip_routes('cu126'))==1 and len(sources.candidates('https://nodejs.org/dist/a'))==1
    monkeypatch.setenv('DFL_MIRROR','china')
    assert 'tuna' in sources.pip_routes('cu128')[0][0]
    assert 'npmmirror' in sources.candidates('https://nodejs.org/dist/a')[0]

def test_redirect_downgrade_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='HTTPS'):
        sources.download('https://example.org/file',tmp_path/'file','0'*64,3,
            opener=lambda u,**k:Response(b'yes','http://example.org/file'))
    assert not (tmp_path/'file').exists()
