"""Bounded sequential download failover. Every published byte stays hash pinned.

No global package-manager settings or credentials are forwarded to mirrors.
The policy contains public endpoints only, not user media/model destinations.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
_cooldowns = {}


def policy(root=ROOT):
    value = json.loads((Path(root) / 'release/download-sources.json').read_text(encoding='utf-8'))
    if value.get('schemaVersion') != 1:
        raise ValueError('Unsupported download source policy')
    return value


def mode():
    value = os.environ.get('DFL_MIRROR', 'auto')
    return value if value in {'auto', 'official', 'china'} else 'auto'


def ordered(urls):
    urls = list(dict.fromkeys(urls))
    if mode() == 'official': return urls[:1]
    if mode() == 'china': return urls[1:] + urls[:1]
    # A failed host cools down for this process. Other hosts are independent.
    return sorted(urls, key=lambda u: _cooldowns.get(urllib.parse.urlsplit(u).netloc, 0) > time.monotonic())


def candidates(url, *, root=ROOT, extra=()):
    parts = urllib.parse.urlsplit(url)
    # Never forward signed URLs, tokens or credentials to a public relay.
    if parts.scheme != 'https' or parts.username or parts.password or parts.query or parts.fragment:
        return [url]
    mirrors = [r['mirror'] + url[len(r['prefix']):] for r in policy(root)['rules'] if url.startswith(r['prefix'])]
    return ordered([url, *extra, *mirrors])


def download(url, partial, sha256, size=None, *, root=ROOT, timeout=12,
             opener=None, transform=None, progress=None, extra=(), allowed_redirect_hosts=None):
    """Use one transfer at a time, validate before returning, remove own partial."""
    opener = opener or urllib.request.urlopen
    partial = Path(partial)
    if partial.exists(): raise FileExistsError('Download partial already exists: ' + str(partial))
    failures = []
    for source in candidates(url, root=root, extra=extra):
        owned = False
        try:
            count = 0; value = hashlib.sha256()
            # A fresh request sends no account cookies or authorization headers.
            with opener(source, timeout=timeout) as response, partial.open('xb') as output:
                owned = True
                final = urllib.parse.urlsplit(getattr(response, 'url', source))
                if final.scheme != 'https': raise ValueError('Download redirected away from HTTPS')
                if allowed_redirect_hosts and final.hostname not in allowed_redirect_hosts:
                    raise ValueError('Unexpected download redirect host')
                while True:
                    block = response.read(min(1024 * 1024, size + 1 - count) if size is not None else 1024 * 1024)
                    if not block: break
                    count += len(block)
                    if size is not None and count > size: raise ValueError('Download exceeds pinned size')
                    output.write(block); value.update(block)
                    if progress: progress(count, source)
                output.flush(); os.fsync(output.fileno())
            if transform:
                transform(partial)
                count = partial.stat().st_size
                with partial.open('rb') as stream: value = hashlib.file_digest(stream, 'sha256')
            if (size is not None and count != size) or value.hexdigest() != sha256:
                raise ValueError('Download size or SHA-256 differs from the pinned bytes')
            print(json.dumps({'stage': 'download', 'status': 'verified', 'sourceHost': urllib.parse.urlsplit(source).hostname}), flush=True)
            return source
        except (OSError, ValueError) as error:
            host = urllib.parse.urlsplit(source).netloc
            _cooldowns[host] = time.monotonic() + 60
            failures.append(host + ': ' + str(error))
            if owned and partial.exists(): partial.unlink()
            print(json.dumps({'stage': 'download', 'status': 'retry-source', 'sourceHost': host}), flush=True)
    raise ValueError('All pinned download sources failed: ' + '; '.join(failures))


def pip_routes(runtime, root=ROOT):
    if runtime not in {'cu126', 'cu128'}: raise ValueError('Unsupported CUDA runtime')
    config = policy(root)
    pypi = config['pythonIndexes']; torch = [u.format(runtime=runtime) for u in config['torchIndexes']]
    routes = [(pypi[0], torch[0]), (pypi[1], torch[1]), (pypi[0], torch[1]), (pypi[1], torch[0])]
    if mode() == 'official': return routes[:1]
    if mode() == 'china': routes = routes[1:] + routes[:1]
    return routes
