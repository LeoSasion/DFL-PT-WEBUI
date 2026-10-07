"""Explicit acquisition of the single FIQA comparison candidate; no pip changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webui/python'))
from vision_fiqa import ASSETS, REVISION, SOURCE, verified_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'workspace/.vision-models/efficient-fiqa')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'workspace/.vision-models'):
        raise ValueError('Acquire comparison assets inside the ignored model cache')
    output.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/sunwei925/Efficient-FIQA/{REVISION}/'
    paths = {'EdgeNeXt_XXS_checkpoint.pt': 'ckpts/EdgeNeXt_XXS_checkpoint.pt',
             'FIQA_model.py': 'models/FIQA_model.py', 'LICENSE': 'LICENSE', 'test.py': 'test.py'}
    pypi = None
    for name, (digest, size) in ASSETS.items():
        target = output / name
        if target.is_file() and target.stat().st_size == size and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        if name.endswith('.whl'):
            pypi = json.load(urllib.request.urlopen('https://pypi.org/pypi/timm/1.0.19/json', timeout=30))
            item = next(x for x in pypi['urls'] if x['filename'] == name and x['digests']['sha256'] == digest)
            url = item['url']
        else:
            url = base + paths[name]
        data = urllib.request.urlopen(url, timeout=60).read()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Upstream bytes changed: {name}')
        temporary = target.with_suffix(target.suffix + '.partial')
        temporary.write_bytes(data)
        temporary.replace(target)
    verified_assets(output)
    record = {'model': 'Efficient-FIQA / EdgeNeXt-XXS student only', 'source': SOURCE,
              'sourceRevision': REVISION, 'codeLicense': 'Apache-2.0',
              'weightTerms': 'official repository checkpoint; no separate redistribution admission',
              'publicDistribution': False, 'admission': 'comparison-only', 'default': False,
              'runtime': 'same project Python/PyTorch; pinned timm wheel isolated in child process',
              'pipEnvironmentChanged': False,
              'architectureLicense': 'timm 1.0.19 Apache-2.0; original wheel license retained inside wheel',
              'files': [{'path': name, 'sha256': item[0], 'sizeBytes': item[1],
                         'source': 'https://pypi.org/project/timm/1.0.19/' if name.endswith('.whl') else base + paths[name]}
                        for name, item in ASSETS.items()]}
    (output / 'SOURCE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'output': str(output), 'files': len(ASSETS), 'admission': 'comparison-only'}))


if __name__ == '__main__':
    main()
