"""Bounded actual aligned-copy and native ME merge inference; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'workspace/.vision-evaluation/merge-quality-20261007-v1'
OUT = BASE / 'restoration-acceptance-v3'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists(): raise ValueError('Acceptance output must be fresh')
    OUT.mkdir()
    inputs = list((BASE / 'aligned').glob('*.jpg')) + list((BASE / 'frames').glob('*.png')) + [BASE / 'model/verification/me.pt']
    original = {str(path): sha(path) for path in inputs}
    env = {**os.environ, 'PYTHONUTF8': '1'}
    report = {'schemaVersion': 1, 'scope': 'actual aligned full independent copy + actual native ME prediction/merging; no training', 'runs': {}}
    for model in ('mambairv2', 'realesrgan-x4plus'):
        args = [sys.executable, str(ROOT / '_internal/DeepFaceLab/main.py'), 'facesettool', 'enhance',
                '--input-dir', str(BASE / 'aligned'), '--enhancement-model', model, '--offset', '1', '--limit', '1', '--force-gpu-idxs', '0']
        result = subprocess.run(args, capture_output=True, env=env, timeout=240)
        (OUT / (model + '-aligned.out.log')).write_bytes(result.stdout)
        (OUT / (model + '-aligned.err.log')).write_bytes(result.stderr)
        if result.returncode: raise RuntimeError('Aligned restoration failed: ' + model)
        receipts = list((BASE / 'aligned_enhanced').glob('enhance-*/enhancement.receipt.json'))
        matching = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in receipts]
        matching = [(path, item) for path, item in matching if item['model']['model'] == model and item['status'] == 'completed']
        if not matching: raise RuntimeError('Real aligned restoration receipt missing')
        path, receipt = max(matching, key=lambda pair: pair[0].stat().st_mtime_ns)
        assert receipt['copiedCount'] == len(list((BASE/'aligned').glob('*.jpg'))) and receipt['selectedCount'] == 1
        config = {'mode': 'overlay', 'maskMode': 4, 'colorTransfer': 'rct', 'workers': 1,
                  'randomSeed': 121, 'blurMask': 16, 'superResolution': 25, 'superResolutionModel': model}
        env['DFL_WEB_MERGE_CONFIG'] = json.dumps(config)
        args = [sys.executable, str(ROOT/'_internal/DeepFaceLab/main.py'), 'merge', '--input-dir', str(BASE/'frames'),
            '--output-dir', str(OUT/model/'merged'), '--output-mask-dir', str(OUT/model/'masks'), '--aligned-dir', str(BASE/'aligned'),
            '--model-dir', str(BASE/'model'), '--model', 'ME', '--force-model-name', 'verification', '--force-gpu-idxs', '0',
            '--xseg-dir', str(ROOT/'_internal/model_generic_xseg')]
        result = subprocess.run(args, capture_output=True, env=env, timeout=300)
        (OUT/(model+'-merge.out.log')).write_bytes(result.stdout)
        (OUT/(model+'-merge.err.log')).write_bytes(result.stderr)
        audit = json.loads((OUT/model/'merged/merge.audit.json').read_text(encoding='utf-8'))
        if result.returncode or audit['status'] != 'complete': raise RuntimeError('Native merge restoration failed: '+model)
        outputs = list((OUT/model/'merged').glob('*.png'))
        assert len(outputs) == len(list((BASE/'frames').glob('*.png')))
        report['runs'][model] = {'alignedReceipt': str(path), 'selectedRange': receipt['selectedRange'],
            'alignedCompleteCopy': True, 'mergeConfig': config, 'actualMergeOutputCount': len(outputs), 'mergeAudit': audit,
            'alignedExitCode': 0, 'mergeExitCode': result.returncode}
        print('Verified actual aligned restoration and ME merging: '+model, flush=True)
    report['originalSha256'] = original
    report['originalsUnchanged'] = all(sha(Path(path)) == digest for path, digest in original.items())
    if not report['originalsUnchanged']: raise RuntimeError('Originals changed')
    (OUT/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
