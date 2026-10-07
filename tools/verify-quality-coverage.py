"""Verify a real aligned window transaction in a new independent QA copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.faceset_transaction import recover_transaction
from mainscripts import Sorter


def inventory(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or not output.is_relative_to(ROOT / 'workspace/.vision-evaluation'):
        raise ValueError('Use a new isolated ignored evaluation directory')
    source_before = inventory(args.source)
    output.mkdir(parents=True)
    copy = output / 'aligned'; shutil.copytree(args.source, copy)
    before = inventory(copy)
    preview = Sorter.main(copy, 'quality-coverage', target_count=2, offset=2, limit=4, dry_run=True)
    if inventory(copy) != before or preview['receipt_path'] is not None:
        raise RuntimeError('Dry-run modified independent faceset')
    receipt = Sorter.main(copy, 'quality-coverage', target_count=2, offset=2, limit=4)
    outside = set(before) - {entry['source'] for entry in receipt['details']['scores']}
    outside_preserved = all((copy / name).is_file() and hashlib.sha256((copy / name).read_bytes()).hexdigest() == before[name]
                            for name in outside if not name.endswith('.landmarks.json'))
    recover_transaction(receipt['receipt_path'])
    restored = inventory(copy) == before
    unchanged = inventory(args.source) == source_before
    report = {'schemaVersion': 1, 'originalsUnchanged': unchanged,
              'dryRunUnchanged': True, 'outsideWindowUnchanged': outside_preserved,
              'independentCopyRestoredExactly': restored,
              'sourceSha256': source_before, 'selectedRange': preview['details']['qualityCoverage']['selectedRange'],
              'selected': preview['details']['selected_count'], 'archived': preview['details']['archived_count'],
              'completeCandidateAudit': len(preview['details']['scores']) == 4,
              'receipt': receipt['receipt_path'], 'newTraining': False}
    (output / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not all((unchanged, restored, outside_preserved)):
        raise RuntimeError('Real window transaction acceptance failed')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
