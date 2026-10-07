"""Bounded real aligned-only development acceptance; no training or originals edits."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'webui/python'), str(ROOT / '_internal/DeepFaceLab')]
import best_training_faceset as backend


def hashes(paths):
    return {str(path): backend.sha256_file(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', default=12, type=int)
    args = parser.parse_args()
    if not 8 <= args.count <= 12: raise ValueError('Real acceptance uses 8..12 samples')
    source = backend._ordinary(args.source, directory=True)
    out = args.output.absolute()
    if out.exists() or out.is_relative_to(source): raise ValueError('Use a new independent acceptance directory')
    images = sorted(source.glob('*.jpg'))
    if len(images) < args.count: raise ValueError('Not enough real aligned samples')
    indexes = [round(i * (len(images) - 1) / (args.count - 1)) for i in range(args.count)]
    chosen = [images[index] for index in indexes]
    originals = chosen + [path.with_suffix('.jpg.landmarks.json') for path in chosen if path.with_suffix('.jpg.landmarks.json').is_file()]
    before = hashes(originals)
    out.mkdir(parents=True); aligned = out / 'aligned'; aligned.mkdir()
    for path in originals: shutil.copyfile(path, aligned / path.name)
    copy_before = hashes(sorted(aligned.iterdir()))
    references = [chosen[index].name for index in (0, args.count // 2, args.count - 1)]
    started = time.monotonic()
    import cv2
    import torch
    cv2.setNumThreads(2); torch.set_num_threads(2)
    state = backend.create_plan(aligned, out / 'plans', target_count=args.count)
    plan = Path(state['planDirectory'])
    models = backend.AnalysisModels(device='cpu')
    first = backend.analyze_batch(plan, offset=0, limit=args.count // 2, models=models)
    try:
        backend.finalize_plan(plan)
        raise AssertionError('Partial inventory must not finalize')
    except backend.PlanConflict:
        pass
    backend.analyze_batch(plan, offset=args.count // 2, limit=args.count, models=models)
    unconfirmed = backend.finalize_plan(plan)
    assert unconfirmed['selection']['counts']['selected'] == 0
    backend.confirm_references(plan, references)
    state = backend.finalize_plan(plan)
    dry = backend.publish_plan(plan, out / 'published', dry_run=True)
    assert not (out / 'published').exists()
    publication = backend.publish_plan(plan, out / 'published')
    receipt = json.loads(Path(publication['receiptPath']).read_text(encoding='utf-8'))
    backend.recover_publication(publication['receiptPath'], dry_run=True)
    recovered = backend.recover_publication(publication['receiptPath'])
    features = [json.loads(row[0]) for row in sqlite3.connect(plan / 'features.sqlite').execute('SELECT feature FROM records ORDER BY position')]
    assert before == hashes(originals) and copy_before == hashes(sorted(aligned.iterdir()))
    for entry in receipt['entries']:
        retained = Path(publication['outputDirectory']) / 'withdrawn' / entry['destination']
        assert backend.sha256_file(retained) == entry['sha256']
    report = {'schemaVersion': 1, 'scope': 'bounded aligned-only development acceptance; no training; no calibration',
        'source': str(source), 'sourceInventoryCount': len(images), 'nonAdjacentSampleIndexes': indexes,
        'alignedCopies': str(aligned), 'planDirectory': str(plan), 'references': references,
        'referenceScope': 'developer-selected known source-side examples for lifecycle acceptance; not identity annotation accuracy',
        'firstWindow': {'selectedRange': first['selectedRange'], 'globalReady': first['globalReady']},
        'unconfirmedCounts': unconfirmed['selection']['counts'], 'selection': state['selection'],
        'qualityScores': [{'member': row['member'], 'quality': row.get('quality'), 'hardReasons': row['hardReasons'],
                           'reviewReasons': row['reviewReasons'], 'native98': row.get('native98'), 'detector': row.get('detector'),
                           'identityFeature': row.get('identityFeature'), 'coverage': row.get('coverage')} for row in features],
        'dryRun': dry, 'publication': publication, 'recovery': recovered,
        'originalSha256Before': before, 'originalsUnchanged': before == hashes(originals),
        'isolatedInputCopiesUnchanged': copy_before == hashes(sorted(aligned.iterdir())),
        'outputPixelsMetadataSidecarsByteExact': True, 'elapsedSeconds': time.monotonic() - started,
        'limitations': ['12 aligned-only samples cannot validate general quality/identity accuracy or expression labels',
                        'No reliable original-frame PTS provided; no temporal duplicate assertion',
                        'Adequacy floor 5/100 and diversity threshold are transparent development policies, not calibrated cutoffs']}
    backend._json(out / 'report.json', report)
    print(json.dumps({'report': str(out / 'report.json'), 'alignedCopies': str(aligned), 'counts': state['selection']['counts'],
                      'elapsedSeconds': report['elapsedSeconds']}, ensure_ascii=False))


if __name__ == '__main__': main()
