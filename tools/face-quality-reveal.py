"""Reveal a completed locked subjective clarity review without fitting scores."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True, type=Path)
    args = parser.parse_args()
    root = args.evaluation.absolute()
    locked_path = root / 'blind/locked-scores.json'
    key_path, proxy_path = root / 'blind-key.json', root / 'proxy-scores.json'
    locked = json.loads(locked_path.read_text(encoding='utf-8'))
    if locked.get('status') != 'locked-before-reveal':
        raise ValueError('Independent review must be locked before revealing')
    key = json.loads(key_path.read_text(encoding='utf-8'))
    proxies = {row['id']: row for row in json.loads(proxy_path.read_text(encoding='utf-8'))}
    subjective = {row['id']: row for row in locked['scores']}
    manifest = json.loads((root / 'blind/manifest.json').read_text(encoding='utf-8'))
    ids = [entry['id'] for entry in manifest['entries']]
    if len(ids) != 12 or set(ids) != set(proxies) or set(ids) != set(subjective) or set(ids) != set(key):
        raise ValueError('Locked review, key and measured proxy cases disagree')
    for entry in manifest['entries']:
        if sha(root / 'blind' / entry['file']) != entry['sha256']:
            raise ValueError('Reviewed image changed')
    groups = defaultdict(dict)
    for identifier in ids:
        groups[key[identifier]['group']][key[identifier]['variant']] = identifier
    metrics = ('foreground_tenengrad', 'foreground_brenner', 'legacy_cpbd', 'source_box_area')
    fields = ('usefulEyeMouthDetail', 'naturalClarity', 'overall')
    results = {}
    for metric in metrics:
        comparisons = []
        for group, variants in sorted(groups.items()):
            for left_variant, right_variant in itertools.combinations(('original', 'blur-moderate', 'blur-strong', 'blur-noise'), 2):
                left, right = variants[left_variant], variants[right_variant]
                proxy_order = sign(proxies[left]['scores'][metric] - proxies[right]['scores'][metric])
                high_confidence = {left_variant, right_variant} != {'blur-strong', 'blur-noise'}
                human = {field: sign(subjective[left][field] - subjective[right][field]) for field in fields}
                comparisons.append({'group': group, 'left': left, 'right': right,
                    'variants': [left_variant, right_variant], 'highConfidence': high_confidence,
                    'proxyDirection': proxy_order, 'subjectiveDirection': human,
                    'matches': {field: proxy_order == human[field] for field in fields}})
        summary = {}
        for field in fields:
            def summarize(items):
                return {'matched': sum(item['matches'][field] for item in items),
                        'total': len(items), 'proxyTies': sum(item['proxyDirection'] == 0 for item in items)}
            coefficient = None
            if np.ptp([proxies[id]['scores'][metric] for id in ids]) > 0:
                coefficient = float(spearmanr([proxies[id]['scores'][metric] for id in ids],
                                              [subjective[id][field] for id in ids]).statistic)
            summary[field] = {'withinSourcePairs': summarize(comparisons),
                'highConfidenceWithinSourcePairs': summarize([item for item in comparisons if item['highConfidence']]),
                'controlledBlurPairs': summarize([item for item in comparisons if 'blur-noise' not in item['variants']]),
                'contaminationVersusOriginalOrModeratePairs': summarize([item for item in comparisons
                    if 'blur-noise' in item['variants'] and item['highConfidence']]),
                'lowConfidenceWithinSourcePairs': summarize([item for item in comparisons if not item['highConfidence']]),
                'allCaseSpearmanDescriptiveOnly': coefficient}
        results[metric] = {'summary': summary, 'comparisons': comparisons,
            'overallDisagreements': [item for item in comparisons if not item['matches']['overall']]}
    ordering = {metric: sorted(ids, key=lambda id: (-proxies[id]['scores'][metric], id)) for metric in metrics}
    identical = ordering['foreground_tenengrad'] == ordering['foreground_brenner']
    report = {'schemaVersion': 1, 'reviewer': locked['reviewer'], 'reviewLockStatus': locked['status'],
        'inputsSha256': {'lockedScores': sha(locked_path), 'blindKey': sha(key_path), 'proxyScores': sha(proxy_path),
                        'manifest': sha(root / 'blind/manifest.json')},
        'caseCount': len(ids), 'sourceGroupCount': len(groups), 'withinSourcePairCount': 18,
        'metrics': results, 'bestToWorstProxyOrdering': ordering,
        'newCandidatesHaveIdenticalCaseOrdering': identical,
        'selectedMetric': 'foreground_tenengrad' if identical else None,
        'selectionReason': 'Equal observed ordering; deterministic conventional Sobel-gradient candidate tie-break, not a demonstrated universal winner.' if identical else 'Further review needed.',
        'scoringFormulaChangedAfterReveal': False,
        'lowConfidencePolicy': 'No tuning to force blur-noise versus blur-strong ordering; retain all disagreements.',
        'limits': ['Three nearby frames from the same person and short clip; controlled variants are not independent samples.',
                   'Pair counts and Spearman are descriptive consistency summaries, not accuracy, AP, identity or annotation metrics.',
                   'Within-variant visual ties are intentional; all-case correlations should not imply meaningful within-variant ordering.',
                   'Eye/mouth coverage uses uncalibrated existing-landmark geometric buckets; broad posture/expression/occlusion quality remains unverified.'],
        'revealedCases': [{'id': id, 'group': key[id]['group'], 'variant': key[id]['variant'],
                           'subjective': subjective[id], 'scores': proxies[id]['scores']} for id in ids]}
    target = root / 'reveal.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old['inputsSha256'] != report['inputsSha256']:
            raise ValueError('Previously revealed inputs changed')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'selectedMetric': report['selectedMetric'], 'identicalOrdering': identical,
        'summary': {metric: result['summary'] for metric, result in results.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
