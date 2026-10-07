"""Global training-subset selection: adequate quality, then coverage/diversity.

This module does not infer identity from a largest cluster and never modifies
images. Anonymous SFace descriptors are compared only to explicitly confirmed
reference samples. Thresholds are conservative development policies, not
calibrated probabilities or annotation-accuracy measurements.
"""
from collections import defaultdict
from bisect import bisect_left, bisect_right, insort
from fractions import Fraction
import math

import numpy as np

PAIR_BATCH_LIMIT = 250
QUALITY_MODELS = ('foreground_tenengrad', 'efficient-fiqa')


def algorithm_suggestion(record):
    """Keep the algorithm classification distinct from subsequent human choices."""
    existing = record.get('algorithmSuggestion')
    if existing is not None:
        return {key: existing[key] for key in ('status', 'classification', 'reasons', 'duplicateOf') if key in existing}
    return {key: record[key] for key in ('status', 'classification', 'reasons', 'duplicateOf') if key in record}


def review_eligibility(record, confirmed_reference_members, *, identity_threshold=.50, references_conflict=False):
    """A manual keep cannot bypass input, identity or quality adequacy gates.

    Alignment/detector soft diagnostics can be explicitly judged by a human.
    Descriptor similarity remains a development policy, never identity truth.
    """
    blocked = list(record.get('hardReasons') or [])
    identity = record.get('identity') or {}
    if not confirmed_reference_members or references_conflict:
        blocked.append('目标身份参照尚未确认或相互冲突，请重新生成选集')
    elif (not identity or identity.get('method') != 'SFace-to-human-confirmed-references'
          or not isinstance(identity.get('maximumSimilarity'), (int, float))
          or isinstance(identity.get('maximumSimilarity'), bool)
          or not math.isfinite(identity['maximumSimilarity'])
          or (record['member'] not in confirmed_reference_members and identity['maximumSimilarity'] < identity_threshold)):
        blocked.append('身份尚不可验证；普通保留不能替代同人参照确认')
    try:
        score, floor = _quality(record)
        if score < floor:
            blocked.append('质量未达到保底要求，不能直接纳入训练')
    except (ValueError, KeyError, TypeError):
        blocked.append('没有有效的质量与保底证据')
    coverage = record.get('coverage') or {}
    if (not coverage.get('bucket') or not isinstance(coverage.get('eyeBuckets'), list)
            or len(coverage['eyeBuckets']) != 2 or not coverage.get('mouthBucket')
            or any(isinstance(coverage.get(key), bool) or not isinstance(coverage.get(key), (int, float))
                   or not math.isfinite(coverage[key]) for key in ('yaw', 'pitch', 'roll'))):
        blocked.append('缺少有效姿态或原生98点覆盖证据')
    warnings = list(record.get('reviewReasons') or [])
    if algorithm_suggestion(record).get('classification') == 'low-value-duplicate':
        warnings.append('算法判断为低价值重复；人工保留会增加相似素材')
    return {'canKeep': not blocked, 'blockedReasons': list(dict.fromkeys(blocked)),
            'warnings': list(dict.fromkeys(warnings))}


def manual_review_decision(record, decision, *, revision, request_id, created_at):
    if decision not in ('keep', 'exclude', 'defer'):
        raise ValueError('Manual review decision must be keep, exclude or defer')
    status = {'keep': 'selected', 'exclude': 'rejected', 'defer': 'review'}[decision]
    label = {'keep': '人工保留', 'exclude': '人工排除', 'defer': '人工暂缓，留待再次复核'}[decision]
    suggestion = algorithm_suggestion(record)
    return {**record, 'status': status, 'classification': 'manual-' + decision,
            'reasons': [label, *suggestion.get('reasons', [])][:3], 'algorithmSuggestion': suggestion,
            'manualDecision': {'decision': decision, 'revision': revision, 'requestId': request_id, 'createdAt': created_at}}


def manual_review_summary(records, *, can_undo=False):
    counts = {choice: 0 for choice in ('keep', 'exclude', 'defer')}
    pending = reserves = 0
    for record in records:
        choice = (record.get('manualDecision') or {}).get('decision')
        if choice in counts:
            counts[choice] += 1
        suggestion = algorithm_suggestion(record)
        if suggestion.get('classification') == 'needs-review' and choice not in ('keep', 'exclude'):
            pending += 1
        if suggestion.get('classification') == 'adequate-reserve' and choice is None:
            reserves += 1
    return {'pendingCount': pending, 'deferredCount': counts['defer'], 'adequateReserveCount': reserves,
            'decidedCount': counts['keep'] + counts['exclude'], 'manualCounts': counts, 'canUndo': can_undo}


def normalized_embedding(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size != 128 or not np.isfinite(vector).all():
        raise ValueError('A finite 128-dimensional SFace descriptor is required')
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('SFace descriptor has no valid norm')
    return vector / norm


def expression_coverage(points98, angles):
    points = np.asarray(points98, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if points.shape != (98, 2) or not np.isfinite(points).all() or min(np.ptp(points, axis=0)) <= 1:
        raise ValueError('Coverage needs valid native WFLW98 geometry')
    if angles.shape != (3,) or not np.isfinite(angles).all() or np.max(np.abs(angles)) > math.pi / 2 + 1e-6:
        raise ValueError('Coverage needs bounded existing-68 PnP pose')
    def aspect(region):
        eye = points[region:region + 8]
        denominator = 3 * float(np.linalg.norm(eye[0] - eye[4]))
        if denominator < 1:
            raise ValueError('Native eye geometry is degenerate')
        return float(sum(np.linalg.norm(eye[i] - eye[8 - i]) for i in (1, 2, 3)) / denominator)
    left, right = aspect(60), aspect(68)
    denominator = float(np.linalg.norm(points[76] - points[82]))
    if denominator < 1:
        raise ValueError('Native mouth geometry is degenerate')
    mouth = float(np.linalg.norm(points[90] - points[94]) / denominator)
    if max(left, right, mouth) > 2:
        raise ValueError('Native expression geometry is inconsistent')
    pitch, yaw, roll = map(math.degrees, angles)
    eyes = ['closed-like' if value < .16 else 'open-like' for value in (left, right)]
    mouth_state = 'closed-like' if mouth < .10 else 'part-open-like' if mouth < .25 else 'open-like'
    return {'poseSource': 'existing-68-landmarks-PnP', 'expressionSource': 'native-TUFA98-geometric-proxies',
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'eyeAspectRatios': [left, right], 'mouthAspectRatio': mouth,
            'eyeBuckets': eyes, 'mouthBucket': mouth_state,
            'bucket': [int(round(yaw / 20)), int(round(pitch / 20)), *eyes, mouth_state],
            'expressionLabelsVerified': False, 'occlusionEstimated': False}


def _quality(row):
    quality = row['quality']
    if quality.get('model') not in QUALITY_MODELS:
        raise ValueError('Only the reviewed detail baseline or Efficient-FIQA may score quality')
    score, floor = quality.get('score'), quality.get('minimumAcceptable')
    if (isinstance(score, bool) or isinstance(floor, bool) or not isinstance(score, (int, float))
            or not isinstance(floor, (int, float)) or not math.isfinite(score) or not math.isfinite(floor)):
        raise ValueError('Quality score and explicit adequacy floor must be finite')
    if not 0 <= score <= 100 or not 0 <= floor <= 100:
        raise ValueError('Quality score and adequacy floor use the shared 0..100 display scale')
    return float(score), float(floor)


def _time(row):
    timeline = row.get('timeline')
    if not timeline:
        return None
    pts, base = timeline.get('pts'), timeline.get('timeBase')
    if type(pts) is not int or not isinstance(base, list) or len(base) != 2 or any(type(v) is not int or v <= 0 for v in base):
        raise ValueError('Temporal evidence requires integer PTS and a positive rational time base')
    return float(pts * Fraction(*base))


def _distance(left, right):
    a, b = left['coverage'], right['coverage']
    pose = min(1., ((a['yaw'] - b['yaw']) / 120) ** 2 + ((a['pitch'] - b['pitch']) / 90) ** 2)
    expression = (sum(x != y for x, y in zip(a['eyeBuckets'], b['eyeBuckets']))
                  + (a['mouthBucket'] != b['mouthBucket'])) / 3
    descriptor = float(np.clip((1 - np.dot(left['_embedding'], right['_embedding'])) / .25, 0, 1))
    illumination = min(1., abs(left.get('appearance', {}).get('brightness', .5) - right.get('appearance', {}).get('brightness', .5)) / .3)
    return .45 * pose + .30 * expression + .15 * descriptor + .10 * illumination


def _distance_batch(rows, reference):
    if len(rows) > PAIR_BATCH_LIMIT:
        raise ValueError('Paired comparison block exceeds 250')
    a = reference['coverage']
    yaw = np.asarray([row['coverage']['yaw'] for row in rows])
    pitch = np.asarray([row['coverage']['pitch'] for row in rows])
    pose = np.minimum(1., ((yaw - a['yaw']) / 120) ** 2 + ((pitch - a['pitch']) / 90) ** 2)
    expression = np.asarray([(sum(x != y for x, y in zip(row['coverage']['eyeBuckets'], a['eyeBuckets']))
                             + (row['coverage']['mouthBucket'] != a['mouthBucket'])) / 3 for row in rows])
    descriptor = np.clip((1 - np.stack([row['_embedding'] for row in rows]) @ reference['_embedding']) / .25, 0, 1)
    brightness = np.asarray([row.get('appearance', {}).get('brightness', .5) for row in rows])
    illumination = np.minimum(1., np.abs(brightness - reference.get('appearance', {}).get('brightness', .5)) / .3)
    return .45 * pose + .30 * expression + .15 * descriptor + .10 * illumination


def global_training_selection(records, target_count, confirmed_reference_members, *, identity_threshold=.50,
                              duplicate_similarity=.985, temporal_window_seconds=.35, minimum_new_diversity=.035):
    """Return full selected/review/rejected decisions for one complete inventory.

    Paired descriptor comparisons are traversed in <=250-item blocks. There
    is no independent per-500 selection and no top-N truncation. Candidates
    failing only the adequacy floor or uncertain identity remain in review;
    rejected is reserved for explicit severe failures and labeled redundancy.
    """
    if type(target_count) is not int or target_count < 1:
        raise ValueError('Training target count must be a positive integer')
    if not .30 <= identity_threshold <= .85 or not .9 <= duplicate_similarity <= 1 or not 0 <= temporal_window_seconds <= 10:
        raise ValueError('Identity/redundancy development policy is invalid')
    if not .005 <= minimum_new_diversity <= .30:
        raise ValueError('Minimum new diversity must be in .005..0.30; zero would fill the quantity ceiling')
    rows = []
    by_member = {}
    for record in records:
        row = {**record, 'reasons': list(record.get('hardReasons', [])), 'classification': None}
        if row['member'] in by_member:
            raise ValueError('Duplicate inventory member')
        by_member[row['member']] = row
        rows.append(row)
        if row['reasons']:
            row.update(status='rejected', classification='severe-error')
            continue
        row['reasons'] = list(record.get('reviewReasons', []))
        try:
            row['_embedding'] = normalized_embedding(row['embedding'])
            score, floor = _quality(row)
            row['_qualityOrder'] = score - floor
            if score < floor:
                row['reasons'].append('质量未达保底要求，先人工复核')
            if not row.get('coverage'):
                row['reasons'].append('缺少有效姿态或原生98点覆盖证据')
        except (ValueError, KeyError, TypeError) as error:
            row['reasons'].append('质量或人脸特征不可用，先复核：' + str(error))
    references = list(dict.fromkeys(confirmed_reference_members))
    if len(references) > 32 or any(member not in by_member for member in references):
        raise ValueError('Confirm 1–32 reference members from this inventory')
    reference_rows = [by_member[member] for member in references]
    invalid_reference = any(row.get('status') == 'rejected' or '_embedding' not in row for row in reference_rows)
    reference_conflict = False
    if not invalid_reference:
        reference_conflict = any(float(np.dot(a['_embedding'], b['_embedding'])) < .30
                                 for i, a in enumerate(reference_rows) for b in reference_rows[:i])
    eligible = []
    for row in rows:
        if row.get('status') == 'rejected':
            continue
        if not references:
            row['reasons'].append('尚未确认目标身份参考图')
        elif invalid_reference or reference_conflict:
            row['reasons'].append('目标参考不可用或相互冲突，请重新确认')
        elif '_embedding' in row:
            similarity = max(float(np.dot(row['_embedding'], reference['_embedding'])) for reference in reference_rows)
            row['identity'] = {'method': 'SFace-to-human-confirmed-references', 'maximumSimilarity': similarity,
                               'threshold': identity_threshold, 'confirmedReference': row['member'] in references,
                               'identityTruthInferred': False}
            if row['member'] not in references and similarity < identity_threshold:
                row['reasons'].append('与目标参考的相似度不足，身份需复核')
        if row['reasons']:
            row.update(status='review', classification='needs-review')
        else:
            eligible.append(row)
    # Exact decoded-pixel duplicates and time-near same-coverage copies are
    # redundant. Compare to retained representatives, not a transitive chain.
    ordered = sorted(eligible, key=lambda row: (-row['_qualityOrder'], row['member'].casefold(), row['member']))
    kept, pixels, temporal = [], {}, defaultdict(list)
    duplicate_count = 0
    for row in ordered:
        duplicate = pixels.get(row.get('pixelSha256')) if row.get('pixelSha256') else None
        timeline = row.get('timeline')
        temporal_key = None
        if not duplicate and timeline:
            temporal_key = (timeline['sourceVideoSha256'], timeline.get('shotId', 'unknown'), tuple(row['coverage']['bucket']))
            center = _time(row)
            indexed = temporal[temporal_key]
            first = bisect_left(indexed, (center - temporal_window_seconds, ''))
            last = bisect_right(indexed, (center + temporal_window_seconds, chr(0x10ffff)))
            candidates = [by_member[member] for _, member in indexed[first:last]]
            for start in range(0, len(candidates), PAIR_BATCH_LIMIT):
                for representative in candidates[start:start + PAIR_BATCH_LIMIT]:
                    if (abs(_time(row) - _time(representative)) <= temporal_window_seconds
                            and float(np.dot(row['_embedding'], representative['_embedding'])) >= duplicate_similarity
                            and _distance(row, representative) < .04):
                        duplicate = representative; break
                if duplicate: break
        if duplicate:
            row.update(status='rejected', classification='low-value-duplicate', duplicateOf=duplicate['member'],
                       reasons=['低价值重复，已保留同组质量代表；不是检测或身份错误'])
            duplicate_count += 1
        else:
            kept.append(row)
            if row.get('pixelSha256'): pixels[row['pixelSha256']] = row
            if timeline:
                temporal_key = (timeline['sourceVideoSha256'], timeline.get('shotId', 'unknown'), tuple(row['coverage']['bucket']))
                insort(temporal[temporal_key], (_time(row), row['member']))
    buckets = defaultdict(list)
    for row in kept:
        buckets[tuple(row['coverage']['bucket'])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (-row['_qualityOrder'], row['member'].casefold(), row['member']))
    selected = []
    # First maximize occupied coverage. A slightly lower adequate-quality
    # rare profile/open-mouth/closed-eye remains ahead of another frontal copy.
    representatives = [bucket[0] for bucket in buckets.values()]
    while representatives and len(selected) < target_count:
        if not selected:
            winner = min(representatives, key=lambda row: (-row['_qualityOrder'], row['member'].casefold()))
        else:
            winner = max(representatives, key=lambda row: (min(_distance(row, old) for old in selected),
                1 / len(buckets[tuple(row['coverage']['bucket'])]), row['_qualityOrder'], row['member']))
        selected.append(winner); representatives = [row for row in representatives if row is not winner]
    remaining = [row for row in kept if not any(row is item for item in selected)]
    # Continue across the entire accumulated inventory with farthest-first
    # diversity; update distances in visible <=250 paired-comparison blocks.
    distances = {row['member']: 1. for row in remaining}
    for start in range(0, len(remaining), PAIR_BATCH_LIMIT):
        block = remaining[start:start + PAIR_BATCH_LIMIT]
        for item in selected:
            for row, value in zip(block, _distance_batch(block, item)):
                distances[row['member']] = min(distances[row['member']], float(value))
    while remaining and len(selected) < target_count:
        winner = max(remaining, key=lambda row: (distances[row['member']], row['_qualityOrder'], row['member']))
        if distances[winner['member']] < minimum_new_diversity:
            break
        selected.append(winner); remaining = [row for row in remaining if row is not winner]
        for start in range(0, len(remaining), PAIR_BATCH_LIMIT):
            block = remaining[start:start + PAIR_BATCH_LIMIT]
            for row, value in zip(block, _distance_batch(block, winner)):
                distances[row['member']] = min(distances[row['member']], float(value))
    selected_members = {row['member'] for row in selected}
    for row in kept:
        if row['member'] in selected_members:
            coverage = row['coverage']
            population = len(buckets[tuple(coverage['bucket'])])
            closed_eyes = coverage['eyeBuckets'].count('closed-like')
            eyes = ('双眼张开', '单眼闭合', '双眼闭合')[closed_eyes]
            mouth = {'closed-like': '闭嘴', 'part-open-like': '微张嘴', 'open-like': '张嘴'}[coverage['mouthBucket']]
            geometry = f"头部偏转 {coverage['yaw']:+.0f}° / {mouth} / {eyes}（几何估计）"
            row.update(status='selected', classification='training-subset',
                reasons=[f"质量 {row['quality']['score']:.1f} 达保底 {row['quality']['minimumAcceptable']:.1f}",
                         geometry, f'非重复代表；同类候选 {population} 张' + ('（稀缺姿态或眼口状态）' if population <= 3 else '')])
        else:
            row.update(status='review', classification='adequate-reserve',
                reasons=[f"质量 {row['quality']['score']:.1f} 达标的备选图",
                         '新增多样性不足，保留供复核' if len(selected) < target_count else '已达数量上限，保留供复核'])
    counts = {status: sum(row['status'] == status for row in rows) for status in ('selected', 'review', 'rejected')}
    for row in rows:
        row.pop('_embedding', None); row.pop('_qualityOrder', None)
    return {'schemaVersion': 1, 'policy': 'quality-floor-before-global-diversity; no-largest-cluster-identity',
        'counts': counts, 'targetCount': target_count, 'eligibleBeforeRedundancy': len(eligible),
        'duplicateCount': duplicate_count, 'occupiedAdequateBuckets': len(buckets),
        'selectedBuckets': len({tuple(row['coverage']['bucket']) for row in selected}),
        'minimumNewDiversity': minimum_new_diversity, 'quantityPolicy': 'upper-limit; stop-when-no-adequate-new-diversity',
        'diversityMetric': 'pose .45 + native98 eye/mouth buckets .30 + SFace appearance .15 + brightness .10; not semantic truth',
        'pairedComparisonLimit': PAIR_BATCH_LIMIT, 'identityReferences': references,
        'identityReferencesConflict': invalid_reference or reference_conflict,
        'selectionIncompleteReason': '达到新增多样性停止门槛或没有更多质量达标且身份已确认的候选；数量仅为上限' if counts['selected'] < target_count else None,
        'records': rows, 'accuracy': None, 'qualityPercentiles': None}
