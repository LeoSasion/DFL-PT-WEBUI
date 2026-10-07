"""Meaningful global coverage/identity/quality-floor selection checks."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal/DeepFaceLab'))
from core.best_training_faceset import global_training_selection


def row(member, score=80., yaw=0, eye='open-like', mouth='closed-like', embedding=None, time=None):
    embedding = np.eye(1, 128, dtype=np.float32)[0] if embedding is None else embedding
    result = {'member': member, 'sha256': hashlib.sha256(member.encode()).hexdigest(),
        'pixelSha256': hashlib.sha256(('pixels-' + member).encode()).hexdigest(),
        'hardReasons': [], 'reviewReasons': [], 'embedding': embedding.tolist(),
        'quality': {'model': 'foreground_tenengrad', 'score': score, 'minimumAcceptable': 5., 'provenance': {}},
        'coverage': {'yaw': yaw, 'pitch': 0, 'roll': 0, 'eyeBuckets': [eye, eye], 'mouthBucket': mouth,
                     'bucket': [round(yaw / 20), 0, eye, eye, mouth]}, 'appearance': {'brightness': .5}}
    if time is not None:
        result['timeline'] = {'sourceVideoSha256': 'a' * 64, 'pts': round(time * 1000), 'timeBase': [1, 1000], 'sourceFrameIndex': round(time * 25), 'shotId': 0}
    return result


def test_global_selection_keeps_adequate_rare_profiles_eyes_mouth_after_500_frontal_samples():
    majority = [row(f'front-{i:04d}.jpg', score=95.) for i in range(501)]
    rare = [row('profile.jpg', score=12., yaw=65), row('closed-eye.jpg', score=11., eye='closed-like'),
            row('open-mouth.jpg', score=10., mouth='open-like')]
    result = global_training_selection(majority + rare, 4, ['front-0000.jpg'])
    selected = {r['member'] for r in result['records'] if r['status'] == 'selected'}
    assert set(r['member'] for r in rare) <= selected
    assert len(selected) == 4 and result['selectedBuckets'] == 4
    assert result['counts']['rejected'] == 0
    assert all(r['classification'] == 'adequate-reserve' for r in result['records'] if r['status'] == 'review')


def test_no_identity_reference_never_turns_largest_group_into_truth():
    records = [row(f'{i}.jpg') for i in range(20)]
    result = global_training_selection(records, 10, [])
    assert result['counts'] == {'selected': 0, 'review': 20, 'rejected': 0}
    assert all('尚未确认目标身份参考图' in r['reasons'] for r in result['records'])


def test_low_similarity_and_below_floor_are_review_not_wrong_identity_rejections():
    stranger = row('uncertain.jpg', embedding=np.roll(np.eye(1, 128)[0], 1))
    profile = row('weak-profile.jpg', score=1., yaw=65)
    result = global_training_selection([row('reference.jpg'), stranger, profile], 3, ['reference.jpg'])
    assert result['counts'] == {'selected': 1, 'review': 2, 'rejected': 0}
    assert result['records'][1]['classification'] == result['records'][2]['classification'] == 'needs-review'


def test_continuous_duplicate_reduction_has_representative_and_keeps_rare_bucket():
    records = [row('best.jpg', 90., time=0), row('continuous.jpg', 70., time=.04),
               row('later.jpg', 80., time=1), row('rare-mouth.jpg', 10., mouth='open-like', time=.04)]
    result = global_training_selection(records, 3, ['best.jpg'])
    assert result['counts'] == {'selected': 2, 'review': 1, 'rejected': 1}
    duplicate = next(r for r in result['records'] if r['status'] == 'rejected')
    assert duplicate['classification'] == 'low-value-duplicate' and duplicate['duplicateOf'] == 'best.jpg'
    assert '不是检测或身份错误' in duplicate['reasons'][0]


def test_temporal_similarity_is_not_a_transitive_chain_that_discards_a_whole_clip():
    records = [row('anchor.jpg', 90., time=0), row('near.jpg', 80., time=.3), row('far.jpg', 70., time=.6)]
    result = global_training_selection(records, 3, ['anchor.jpg'])
    assert result['duplicateCount'] == 1
    assert next(r for r in result['records'] if r['member'] == 'far.jpg')['classification'] == 'adequate-reserve'


def test_inconsistent_references_and_severe_error_are_transparently_separate():
    bad = row('unreadable.jpg'); bad['hardReasons'] = ['无法读取']
    records = [row('a.jpg'), row('b.jpg', embedding=np.roll(np.eye(1, 128)[0], 1)), bad]
    result = global_training_selection(records, 3, ['a.jpg', 'b.jpg'])
    assert result['identityReferencesConflict']
    assert result['counts'] == {'selected': 0, 'review': 2, 'rejected': 1}
    assert result['records'][2]['classification'] == 'severe-error'


def test_efficient_fiqa_hook_is_unique_and_has_no_automatic_quality_cutoff():
    records = [row('reference.jpg'), row('profile.jpg', yaw=65)]
    for item in records:
        item['quality'] = {'model': 'efficient-fiqa', 'score': 2., 'minimumAcceptable': 0.,
                           'provenance': {'cutoffStatus': 'uncalibrated'}}
    assert global_training_selection(records, 2, ['reference.jpg'])['counts']['selected'] == 2
    records[1]['quality']['model'] = 'unapproved-model'
    result = global_training_selection(records, 2, ['reference.jpg'])
    assert result['counts']['review'] == 1


def test_results_are_deterministic_and_do_not_mutate_feature_inputs():
    records = [row(f'{i}.jpg', yaw=i * 5) for i in range(12)]
    original = copy.deepcopy(records)
    a = global_training_selection(records, 5, ['0.jpg'])
    b = global_training_selection(records, 5, ['0.jpg'])
    assert a == b and records == original
    with pytest.raises(ValueError): global_training_selection(records, True, ['0.jpg'])


def test_quantity_is_a_ceiling_and_500_near_identical_untimed_faces_do_not_fill_it():
    records = [row(f'front-{i:04d}.jpg', score=95., yaw=(i % 5) - 2) for i in range(500)]
    rare = [row('profile.jpg', score=12., yaw=65), row('mouth.jpg', score=10., mouth='open-like'),
            row('eye.jpg', score=11., eye='closed-like')]
    result = global_training_selection(records + rare, 400, ['front-0000.jpg'])
    selected = [item for item in result['records'] if item['status'] == 'selected']
    assert {item['member'] for item in rare} <= {item['member'] for item in selected}
    assert len(selected) == 4 < result['targetCount']
    assert result['minimumNewDiversity'] == .035
    assert result['counts']['rejected'] == 0
    assert all('连续' not in ' '.join(item['reasons']) for item in selected)
    assert all('质量 ' in item['reasons'][0] and '偏转 ' in item['reasons'][1] for item in selected)
    with pytest.raises(ValueError): global_training_selection(records, 400, ['front-0000.jpg'], minimum_new_diversity=0)
