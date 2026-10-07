"""Quality proxies, coverage behavior and real recoverable window sorting."""
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.faceset_transaction import recover_transaction, sha256
from core.imagelib.face_quality import face_detail_signals, metadata_coverage, select_quality_coverage
from DFLIMG import DFLJPG
from facelib import LandmarksProcessor
from mainscripts import Sorter
sys.path.insert(0, str(ROOT / 'webui/python'))
from face_quality_review import quality_review


def points():
    value = LandmarksProcessor.landmarks_68_3D[:, :2].copy().astype(np.float32)
    value -= value.min(axis=0)
    return value / np.ptp(value, axis=0) * 150 + 48


def image():
    result = np.full((256, 256, 3), 128, np.uint8)
    cv2.ellipse(result, (128, 132), (75, 84), 0, 0, 360, (150, 150, 150), -1)
    for x in (88, 164):
        cv2.ellipse(result, (x, 110), (18, 7), 0, 0, 360, (25, 25, 25), 2)
        cv2.circle(result, (x, 110), 4, (5, 5, 5), -1)
    cv2.ellipse(result, (128, 167), (32, 12), 0, 0, 180, (20, 20, 20), 3)
    return result


def make_faces(directory, count=6):
    directory.mkdir()
    for index in range(count):
        target = directory / f'face-{index:02d}.jpg'
        pixels = image() if index % 2 == 0 else cv2.GaussianBlur(image(), (0, 0), 2.)
        assert cv2.imwrite(str(target), pixels, [cv2.IMWRITE_JPEG_QUALITY, 100])
        dfl = DFLJPG.load(target)
        dfl.set_dict({'face_type': 'whole_face', 'landmarks': points().tolist(),
                      'source_filename': f'source-{index:02d}.png',
                      'source_rect': [0, 0, 120 + index, 130 + index]})
        dfl.save()
        target.with_suffix('.jpg.landmarks.json').write_text(json.dumps({'original': index}), encoding='utf-8')
    return directory


def test_mask_edges_and_outside_background_cannot_create_foreground_detail():
    mask = np.zeros((256, 256), np.uint8); mask[50:206, 50:206] = 1
    plain = np.full((256, 256, 3), 128, np.uint8)
    patterned = plain.copy()
    checker = np.uint8(np.indices((256, 256)).sum(axis=0) % 2 * 255)
    patterned[mask == 0] = checker[mask == 0, None]
    baseline = face_detail_signals(plain, points(), foreground=mask)
    alternate = face_detail_signals(patterned, points(), foreground=mask)
    assert baseline['scores'] == alternate['scores'] == {'foreground_tenengrad': 0., 'foreground_brenner': 0.}


def test_controlled_blur_and_added_noise_do_not_beat_the_clean_face():
    plain = image()
    blurred = cv2.GaussianBlur(plain, (0, 0), 4.5)
    noisy = np.uint8(np.clip(blurred.astype(float) + np.random.default_rng(12).normal(0, 25, plain.shape), 0, 255))
    readings = [face_detail_signals(value, points()) for value in (plain, blurred, noisy)]
    for metric in ('foreground_tenengrad', 'foreground_brenner'):
        assert readings[0]['scores'][metric] > readings[1]['scores'][metric]
        assert readings[0]['scores'][metric] > readings[2]['scores'][metric]
    assert readings[2]['highFrequencyResidual'] > readings[0]['highFrequencyResidual']
    for reading in readings:
        json.dumps(reading, allow_nan=False)


def test_invalid_images_masks_landmarks_and_pose_fail_without_guessed_truth():
    for invalid in (np.full((256, 256, 3), .5), np.zeros((4, 4, 3), np.uint8)):
        with pytest.raises(ValueError): face_detail_signals(invalid, points())
    for invalid in (np.zeros((68, 2)), np.full((68, 2), np.nan), np.zeros((98, 2))):
        with pytest.raises(ValueError): face_detail_signals(image(), invalid)
    with pytest.raises(ValueError): face_detail_signals(image(), points(), foreground=np.zeros((256, 256)))
    with pytest.raises(ValueError): metadata_coverage(points(), (np.nan, 0, 0), canvas_size=256)
    coverage = metadata_coverage(points(), (0, 0, 0), canvas_size=256)
    assert not coverage['occlusionAvailable'] and not coverage['annotationAccuracyAvailable']


def test_exact_small_targets_keep_pose_and_eye_mouth_buckets_without_frontal_bias():
    samples = []
    buckets = [(5, 4, 'open-like', 'closed-like'), (5, 4, 'closed-like', 'closed-like'),
               (5, 4, 'open-like', 'open-like'), (0, 4, 'open-like', 'closed-like'),
               (11, 4, 'open-like', 'closed-like')]
    for index in range(15):
        bucket = buckets[index % len(buckets)]
        samples.append([f'{index}.jpg', 1. - index / 20, None, 0., 0., {'coverage': {'bucket': list(bucket)}}])
    for count in (1, 2, 3, 5, 8, 15, 2000):
        selected, rejected = select_quality_coverage(copy.deepcopy(samples), count)
        assert len(selected) == min(count, len(samples))
        assert len(selected) + len(rejected) == len(samples)
        assert len({row[0] for row in selected + rejected}) == len(samples)
    selected, _ = select_quality_coverage(copy.deepcopy(samples), 5)
    assert len({tuple(row[5]['coverage']['bucket']) for row in selected}) == 5
    assert all(row[5]['selectionReason'] == 'coverage-representative' for row in selected)
    with pytest.raises(ValueError): select_quality_coverage(samples, True)


def test_real_window_sort_preview_commit_and_recovery_preserve_outside_originals(tmp_path):
    directory = make_faces(tmp_path / 'aligned')
    before = {target.name: target.read_bytes() for target in directory.iterdir()}
    preview = Sorter.main(directory, 'quality-coverage', target_count=2, offset=1, limit=3, dry_run=True)
    assert {target.name: target.read_bytes() for target in directory.iterdir()} == before
    assert preview['dry_run'] and preview['receipt_path'] is None
    context = preview['details']['qualityCoverage']
    assert context['selectedRange'] == {'start': 2, 'end': 4, 'total': 6, 'offset': 1, 'limit': 3, 'count': 3}
    assert context['preservedOutsideCount'] == 3
    assert len(preview['details']['scores']) == 3 and not preview['details']['scores_truncated']
    receipt = Sorter.main(directory, 'quality-coverage', target_count=2, offset=1, limit=3)
    assert receipt['state'] == 'committed'
    assert len(receipt['entries']) == 12  # every image and its independent sidecar backup
    for name in ('face-00.jpg', 'face-04.jpg', 'face-05.jpg'):
        assert (directory / name).read_bytes() == before[name]
        assert (directory / (name + '.landmarks.json')).read_bytes() == before[name + '.landmarks.json']
    assert len(list(directory.glob('quality-000001-*.jpg'))) == 2
    for evidence in receipt['details']['scores']:
        assert evidence['sourceSha256'] == sha256(Path(receipt['archive_path']) / next(
            entry['backup'] for entry in receipt['entries'] if entry['source'] == evidence['source']))
        assert set(evidence['detail']['scores']) == {'foreground_tenengrad', 'foreground_brenner'}
        assert evidence['selectionReason']
    recovered = recover_transaction(receipt['receipt_path'])
    assert recovered['state'] == 'rolled_back'
    assert {target.name: target.read_bytes() for target in directory.iterdir()} == before


def test_bad_window_and_score_staleness_do_not_mutate_sources(tmp_path):
    directory = make_faces(tmp_path / 'aligned', 2)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    for args in ({'offset': -1}, {'offset': 2}, {'limit': 501}, {'limit': True}, {'quality_metric': 'accuracy'}):
        with pytest.raises(ValueError): Sorter.main(directory, 'quality-coverage', dry_run=True, **args)
    selected, rejected, preserved, context = Sorter.sort_quality_coverage(directory, target_count=1)
    target = Path(selected[0][0]); target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='评分后源图发生变化'):
        Sorter.final_process(directory, selected, rejected, dry_run=True, preserved_files=preserved, quality_context=context)
    assert not (tmp_path / 'aligned_trash').exists()
    assert all((directory / name).read_bytes() == raw for name, raw in before.items() if name != target.name)


def test_unreadable_batch_member_has_explicit_audit_reason_and_recoverable_archive(tmp_path):
    directory = make_faces(tmp_path / 'aligned', 2)
    invalid = directory / 'face-02.jpg'; invalid.write_bytes(b'not a jpeg')
    preview = Sorter.main(directory, 'quality-coverage', target_count=2, dry_run=True)
    entry = next(item for item in preview['details']['scores'] if item['source'] == invalid.name)
    assert not entry['available'] and not entry['selected'] and entry['reason']
    assert entry['sourceSha256'] == sha256(invalid)
    assert invalid.read_bytes() == b'not a jpeg'


def test_review_ranking_uses_face_detail_and_rejects_noise_as_useful_detail():
    class Metadata:
        def get_landmarks(self): return points()
    plain = image()
    noisy = np.uint8(np.clip(cv2.GaussianBlur(plain, (0, 0), 2.).astype(float)
                            + np.random.default_rng(14).normal(0, 25, plain.shape), 0, 255))
    # Old full-frame sharpness could assign the noisy image a higher signal.
    # Existing valid face geometry enables the reviewed foreground proxy.
    clear = quality_review(plain, {'sharpness': .4, 'brightness': .5}, Metadata())
    contaminated = quality_review(noisy, {'sharpness': 1., 'brightness': .5}, Metadata())
    assert clear['score'] > contaminated['score']
    assert clear['detailCandidates']['selectedMetric'] == 'foreground_tenengrad'
    assert contaminated['baselineSharpness'] == 1.
    assert not contaminated['occlusion']['available']
    json.dumps(contaminated, allow_nan=False)


def test_review_without_existing_geometry_retains_the_baseline_without_guessing():
    result = quality_review(image(), {'sharpness': .7, 'brightness': .5})
    assert not result['detailCandidates']['available']
    assert result['components']['sharpness']['score'] == .7
    assert not result['pose']['available'] and not result['occlusion']['available']
