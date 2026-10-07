"""Global plan lifecycle, evidence bindings and byte-preserving publication."""
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'webui/python'), str(ROOT / '_internal/DeepFaceLab')]
import best_training_faceset as backend
from DFLIMG import DFLJPG
from facelib import LandmarksProcessor
import dfl_asset_tool
from face_quality_review import quality_review


def geometry():
    angle = np.linspace(np.pi, 0, 17)
    jaw = np.column_stack((128 + 86 * np.cos(angle), 115 + 92 * np.sin(angle)))
    return np.concatenate((jaw, LandmarksProcessor.landmarks_2D * [155, 135] + [50, 65])).astype(np.float32)


def make_image(path, seed=1, metadata=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = rng.integers(35, 215, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    if metadata:
        dfl = DFLJPG.load(path)
        points = geometry()
        dfl.set_face_type('whole_face'); dfl.set_landmarks(points)
        dfl.set_source_filename('frame.png'); dfl.set_source_rect([0, 0, 256, 256])
        dfl.set_source_landmarks(points); dfl.set_image_to_face_mat([[1., 0., 0.], [0., 1., 0.]])
        dfl.save()
    return path


def files_sha(directory):
    return {p.relative_to(directory).as_posix(): backend.sha256_file(p) for p in directory.rglob('*') if p.is_file()}


def fake_analysis(ref, metadata, models, quality_scorer=None, timeline_records=None):
    raw, pixels, _ = backend._read(ref, metadata=False)
    yaw = 65 if 'rare' in ref.member else 0
    return {'member': ref.member, 'sha256': ref.sha256, 'pixelSha256': backend._pixel_hash(pixels),
            'metadataSha256': backend._metadata_hash(raw), 'hardReasons': [], 'reviewReasons': [],
            'embedding': [1.] + [0.] * 127,
            'quality': {'model': metadata['qualityModel'], 'score': 12. if yaw else 90., 'minimumAcceptable': 5., 'provenance': {}},
            'coverage': {'yaw': yaw, 'pitch': 0, 'roll': 0, 'eyeBuckets': ['open-like', 'open-like'],
                         'mouthBucket': 'closed-like', 'bucket': [round(yaw / 20), 0, 'open-like', 'open-like', 'closed-like']},
            'appearance': {'brightness': .5}, 'timeline': None}


def ready_plan(tmp_path, monkeypatch, count=3):
    source = tmp_path / 'source'
    for i in range(count): make_image(source / ('rare.jpg' if i == count - 1 else f'{i}.jpg'), seed=i)
    (source / '0.jpg.landmarks.json').write_text('{"original": true}', encoding='utf-8')
    monkeypatch.setattr(backend, 'analyze_image', fake_analysis)
    state = backend.create_plan(source, tmp_path / 'plans', target_count=100, confirmed_reference_members=['0.jpg'])
    plan = state['planDirectory']
    backend.analyze_batch(plan, models=object())
    state = backend.finalize_plan(plan)
    return source, plan, state


def test_accumulated_windows_cannot_finalize_early_or_fill_quantity_ceiling(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    for i in range(503): make_image(source / (f'front-{i:04d}.jpg' if i < 502 else 'rare.jpg'), seed=i, metadata=False)
    monkeypatch.setattr(backend, 'analyze_image', fake_analysis)
    state = backend.create_plan(source, tmp_path / 'plans', target_count=400, confirmed_reference_members=['front-0000.jpg'])
    plan = state['planDirectory']
    first = backend.analyze_batch(plan, offset=0, limit=500, models=object())
    assert first['completedCount'] == 500 and first['nextOffset'] == 500 and not first['globalReady']
    assert first['selectedRange'] == {'start': 1, 'end': 500, 'offset': 0, 'limit': 500, 'count': 500, 'total': 503}
    with pytest.raises(backend.PlanConflict, match='offset=500'): backend.finalize_plan(plan)
    backend.analyze_batch(plan, offset=500, limit=500, models=object())
    result = backend.finalize_plan(plan)
    assert result['selection']['counts'] == {'selected': 2, 'review': 501, 'rejected': 0}
    filtered = backend.inspect_plan(plan, status='selected', limit=1, offset=1)
    assert filtered['selectedRange']['total'] == 2 and filtered['items'][0]['member'] == 'rare.jpg'
    assert filtered['items'][0]['position'] == 502
    with pytest.raises(ValueError): backend.analyze_batch(plan, limit=501)


def test_dryrun_publish_recover_preserve_every_byte_and_sidecar(tmp_path, monkeypatch):
    source, plan, state = ready_plan(tmp_path, monkeypatch)
    before = files_sha(source)
    root = tmp_path / 'new-output'
    dry = backend.publish_plan(plan, root, dry_run=True)
    assert dry['counts'] == {'selected': 2, 'review': 1, 'rejected': 0}
    assert not root.exists() and dry['receiptPath'] is None
    published = backend.publish_plan(plan, root)
    batch = Path(published['outputDirectory'])
    receipt = json.loads(Path(published['receiptPath']).read_text(encoding='utf-8'))
    for entry in receipt['entries']:
        copied = batch / entry['destination']
        assert copied.read_bytes() == (source / entry['member']).read_bytes()
        assert backend._metadata_hash(copied.read_bytes()) == entry['metadataSha256']
        if entry.get('sidecarSha256'):
            assert copied.with_suffix('.jpg.landmarks.json').read_bytes() == (source / entry['member']).with_suffix('.jpg.landmarks.json').read_bytes()
    assert receipt['counts'] == state['selection']['counts'] and files_sha(source) == before
    assert backend.recover_publication(published['receiptPath'], dry_run=True)['conflicts'] == []
    withdrawn = backend.recover_publication(published['receiptPath'])
    assert withdrawn['state'] == 'withdrawn' and files_sha(source) == before
    for entry in receipt['entries']:
        assert (batch / 'withdrawn' / entry['destination']).read_bytes() == (source / entry['member']).read_bytes()
    assert backend.recover_publication(published['receiptPath'])['reused']
    assert backend.inspect_plan(plan)['state'] == 'withdrawn'
    with pytest.raises(backend.PlanConflict, match='已撤回'): backend.publish_plan(plan, root)


@pytest.mark.parametrize('sidecar', [False, True])
def test_source_or_sidecar_change_blocks_publication_without_moving_originals(tmp_path, monkeypatch, sidecar):
    source, plan, _ = ready_plan(tmp_path, monkeypatch)
    changed = source / ('0.jpg.landmarks.json' if sidecar else '0.jpg')
    changed.write_bytes(changed.read_bytes() + b'changed')
    with pytest.raises(backend.PlanConflict, match='已变化'): backend.publish_plan(plan, tmp_path / 'output')
    assert changed.exists() and not (tmp_path / 'output').exists()


def test_modified_or_added_publication_files_block_withdrawal(tmp_path, monkeypatch):
    source, plan, _ = ready_plan(tmp_path, monkeypatch)
    before = files_sha(source)
    publication = backend.publish_plan(plan, tmp_path / 'output')
    batch = Path(publication['outputDirectory'])
    extra = batch / 'review' / 'user-added.txt'; extra.write_text('retain me')
    assert backend.recover_publication(publication['receiptPath'], dry_run=True)['conflicts']
    with pytest.raises(backend.PlanConflict): backend.recover_publication(publication['receiptPath'])
    assert extra.is_file() and (batch / 'selected').is_dir() and files_sha(source) == before


def test_unconfirmed_identity_is_all_review_and_can_be_confirmed_after_features(tmp_path, monkeypatch):
    source = tmp_path / 'source'; make_image(source / 'reference.jpg')
    monkeypatch.setattr(backend, 'analyze_image', fake_analysis)
    plan = backend.create_plan(source, tmp_path / 'plans')['planDirectory']
    backend.analyze_batch(plan, models=object())
    assert backend.finalize_plan(plan)['selection']['counts']['review'] == 1
    confirmed = backend.confirm_references(plan, ['reference.jpg'])
    assert confirmed['globalReady'] and confirmed['selection'] is None
    assert backend.finalize_plan(plan)['selection']['counts']['selected'] == 1


def test_quality_worker_closes_when_an_image_aborts_the_batch(tmp_path, monkeypatch):
    source = tmp_path / 'source'; make_image(source / 'reference.jpg')
    plan = backend.create_plan(source, tmp_path / 'plans', quality_model='efficient-fiqa')['planDirectory']
    class Scorer:
        closed = False
        def close(self): self.closed = True
    scorer = Scorer()
    def fail(*args, **kwargs): raise RuntimeError('worker failed')
    monkeypatch.setattr(backend, 'analyze_image', fail)
    with pytest.raises(RuntimeError, match='worker failed'): backend.analyze_batch(plan, quality_scorer=scorer, models=object())
    assert scorer.closed and backend.inspect_plan(plan)['completedCount'] == 0


def test_corrupt_jpeg_missing_metadata_and_executable_metadata_are_opaque_rejected_copies(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'corrupt.jpg').write_bytes(b'not a jpeg')
    make_image(source / 'plain.jpg', metadata=False)
    sentinel = tmp_path / 'must-not-exist'
    class Executable:
        def __reduce__(self): return (eval, (f"__import__('pathlib').Path({str(sentinel)!r}).write_text('executed')",))
    payload = pickle.dumps(Executable(), protocol=4)
    raw = (source / 'plain.jpg').read_bytes()
    (source / 'unsafe.jpg').write_bytes(raw[:2] + b'\xff\xef' + struct.pack('>H', len(payload) + 2) + payload + raw[2:])
    before = files_sha(source)
    plan = backend.create_plan(source, tmp_path / 'plans')['planDirectory']
    backend.analyze_batch(plan, models=object())
    finalized = backend.finalize_plan(plan)
    assert finalized['selection']['counts'] == {'selected': 0, 'review': 0, 'rejected': 3}
    result = backend.publish_plan(plan, tmp_path / 'output')
    for name in before:
        assert (Path(result['outputDirectory']) / 'rejected' / name).read_bytes() == (source / name).read_bytes()
    assert not sentinel.exists() and files_sha(source) == before


@pytest.mark.parametrize('quality_model', ['foreground_tenengrad', 'efficient-fiqa'])
def test_corrupt_embedded_xseg_does_not_block_other_images_or_call_quality_model(tmp_path, quality_model):
    source = tmp_path / 'source'
    bad = make_image(source / 'bad-mask.jpg')
    make_image(source / 'good.jpg', seed=2)
    dfl = DFLJPG.load(bad); dfl.get_dict()['xseg_mask'] = b'not a compressed mask'; dfl.save()
    before = files_sha(source)
    calls = []
    def scorer(image, metadata):
        calls.append(True)
        return {'model': 'efficient-fiqa', 'score': 50., 'minimumAcceptable': 0., 'provenance': {}}
    class Models:
        def detect(self, rgb): return {'detections': []}
        def embedding(self, bgr, points): return {'embedding': [1.] + [0.] * 127, 'model': 'sface'}
    plan = backend.create_plan(source, tmp_path / 'plans', quality_model=quality_model)['planDirectory']
    analyzed = backend.analyze_batch(plan, models=Models(),
        quality_scorer=scorer if quality_model == 'efficient-fiqa' else None)
    assert analyzed['completedCount'] == 2 and analyzed['globalReady']
    result = backend.finalize_plan(plan)
    assert result['selection']['counts'] == {'selected': 0, 'review': 1, 'rejected': 1}
    item = next(item for item in result['items'] if item['member'] == 'bad-mask.jpg')
    assert item['status'] == 'rejected' and any('XSeg' in reason for reason in item['reasons'])
    assert len(calls) == (1 if quality_model == 'efficient-fiqa' else 0)
    if quality_model == 'foreground_tenengrad':
        published = backend.publish_plan(plan, tmp_path / 'output')
        assert (Path(published['outputDirectory']) / 'rejected/bad-mask.jpg').read_bytes() == bad.read_bytes()
        assert backend.recover_publication(published['receiptPath'], dry_run=True)['conflicts'] == []
    assert files_sha(source) == before


def test_default_quality_matches_exact_review_aggregate_and_fresh98_is_bound_to_current_pixels(tmp_path, monkeypatch):
    source = tmp_path / 'source'; make_image(source / 'reference.jpg')
    state = backend.create_plan(source, tmp_path / 'plans', confirmed_reference_members=['reference.jpg'])
    _, metadata = backend._load(state['planDirectory'])
    ref = backend._inventory(metadata)[0]
    points = geometry(); native = np.zeros((98, 2), dtype=np.float32)
    native[:] = [128, 128]; native[0] = [45, 45]; native[1] = [210, 220]
    angle = np.linspace(np.pi, -np.pi, 8, endpoint=False)
    for start, center in ((60, 93), (68, 163)):
        native[start:start + 8] = np.column_stack((center + 20 * np.cos(angle), 102 + 6 * np.sin(angle)))
    native[76], native[82], native[90], native[94] = [92, 176], [163, 176], [128, 173], [128, 178]
    class Models:
        fresh_calls = 0
        def detect(self, rgb): return {'detections': [{'confidence': .99, 'box_xyxy': [40, 60, 214, 222]}]}
        def landmarks(self, bgr, box):
            self.fresh_calls += 1
            return {'points68': points, 'native98': {'points_original': native, 'asset_identity': {'sha256': 'fixed'}}}
        def embedding(self, bgr, landmarks): return {'embedding': [1.] + [0.] * 127, 'model': 'sface'}
    monkeypatch.setattr(dfl_asset_tool, 'inspect_native_landmarks', lambda *args: {'available': False})
    models = Models(); feature = backend.analyze_image(ref, metadata, models)
    _, image, dfl = backend._read(ref)
    expected = quality_review(image, dfl_asset_tool.bounded_image_metrics(image, dfl.get_xseg_mask()), dfl,
                              pose_estimator=LandmarksProcessor.estimate_pitch_yaw_roll)
    assert feature['quality']['score'] == pytest.approx(expected['score'] * 100.)
    assert feature['quality']['minimumAcceptable'] == 5.
    assert feature['quality']['provenance']['sourceHash'] == ref.sha256
    assert feature['coverage']['expressionLabelsVerified'] is False
    assert models.fresh_calls == 1 and feature['native98']['inputSha256'] == ref.sha256
    assert not feature['hardReasons'] and not feature['reviewReasons']


def test_efficient_is_comparison_only_and_cannot_publish_training(tmp_path, monkeypatch):
    source = tmp_path / 'source'; make_image(source / 'reference.jpg')
    monkeypatch.setattr(backend, 'analyze_image', fake_analysis)
    state = backend.create_plan(source, tmp_path / 'plans', quality_model='efficient-fiqa', confirmed_reference_members=['reference.jpg'])
    plan = state['planDirectory']; backend.analyze_batch(plan, models=object(), quality_scorer=object())
    backend.finalize_plan(plan)
    with pytest.raises(backend.PlanConflict, match='仅供评测比较'): backend.publish_plan(plan, tmp_path / 'output', dry_run=True)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('landmarks', [None, 'malformed-landmarks'])
def test_missing_or_malformed_landmarks_are_rejected_without_aborting_inventory(tmp_path, landmarks):
    source = tmp_path / 'source'; path = make_image(source / 'bad-landmarks.jpg')
    dfl = DFLJPG.load(path)
    if landmarks is None: dfl.get_dict().pop('landmarks')
    else: dfl.get_dict()['landmarks'] = landmarks
    dfl.save()
    plan = backend.create_plan(source, tmp_path / 'plans')['planDirectory']
    backend.analyze_batch(plan, models=object())
    assert backend.finalize_plan(plan)['selection']['counts']['rejected'] == 1
    publication = backend.publish_plan(plan, tmp_path / 'output')
    assert (Path(publication['outputDirectory']) / 'rejected/bad-landmarks.jpg').read_bytes() == path.read_bytes()


def test_publication_is_recoverable_when_rename_commits_before_plan_state_save_fails(tmp_path, monkeypatch):
    source, plan, _ = ready_plan(tmp_path, monkeypatch)
    before = files_sha(source); save = backend._json
    def fail_plan_commit(path, value):
        if path.name == 'plan.json' and value.get('state') == 'published': raise OSError('simulated metadata write failure')
        save(path, value)
    monkeypatch.setattr(backend, '_json', fail_plan_commit)
    with pytest.raises(OSError): backend.publish_plan(plan, tmp_path / 'output')
    monkeypatch.setattr(backend, '_json', save)
    adopted = backend.publish_plan(plan, tmp_path / 'output')
    assert adopted['reused'] and backend.inspect_plan(plan)['state'] == 'published'
    assert backend.publish_plan(plan, tmp_path / 'output')['reused'] and files_sha(source) == before


def test_withdrawal_rolls_back_partial_rename_and_is_retryable(tmp_path, monkeypatch):
    source, plan, _ = ready_plan(tmp_path, monkeypatch)
    publication = backend.publish_plan(plan, tmp_path / 'output'); original = Path.rename
    def fail_second(path, target):
        if path.name == 'review' and Path(target).parent.name == 'withdrawn': raise OSError('simulated rename failure')
        return original(path, target)
    monkeypatch.setattr(Path, 'rename', fail_second)
    with pytest.raises(OSError): backend.recover_publication(publication['receiptPath'])
    batch = Path(publication['outputDirectory'])
    assert all((batch / status).is_dir() for status in ('selected', 'review', 'rejected'))
    assert not (batch / 'withdrawn').exists()
    monkeypatch.setattr(Path, 'rename', original)
    assert backend.recover_publication(publication['receiptPath'])['state'] == 'withdrawn'


def test_changed_original_frame_invalidates_previously_bound_temporal_evidence(tmp_path):
    frame_dir = tmp_path / 'frames'; frame_dir.mkdir(); frame = frame_dir / 'frame.png'; frame.write_bytes(b'original frame')
    image_sha = backend.sha256_file(frame)
    backend._json(frame_dir / 'frames.timeline.json', {'schemaVersion': 1, 'kind': 'extracted-frames',
        'source': {'sha256': 'a' * 64}, 'frames': [{'file': 'frame.png', 'imageSha256': image_sha}]})
    metadata = {'framesDirectory': str(frame_dir), 'frameManifestSha256': backend.sha256_file(frame_dir / 'frames.timeline.json')}
    rows = [{'sourceFilename': 'frame.png', 'timeline': {'sourceFrameIndex': 0, 'imageSha256': image_sha}}]
    backend._validate_feature_bindings(rows, metadata)
    frame.write_bytes(b'changed frame')
    with pytest.raises(backend.PlanConflict, match='原始帧SHA已变化'): backend._validate_feature_bindings(rows, metadata)
