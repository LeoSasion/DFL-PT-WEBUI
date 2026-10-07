import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal/DeepFaceLab'))
from core.imagelib.color_transfer import reinhard_color_transfer, robust_lab_transfer, color_transfer_sot, color_transfer_idt
from merger.quality_merge import distance_feather, multiband_blend, load_reviewed_masks, project_reviewed_mask, sha256, points_digest
from merger.temporal_geometry import bind_timeline, stabilize_geometry, adjusted_affine
from DFLIMG import DFLIMG
from core.safe_pickle import loads as safe_loads
from merger.session_data import normalize_session, restore_frames, encode_session, save_session
from merger import FrameInfo, InteractiveMergerSubprocessor, MergerConfigMasked
import pickle


def test_rct_ignores_background_in_statistics_and_flat_channel_is_finite():
    rng = np.random.default_rng(1)
    a, b = rng.random((20, 20, 3), dtype=np.float32), rng.random((20, 20, 3), dtype=np.float32)
    mask = np.zeros((20, 20, 1), np.float32); mask[5:15, 5:15] = 1
    out = reinhard_color_transfer(a, b, mask, mask)
    a2, b2 = a.copy(), b.copy(); a2[mask[..., 0] == 0] = 0; b2[mask[..., 0] == 0] = 1
    other = reinhard_color_transfer(a2, b2, mask, mask)
    np.testing.assert_allclose(out[5:15, 5:15], other[5:15, 5:15], atol=1e-6)
    flat = np.ones_like(a) * .4
    assert np.isfinite(reinhard_color_transfer(flat, b, mask, mask)).all()
    np.testing.assert_allclose(reinhard_color_transfer(a, b, mask * 0, mask * 0), a, atol=.004)


@pytest.mark.parametrize('algorithm', [color_transfer_sot, color_transfer_idt])
def test_seed_is_repeatable_without_global_rng_mutation(algorithm):
    rng = np.random.default_rng(4); a, b = rng.random((12, 12, 3), dtype=np.float32), rng.random((12, 12, 3), dtype=np.float32)
    before = np.random.get_state()
    x, y = algorithm(a, b, seed=53), algorithm(a, b, seed=53)
    assert np.array_equal(x, y)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert np.isfinite(algorithm(np.ones_like(a)*.5, np.ones_like(a)*.5, seed=1)).all()


@pytest.mark.parametrize('quantile', [False, True])
def test_robust_color_candidate_is_finite_and_ignores_background(quantile):
    a = np.random.default_rng(6).random((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16, 1), np.float32); mask[4:12, 4:12] = 1
    b = a * .7
    x = robust_lab_transfer(a, b, mask, mask, quantile)
    b[mask[..., 0] == 0] = 1
    y = robust_lab_transfer(a, b, mask, mask, quantile)
    assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    np.testing.assert_allclose(x, y, atol=1e-6)


def test_feather_preserves_holes_and_scales_with_face_grid():
    mask = np.zeros((64, 64), np.float32); mask[8:56, 8:56] = 1; mask[25:35, 25:35] = 0
    out = distance_feather(mask, 5)
    assert np.all(out[mask == 0] == 0) and np.all(out <= mask)
    assert out[9, 9] < 1 and out[16, 16] == 1
    bigger = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    assert abs(distance_feather(bigger, 10)[18, 18] - out[9, 9]) < .11


@pytest.mark.parametrize('shape', [(31, 47), (64, 64)])
def test_multiband_exact_exterior_and_occluder_protection(shape):
    base = np.ones((*shape, 3), np.float32) * .2; replacement = np.ones_like(base) * .8
    mask = np.zeros((*shape, 1), np.float32); mask[8:-8, 8:-8] = 1; mask[15:18, 20:23] = 0
    out = multiband_blend(base, replacement, mask)
    assert out.shape == base.shape and np.isfinite(out).all()
    assert np.array_equal(out[mask[..., 0] == 0], base[mask[..., 0] == 0])
    np.testing.assert_allclose(multiband_blend(base, base, mask), base, atol=1e-6)


def records(centers, times=None, indexes=None, shots=None):
    times = times if times is not None else np.arange(len(centers)) / 25
    return [{'seconds': float(t), 'pts': int(t*1000000), 'timeBase': [1, 1000000], 'durationPts': 40000,
             'sourceFrameIndex': indexes[i] if indexes else i, 'segmentIndex': shots[i] if shots else 0,
             'faces': [{'center': list(center), 'scale': 100.} for center in faces]}
            for i, (t, faces) in enumerate(zip(times, centers))]


def test_uneven_pts_preserves_true_linear_motion_and_expression_points():
    times = np.array([0, .03, .08, .12, .17])
    data = records([[[50+20*t, 50+4*t]] for t in times], times)
    original = copy.deepcopy(data)
    stabilize_geometry(data, 100)
    for before, after in zip(original, data):
        face = after['faces'][0]
        np.testing.assert_allclose(face['usedCenter'], before['faces'][0]['center'], atol=1e-10)
        np.testing.assert_allclose(face['velocityPixelsPerSecond'], [20, 4], atol=1e-10)
        assert face['motionDeltaSeconds'] > 0
    matrix = np.array([[.5, .1, 2], [-.1, .5, 7]], np.float32)
    assert np.array_equal(adjusted_affine(matrix, data[2]['faces'][0]), matrix)


def test_geometry_jitter_is_reduced_but_correction_is_bounded():
    data = records([[[50 + i*2 + (-1)**i*1.5, 50]] for i in range(15)])
    stabilize_geometry(data, 100)
    raw = np.array([r['faces'][0]['rawCenter'][0] for r in data])
    used = np.array([r['faces'][0]['usedCenter'][0] for r in data])
    truth = 50 + np.arange(15)*2
    assert np.mean((used-truth)**2) < np.mean((raw-truth)**2)
    assert max(abs(used-raw)) <= 2.5 + 1e-8
    assert np.ptp(used) > 20


@pytest.mark.parametrize('scenario,reason', [('cut', 'shot-change'), ('index', 'source-index-gap'), ('missing', 'missing-face'), ('gap', 'pts-gap')])
def test_tracks_never_bridge_boundaries(scenario, reason):
    data = records([[[50, 50]]] * 7)
    position = 3
    if scenario == 'cut':
        data[position]['cutBefore'] = True
    elif scenario == 'index':
        for r in data[position:]: r['sourceFrameIndex'] += 4
    elif scenario == 'missing':
        data[position-1]['faces'] = []
    else:
        for r in data[position:]: r['seconds'] += .8
    stabilize_geometry(data, 100)
    assert data[position]['faces'][0]['breakReason'] == reason
    assert data[position]['faces'][0]['trackId'] != data[position-2]['faces'][0]['trackId']
    assert all(x >= data[position]['sourceFrameIndex'] for x in data[position]['faces'][0]['contributors'])


def test_bidirectional_unique_association_and_duplicate_rejection():
    data = records([[[20, 50], [70, 50]], [[45, 50]], [[20, 50], [70, 50]]])
    stabilize_geometry(data, 100)
    assert data[1]['faces'][0]['breakReason'] == 'ambiguous-geometry'
    assert data[1]['faces'][0]['contributors'] == []
    duplicate = records([[[50, 50], [50, 50]]] * 3)
    stabilize_geometry(duplicate, 100)
    assert all(f['trackId'] is None and f['motionPower'] == 0 for r in duplicate for f in r['faces'])


def test_timeline_sha_integer_pts_and_no_guessing(tmp_path):
    path = tmp_path / '00001.png'; cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
    bound, audit = bind_timeline([path]); assert bound == [None] and audit['motionDisabled']
    frame = {'file': path.name, 'imageSha256': sha256(path), 'pts': -5, 'timeBase': [1, 1000], 'sourceFrameIndex': 0}
    payload = {'schemaVersion': 1, 'kind': 'extracted-frames', 'source': {'sha256': 'a'*64}, 'frames': [frame]}
    marker = tmp_path/'frames.timeline.json'; marker.write_text(json.dumps(payload), encoding='utf-8')
    assert bind_timeline([path])[0][0]['seconds'] == -.005
    frame['pts'] = 1.5; marker.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='integer PTS'): bind_timeline([path])
    frame['pts'] = 1; marker.write_text(json.dumps(payload), encoding='utf-8'); path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'): bind_timeline([path])


def reviewed_fixture(tmp_path):
    aligned, source = tmp_path/'aligned', tmp_path/'source'; aligned.mkdir(); source.mkdir()
    path, frame = aligned/'00001_0.jpg', source/'00001.png'
    image = np.full((64, 64, 3), 150, np.uint8)
    cv2.imwrite(str(path), image); cv2.imwrite(str(frame), image)
    dfl = DFLIMG.load(path); points = np.arange(136).reshape(68, 2).astype(np.float32) / 3
    matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    dfl.set_landmarks(points); dfl.set_source_landmarks(points); dfl.set_source_filename(frame.name)
    dfl.set_image_to_face_mat(matrix); dfl.set_xseg_mask(np.pad(np.ones((32, 32, 1), np.float32), ((16, 16), (16, 16), (0, 0)))); dfl.save()
    binding = {'file': frame.name, 'sha256': sha256(frame), 'width': 64, 'height': 64,
               'sourceToAlignedAffine': matrix.tolist(), 'sourceLandmarksSha256': points_digest(points),
               'alignedLandmarksSha256': points_digest(points), 'alignedCanvasWh': [64, 64]}
    entry = {'file': path.name, 'assisted': True, 'outputSha256': sha256(path), 'sourceFrame': binding}
    receipt = {'schemaVersion': 1, 'side': 'dst', 'reviewed': True, 'status': 'published', 'entries': [entry]}
    marker = aligned/'mask-assist-provenance.json'; marker.write_text(json.dumps(receipt), encoding='utf-8')
    return aligned, source, path, frame, marker, receipt


def test_reviewed_mask_roundtrip_projection_and_changed_inputs(tmp_path):
    aligned, source, path, frame, marker, receipt = reviewed_fixture(tmp_path)
    record = load_reviewed_masks(aligned, source, [path.name])[path.name]
    projected = project_reviewed_mask(record, np.array([[1, 0, 5], [0, 1, 0]], np.float32), 64)
    assert projected[30, 20] == 0 and projected[30, 25] == 1
    frame.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): project_reviewed_mask(record, np.eye(2, 3), 64)
    with pytest.raises(ValueError, match='source frame changed'): load_reviewed_masks(aligned, source, [path.name])


@pytest.mark.parametrize('field', ['sourceToAlignedAffine', 'alignedCanvasWh', 'alignedLandmarksSha256', 'sourceFrame'])
def test_reviewed_mask_does_not_accept_stale_affine_or_missing_bindings(tmp_path, field):
    aligned, source, path, frame, marker, receipt = reviewed_fixture(tmp_path)
    if field == 'sourceFrame': receipt['entries'][0]['sourceFrame'] = None
    elif field == 'sourceToAlignedAffine': receipt['entries'][0]['sourceFrame'][field][0][2] = 5
    elif field == 'alignedCanvasWh': receipt['entries'][0]['sourceFrame'][field] = [128, 128]
    else: receipt['entries'][0]['sourceFrame'][field] = '0'*64
    marker.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError): load_reviewed_masks(aligned, source, [path.name])


def test_legacy_session_conversion_is_data_only_and_keeps_backup(tmp_path):
    frame = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png')))
    frame.cfg = MergerConfigMasked(); frame.idx = 0; frame.is_done = True
    raw = pickle.dumps({'frames': [frame], 'frames_idxs': [], 'frames_done_idxs': [0], 'model_iter': 12}, protocol=4)
    payload, legacy = normalize_session(safe_loads(raw, profile='session'))
    assert legacy and payload['sessionSchema'] == 2
    fresh = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png'), source_sha256='a'*64))
    fresh.frame_info.aligned_identities = [{'file': '00001_0.jpg', 'sha256': 'b'*64}]
    restored, recompute = restore_frames(payload, [fresh], MergerConfigMasked())
    assert recompute and restored[0].is_done is False and restored[0].cfg.mask_mode == 4
    path = tmp_path/'merger_session.dat'; path.write_bytes(raw)
    encoded = encode_session(restored, [0], [], 12, payload['conversionSource'])
    save_session(path, encoded, preserve_legacy=True)
    assert len(list(tmp_path.glob('*.bak'))) == 1 and next(tmp_path.glob('*.bak')).read_bytes() == raw
    assert normalize_session(safe_loads(path.read_bytes(), profile='session'))[1] is False


def test_data_session_resume_checks_source_sha_and_aligned_sha():
    frame = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png'), source_sha256='a'*64))
    frame.frame_info.aligned_identities = [{'file': 'face.jpg', 'sha256': 'b'*64}]
    frame.cfg, frame.is_done = MergerConfigMasked(), True
    payload, _ = normalize_session(encode_session([frame], [], [0], 2))
    restored, recompute = restore_frames(payload, [frame], MergerConfigMasked())
    assert not recompute and restored[0].is_done
    frame.frame_info.aligned_identities[0]['sha256'] = 'c'*64
    # The on-disk/readback session is independent of current mutable objects.
    payload, _ = normalize_session(safe_loads(pickle.dumps(payload), profile='session'))
    payload['frames'][0]['alignedIdentities'][0]['sha256'] = 'b'*64
    restored, recompute = restore_frames(payload, [frame], MergerConfigMasked())
    assert recompute and not restored[0].is_done


def test_bad_session_indices_config_and_reviewed_preflight_are_rejected():
    frame = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png'), source_sha256='a'*64))
    frame.cfg = MergerConfigMasked(mask_mode=10)
    payload = encode_session([frame], [0], [], 2)
    with pytest.raises(ValueError, match='reviewed DST mode'): restore_frames(payload, [frame], MergerConfigMasked())
    payload['frames_done_idxs'] = [0]
    with pytest.raises(ValueError, match='partition'): normalize_session(payload)
    payload['frames_done_idxs'] = []; payload['frames'][0]['cfg']['blur_mask_modifier'] = float('inf')
    with pytest.raises(ValueError, match='parameter'): normalize_session(payload)


def test_removed_enhancer_session_cannot_silently_use_new_model():
    frame = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png')))
    frame.cfg = MergerConfigMasked(super_resolution_power=50)
    payload = encode_session([frame], [0], [], 2)
    payload['frames'][0]['cfg']['super_resolution_model'] = 'legacy-dfl'
    with pytest.raises(ValueError, match='MambaIRv2'): normalize_session(payload)
    payload['frames'][0]['cfg']['super_resolution_power'] = 0
    normalized, _ = normalize_session(payload)
    assert normalized['frames'][0]['cfg']['super_resolution_model'] == 'mambairv2'


@pytest.mark.parametrize('extension', ['pak', 'zip'])
@pytest.mark.parametrize('change', ['pixels', 'landmarks'])
def test_packed_session_checks_member_bytes_and_original_geometry(tmp_path, monkeypatch, extension, change):
    from mainscripts.Merger import read_packed_alignment, prepare_merge_frames
    from samplelib import PackedFaceset
    from core.interact import interact as io

    aligned, source, path, frame, _, _ = reviewed_fixture(tmp_path)
    monkeypatch.setattr(io, 'input_str', lambda *_args, **_kwargs: '')
    cfg = MergerConfigMasked()
    def collect():
        packed = PackedFaceset.load(aligned)
        assert packed is not None and len(packed) == 1
        filename, dfl, digest = read_packed_alignment(packed[0])
        mapping = {frame.stem: [(dfl.get_source_landmarks(), filename, Path(dfl.get_source_filename()))]}
        return prepare_merge_frames([frame], mapping, cfg, aligned,
                                    packed_identities={filename.as_posix(): digest})[0]

    PackedFaceset.pack(aligned, ext=extension, delete_original=False)
    first = collect()
    first[0].cfg, first[0].is_done = cfg, True
    saved, _ = normalize_session(safe_loads(pickle.dumps(encode_session(first, [], [0], 2)), profile='session'))
    unchanged, recompute = restore_frames(saved, collect(), cfg)
    assert not recompute and unchanged[0].is_done
    before = first[0].frame_info.aligned_identities[0]
    assert before['sha256'] == sha256(path) and before['storage'] == 'packed'
    assert before['packedMember'] == path.name and before['sourceFilename'] == frame.name
    assert len(before['sourceLandmarksSha256']) == 64

    dfl = DFLIMG.load(path)
    if change == 'landmarks':
        dfl.set_source_landmarks(dfl.get_source_landmarks() + 4)
    else:
        metadata = copy.deepcopy(dfl.get_dict())
        assert cv2.imwrite(str(path), np.full((64, 64, 3), 80, np.uint8))
        dfl = DFLIMG.load(path)
        dfl.set_dict(metadata)
    dfl.save()
    PackedFaceset.pack(aligned, ext=extension, delete_original=False)
    changed = collect()
    current = changed[0].frame_info.aligned_identities[0]
    assert current['sha256'] != before['sha256']
    if change == 'landmarks':
        assert current['sourceLandmarksSha256'] != before['sourceLandmarksSha256']
    else:
        assert current['sourceLandmarksSha256'] == before['sourceLandmarksSha256']
    restored, recompute = restore_frames(saved, changed, cfg)
    assert recompute and not restored[0].is_done


def test_packed_same_basename_members_retain_distinct_content_bindings():
    from mainscripts.Merger import read_packed_alignment, alignment_identity
    from samplelib import Sample

    # These test member bytes remain inert; the DFL container can be unavailable
    # while the member identity still has to be distinct before session reuse.
    samples = [Sample(filename='face.jpg', person_name=name) for name in ('person-A', 'person-B')]
    class Member:
        def __init__(self, sample, raw):
            self.filename, self.person_name, self.raw = sample.filename, sample.person_name, raw
        def read_raw_file(self):
            return self.raw
    records = [read_packed_alignment(Member(sample, raw))
               for sample, raw in zip(samples, (b'member A', b'member B'))]
    identities = {filename.as_posix(): digest for filename, _, digest in records}
    points = np.arange(136).reshape(68, 2)
    bound = [alignment_identity((points, filename, Path('source.png')), identities, False)
             for filename, _, _ in records]
    assert len(identities) == 2 and bound[0]['sha256'] != bound[1]['sha256']
    assert [item['packedMember'] for item in bound] == ['person-A/face.jpg', 'person-B/face.jpg']


def test_old_packed_session_without_member_digest_cannot_reuse_outputs():
    frame = InteractiveMergerSubprocessor.Frame(frame_info=FrameInfo(filepath=Path('00001.png'), source_sha256='a'*64))
    frame.frame_info.aligned_identities = [{'file': 'face.jpg', 'sha256': None, 'storage': 'packed'}]
    frame.cfg, frame.is_done = MergerConfigMasked(), True
    payload, _ = normalize_session(encode_session([frame], [], [0], 2))
    restored, recompute = restore_frames(payload, [frame], MergerConfigMasked())
    assert recompute and not restored[0].is_done
