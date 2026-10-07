import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'webui/python'))
from mask_assist import inspect_aligned, prepare_draft
from DFLIMG import DFLJPG, DFLIMG


class CandidateFixture:
    provenance = {'frozen': True, 'newTraining': False, 'identity': {'id': 'fixture-not-an-official-model'}}

    def predict(self, image):
        labels = np.zeros(image.shape[:2], np.uint8)
        labels[12:-12, 14:-14] = 1
        labels[22:28, 22:30] = 4  # eye class must remain foreground
        labels[42:48, 25:39] = 11  # mouth class must remain foreground
        labels[:12] = 17  # hair must be excluded
        return {'labels': labels}


def aligned_fixture(folder, name='face.jpg', points=68):
    folder.mkdir(exist_ok=True)
    target = folder / name
    image = np.random.default_rng(9).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(target), image)
    dfl = DFLJPG.load(target)
    dfl.set_dict({'face_type': 'whole_face', 'source_filename': 'original-frame.png',
                  'source_rect': [1, 2, 90, 100], 'landmarks': [[20 + i % 20, 20 + i // 20] for i in range(points)],
                  'seg_ie_polys': {'polys': [{'type': 1, 'pts': [[12, 12], [30, 12], [30, 30]]}]}})
    dfl.save()
    return target


def test_draft_preserves_original_pixels_and_polygons_and_embeds_mask(tmp_path):
    source = aligned_fixture(tmp_path / 'aligned')
    before = source.read_bytes()
    original = DFLIMG.load(source)
    report = prepare_draft(source.parent, [source.name], tmp_path / 'draft', tmp_path / 'assets', candidate=CandidateFixture())
    assert source.read_bytes() == before
    copied = DFLIMG.load(tmp_path / 'draft/copies/face.jpg')
    assert np.array_equal(cv2.imread(str(source)), cv2.imread(str(tmp_path / 'draft/copies/face.jpg')))
    assert copied.get_dict()['seg_ie_polys'] == original.get_dict()['seg_ie_polys']
    assert copied.get_dict()['source_filename'] == 'original-frame.png'
    assert np.array_equal(copied.get_landmarks(), original.get_landmarks())
    mask = copied.get_xseg_mask()[:, :, 0] >= .5
    assert mask[24, 24] and mask[44, 30] and not mask[5, 30]
    assert copied.get_dict()['mask_assist']['humanGroundTruth'] is False
    assert report['semantics']['iou'] is None and report['semantics']['dice'] is None
    assert report['entries'][0]['embeddedMaskReadbackExact']


def test_raw_ffhq_or_98_point_images_are_not_treated_as_native_aligned(tmp_path):
    source = aligned_fixture(tmp_path / 'aligned', points=98)
    with pytest.raises(ValueError, match='68-point'):
        inspect_aligned(source)
    raw = tmp_path / 'raw.jpg'
    cv2.imwrite(str(raw), np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ValueError, match='metadata'):
        inspect_aligned(raw)
    full = aligned_fixture(tmp_path / 'full')
    dfl = DFLIMG.load(full)
    dfl.set_face_type('full_face')
    dfl.save()
    with pytest.raises(ValueError, match='whole-face'):
        inspect_aligned(full)


def test_draft_cannot_overwrite_source_dataset_or_use_traversal(tmp_path):
    source = aligned_fixture(tmp_path / 'aligned')
    with pytest.raises(ValueError, match='outside'):
        prepare_draft(source.parent, [source.name], source.parent / 'new', tmp_path / 'assets', candidate=CandidateFixture())
    with pytest.raises(ValueError, match='filenames'):
        prepare_draft(source.parent, ['../face.jpg'], tmp_path / 'draft', tmp_path / 'assets', candidate=CandidateFixture())


def test_partial_generation_never_claims_ready(tmp_path):
    source = aligned_fixture(tmp_path / 'aligned')
    class FailedCandidate:
        provenance = {}
        def predict(self, image):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError):
        prepare_draft(source.parent, [source.name], tmp_path / 'draft', tmp_path / 'assets', candidate=FailedCandidate())
    assert not (tmp_path / 'draft/report.json').exists()
