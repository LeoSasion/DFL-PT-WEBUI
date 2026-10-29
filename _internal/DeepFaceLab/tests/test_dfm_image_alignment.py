"""Image-only consumers share alignment and retain crop/XSeg validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from facelib import FaceType, LandmarksProcessor
from me_backend import data
from me_backend.config import MEConfig
from dfm_export.quantization import calibration_faces
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture


@pytest.mark.parametrize('extension', [None, 'pak', 'zip'])
@pytest.mark.parametrize('face_type', ['f', 'wf'])
def test_image_only_alignment_matches_masked_alignment_without_mask_work(tmp_path, monkeypatch, extension, face_type):
    folder = make_aligned(tmp_path / 'aligned', count=2, size=256)
    if extension:
        pack_fixture(folder, extension)
    _, references, _ = data._load_inventory(folder)
    config = MEConfig(resolution=256, face_type=face_type)
    expected = [data.read_aligned(ref, config)[0] for ref in references]

    def unused(*args, **kwargs):
        raise AssertionError('An image-only consumer must not generate unused masks')

    for name in ['get_image_hull_mask', 'get_image_eye_mask', 'get_image_mouth_mask']:
        monkeypatch.setattr(LandmarksProcessor, name, unused)
    for reference, image in zip(references, expected):
        actual, full, encoded = data.read_aligned(reference, config, include_masks=False)
        np.testing.assert_array_equal(actual, image)
        assert actual.dtype == np.float32 and actual.shape == (256, 256, 3)
        assert full is encoded is None


def test_calibration_skips_unused_masks_and_reuses_decoded_payload(tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / 'aligned', count=8, size=256)
    config = MEConfig(resolution=256)
    expected = [data.read_aligned(path, config)[0] for path in sorted(folder.glob('*.jpg'))]
    original_decode, original_align = data.decode_aligned, data.read_aligned
    decodes, alignments = [], []

    def decode(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        decodes.append(result)
        return result

    def align(*args, **kwargs):
        assert kwargs['include_masks'] is False
        assert kwargs['decoded'] is decodes[-1]
        alignments.append(kwargs['decoded'])
        return original_align(*args, **kwargs)

    def unused(*args, **kwargs):
        raise AssertionError('Calibration does not consume training masks')

    monkeypatch.setattr(data, 'decode_aligned', decode)
    monkeypatch.setattr(data, 'read_aligned', align)
    for name in ['get_image_hull_mask', 'get_image_eye_mask', 'get_image_mouth_mask']:
        monkeypatch.setattr(LandmarksProcessor, name, unused)
    actual = list(calibration_faces(folder, config, 8))
    assert len(decodes) == len(alignments) == len(actual) == 8
    for (tensor, _), image in zip(actual, expected):
        np.testing.assert_array_equal(tensor.numpy(), image[None])


@pytest.mark.parametrize('include_masks', [False, True])
@pytest.mark.parametrize('invalid,error', [
    ('image', 'invalid BGR image'), ('landmark-shape', 'finite 68-point'),
    ('landmark-nan', 'finite 68-point'), ('mark-only', 'mark_only'),
    ('xseg-nan', 'invalid XSeg mask'), ('xseg-channels', 'invalid XSeg mask'),
    ('xseg-empty', 'invalid XSeg mask'),
])
def test_image_only_alignment_retains_input_rejections(tmp_path, include_masks, invalid, error):
    folder = make_aligned(tmp_path / 'aligned', count=1, size=256)
    path = folder / '000.jpg'
    original = data.decode_aligned(path)
    image, landmarks, mask, face_type = original.image, original.dfl.get_landmarks(), None, original.face_type
    if invalid == 'image':
        image = image[..., 0]
    elif invalid == 'landmark-shape':
        landmarks = landmarks[:67]
    elif invalid == 'landmark-nan':
        landmarks = np.asarray(landmarks, np.float32).copy()
        landmarks[0, 0] = np.nan
    elif invalid == 'mark-only':
        face_type = FaceType.MARK_ONLY
    elif invalid == 'xseg-nan':
        mask = np.full((16, 16, 1), np.nan, np.float32)
    elif invalid == 'xseg-channels':
        mask = np.zeros((16, 16, 3), np.float32)
    elif invalid == 'xseg-empty':
        mask = np.zeros((0, 16, 1), np.float32)
    dfl = SimpleNamespace(get_landmarks=lambda: landmarks, get_xseg_mask=lambda: mask,
        get_eyebrows_expand_mod=lambda: 1.)
    decoded = SimpleNamespace(name=str(path), image=image, face_type=face_type, dfl=dfl, packed=None)
    with pytest.raises(ValueError, match=error):
        data.read_aligned(path, MEConfig(resolution=256), decoded=decoded, include_masks=include_masks)


@pytest.mark.parametrize('include_masks', [False, True])
@pytest.mark.parametrize('eyebrows', [None, 'invalid'])
def test_image_only_keeps_hull_eyebrow_metadata_rejection(tmp_path, include_masks, eyebrows):
    folder = make_aligned(tmp_path / 'aligned', count=1, size=256)
    original = data.decode_aligned(folder / '000.jpg')
    original.dfl.dfl_dict['eyebrows_expand_mod'] = eyebrows
    with pytest.raises(TypeError):
        data.read_aligned(folder / '000.jpg', MEConfig(resolution=256),
            decoded=original, include_masks=include_masks)
