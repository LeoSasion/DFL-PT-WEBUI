import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'webui/python'))
from mask_source_binding import bind_batch, source_binding
from DFLIMG import DFLJPG


def fixture(tmp_path):
    frames, crops = tmp_path / 'frames', tmp_path / 'crops'
    frames.mkdir(); crops.mkdir()
    image = np.random.default_rng(8).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(frames / 'frame.png'), image)
    aligned = crops / 'face.jpg'
    cv2.imwrite(str(aligned), image)
    points = np.array([[10 + i % 20, 10 + i // 20] for i in range(68)], np.float32)
    dfl = DFLJPG.load(aligned)
    dfl.set_dict({'face_type': 'whole_face', 'source_filename': 'frame.png', 'landmarks': points,
                  'source_landmarks': points, 'image_to_face_mat': [[1, 0, 0], [0, 1, 0]]})
    dfl.save()
    return aligned, frames


def test_binding_preserves_originals_and_checks_exact_geometry(tmp_path):
    aligned, frames = fixture(tmp_path)
    before = aligned.read_bytes(), (frames / 'frame.png').read_bytes()
    result = source_binding(aligned, frames)['sourceFrame']
    assert result['maxGeometryErrorPx'] == 0
    assert result['alignedCanvasWh'] == [64, 64]
    assert result['sourceLandmarksSha256'] == result['alignedLandmarksSha256']
    assert (aligned.read_bytes(), (frames / 'frame.png').read_bytes()) == before
    dfl = DFLJPG.load(aligned)
    dfl.set_image_to_face_mat([[1, 0, 8], [0, 1, 0]])
    dfl.save()
    with pytest.raises(ValueError, match='do not match affine'):
        source_binding(aligned, frames)


def test_missing_original_is_explicit_and_duplicate_traversal_rejected(tmp_path):
    aligned, frames = fixture(tmp_path)
    (frames / 'frame.png').unlink()
    assert source_binding(aligned, frames) == {'sourceFrame': None, 'reason': 'original-source-frame-missing'}
    for names in (['../face.jpg'], ['face.jpg', 'FACE.jpg'], []):
        with pytest.raises(ValueError):
            bind_batch(aligned.parent, frames, names)
