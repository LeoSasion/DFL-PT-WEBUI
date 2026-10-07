"""Read-only binding of reviewed aligned masks to their original source frames."""
import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal' / 'DeepFaceLab'))
from DFLIMG import DFLIMG
from vision_assets import canonical_digest, file_sha256

NAME = re.compile(r'^[^<>:"/\\|?*\x00-\x1f]{1,220}\.(?:jpe?g|png)$', re.I)


def source_binding(aligned, frames):
    aligned, frames = Path(aligned), Path(frames)
    if aligned.is_symlink() or not aligned.is_file():
        raise ValueError('Reviewed aligned must be a plain file')
    dfl = DFLIMG.load(aligned)
    if dfl is None or not dfl.has_data():
        raise ValueError('DFL metadata required for reviewed mask')
    name = dfl.get_source_filename()
    if not isinstance(name, str) or not NAME.fullmatch(name):
        return {'sourceFrame': None, 'reason': 'source-frame-name-unavailable'}
    source = frames / name
    if not source.exists():
        return {'sourceFrame': None, 'reason': 'original-source-frame-missing'}
    if frames.is_symlink() or source.is_symlink() or not source.is_file() or source.resolve().parent != frames.resolve():
        raise ValueError('Source frame is not a plain contained file')
    source_hash, aligned_hash = file_sha256(source), file_sha256(aligned)
    image, crop = cv2.imread(str(source)), cv2.imread(str(aligned))
    if image is None or crop is None:
        raise ValueError('Original frame/aligned cannot be decoded')
    src = np.asarray(dfl.get_source_landmarks(), dtype=np.float64)
    dst = np.asarray(dfl.get_landmarks(), dtype=np.float64)
    matrix = np.asarray(dfl.get_image_to_face_mat(), dtype=np.float64)
    if src.shape != (68, 2) or dst.shape != (68, 2) or matrix.shape != (2, 3) or not all(np.isfinite(v).all() for v in (src, dst, matrix)):
        raise ValueError('Complete finite source/aligned geometry required')
    if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-8:
        raise ValueError('Degenerate source-to-aligned transform')
    error = float(np.max(np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)))
    if error > .1:
        raise ValueError('Stored source/aligned landmarks do not match affine')
    if file_sha256(source) != source_hash or file_sha256(aligned) != aligned_hash:
        raise ValueError('Input changed during source binding')
    return {'reason': None, 'sourceFrame': {'file': name, 'sha256': source_hash,
        'width': image.shape[1], 'height': image.shape[0], 'sourceToAlignedAffine': matrix.tolist(),
        'sourceLandmarksSha256': canonical_digest(src.tolist()), 'alignedLandmarksSha256': canonical_digest(dst.tolist()),
        'alignedCanvasWh': [crop.shape[1], crop.shape[0]], 'maxGeometryErrorPx': error}}


def bind_batch(aligned_root, frames_root, names):
    root = Path(aligned_root)
    if not isinstance(names, list) or not 1 <= len(names) <= 500 or len(set(n.casefold() for n in names if isinstance(n, str))) != len(names):
        raise ValueError('Select 1–500 distinct aligned files')
    result = []
    for name in names:
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError('Plain aligned filename required')
        result.append({'file': name, **source_binding(root / name, frames_root)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned-root', required=True)
    parser.add_argument('--frames-root', required=True)
    args = parser.parse_args()
    print(json.dumps(bind_batch(args.aligned_root, args.frames_root, json.load(sys.stdin)['names']), allow_nan=False))
