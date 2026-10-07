"""Auditable optional merge candidates; no global state or model downloads."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from DFLIMG import DFLIMG
from core.imagelib.color_transfer import robust_lab_transfer


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points_digest(points):
    return hashlib.sha256(json.dumps(np.asarray(points).tolist(), sort_keys=True,
                                    separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def load_reviewed_masks(aligned_path, input_path, required_names):
    """Receipt, aligned pixels, affine and original frame must all still agree."""
    aligned_path, input_path = Path(aligned_path), Path(input_path)
    marker = aligned_path / 'mask-assist-provenance.json'
    if not marker.is_file() or marker.is_symlink():
        raise ValueError('Reviewed DST masks require a published mask-assist receipt')
    receipt = json.loads(marker.read_text(encoding='utf-8'))
    if receipt.get('schemaVersion') != 1 or receipt.get('status') != 'published' or receipt.get('reviewed') is not True or receipt.get('side') != 'dst':
        raise ValueError('A reviewed published DST mask batch is required')
    entries = receipt.get('entries', [])
    if not isinstance(entries, list) or len(entries) > 100000:
        raise ValueError('Invalid mask publication entries')
    by_name = {entry['file']: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError('Duplicate reviewed mask filenames')
    result = {}
    for name in required_names:
        entry = by_name.get(name)
        if not entry or entry.get('assisted') is not True or not isinstance(entry.get('sourceFrame'), dict):
            raise ValueError(f'Reviewed DST mask/source binding missing: {name}; generate and review a new batch with original frames')
        path = aligned_path / name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or sha256(path) != entry.get('outputSha256'):
            raise ValueError(f'Reviewed aligned file changed: {name}')
        binding = entry['sourceFrame']
        source_name = binding.get('file', '')
        source = input_path / source_name
        if Path(source_name).name != source_name or source.is_symlink() or not source.is_file() or sha256(source) != binding.get('sha256'):
            raise ValueError(f'Reviewed source frame changed or unavailable: {source_name}')
        dfl = DFLIMG.load(path)
        if dfl is None or not dfl.has_data() or Path(dfl.get_source_filename() or '').name != source_name:
            raise ValueError(f'Reviewed DFL source metadata changed: {name}')
        landmarks = np.asarray(dfl.get_landmarks())
        source_points = np.asarray(dfl.get_source_landmarks())
        matrix = np.asarray(dfl.get_image_to_face_mat(), dtype=np.float64)
        canvas = list(reversed(dfl.get_shape()[:2]))
        image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != [binding.get('width'), binding.get('height')]:
            raise ValueError('Reviewed source frame canvas changed')
        if landmarks.shape != (68, 2) or source_points.shape != (68, 2) or matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-10:
            raise ValueError('Finite invertible reviewed affine and real 68-point coordinates required')
        if canvas != binding.get('alignedCanvasWh') or not np.array_equal(matrix, np.asarray(binding.get('sourceToAlignedAffine'))):
            raise ValueError('Reviewed mask alignment or canvas changed')
        if points_digest(landmarks) != binding.get('alignedLandmarksSha256') or points_digest(source_points) != binding.get('sourceLandmarksSha256'):
            raise ValueError('Reviewed mask landmarks changed')
        projected = source_points @ matrix[:, :2].T + matrix[:, 2]
        if not np.isfinite(projected).all() or np.max(np.abs(projected - landmarks)) > 2.0:
            raise ValueError('Reviewed affine is inconsistent with aligned landmarks')
        mask = dfl.get_xseg_mask()
        if mask is None or mask.shape[:2] != tuple(reversed(canvas)) or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('Reviewed embedded mask is missing or has changed coordinates')
        result[name] = {'mask': mask[..., 0].astype(np.float32), 'sourceToAlignedAffine': matrix,
                        'alignedPath': str(path), 'alignedSha256': entry['outputSha256'],
                        'sourceFile': source_name, 'sourcePath': str(source), 'sourceSha256': binding['sha256'],
                        'receiptPath': str(marker), 'receiptSha256': sha256(marker)}
    return result


def project_reviewed_mask(record, source_to_predictor, size):
    # Re-check immediately before each prediction; workers must not consume stale data.
    if sha256(record['alignedPath']) != record['alignedSha256'] or sha256(record['receiptPath']) != record['receiptSha256'] or sha256(record['sourcePath']) != record['sourceSha256']:
        raise ValueError('Reviewed mask inputs changed after merge preflight')
    aligned_to_source = cv2.invertAffineTransform(record['sourceToAlignedAffine'])
    transform = np.vstack((source_to_predictor, [0, 0, 1])) @ np.vstack((aligned_to_source, [0, 0, 1]))
    return np.clip(cv2.warpAffine(record['mask'], transform[:2], (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1)


def merge_color(mode, predicted, destination, mask, seed):
    from core import imagelib
    if mode == 0:
        return predicted
    if mode == 1:
        return imagelib.reinhard_color_transfer(predicted, destination, target_mask=mask, source_mask=mask)
    if mode == 2:
        return imagelib.linear_color_transfer(predicted, destination)
    if mode in (3, 4):
        return imagelib.color_transfer_mkl(predicted if mode == 3 else predicted * mask, destination if mode == 3 else destination * mask)
    if mode in (5, 6):
        return imagelib.color_transfer_idt(predicted if mode == 5 else predicted * mask, destination if mode == 5 else destination * mask, seed=seed)
    if mode == 7:
        return np.clip(imagelib.color_transfer_sot(predicted * mask, destination * mask, steps=10, batch_size=30, seed=seed), 0, 1)
    if mode == 8:
        return imagelib.color_transfer_mix(predicted * mask, destination * mask, seed=seed)
    if mode in (9, 10):
        return robust_lab_transfer(predicted, destination, mask, mask, quantile=mode == 10)
    raise ValueError('Unknown color transfer mode')


def distance_feather(mask, width):
    """Only contracts foreground; never fills genuine occlusion holes."""
    mask = np.clip(np.asarray(mask, np.float32), 0, 1)
    if width <= 0:
        return mask
    binary = (mask >= .5).astype(np.uint8)
    padded = np.pad(binary, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    alpha = np.clip(distance / width, 0, 1)
    return np.minimum(mask, alpha).astype(np.float32)


def multiband_blend(base, replacement, mask, levels=4):
    """Laplacian candidate with exact exterior protection and bounded levels."""
    mask = np.asarray(mask, np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    levels = min(int(levels), max(1, int(np.log2(min(base.shape[:2]))) - 2))
    gb, gr, gm = [base], [replacement], [mask]
    for _ in range(levels):
        gb.append(cv2.pyrDown(gb[-1]))
        gr.append(cv2.pyrDown(gr[-1]))
        gm.append(cv2.pyrDown(gm[-1])[..., None])
    out = gb[-1] * (1 - gm[-1]) + gr[-1] * gm[-1]
    for i in range(levels - 1, -1, -1):
        size = (gb[i].shape[1], gb[i].shape[0])
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        lr = gr[i] - cv2.pyrUp(gr[i + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + lb * (1 - gm[i]) + lr * gm[i]
    # Pyramid spreading must not leak into glasses/occluders or outside the mask.
    return np.clip(np.where(mask > 0, out, base), 0, 1).astype(np.float32)
