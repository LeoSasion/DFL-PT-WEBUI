"""Generate reviewable BiSeNet masks on real DFL aligned coordinates, copy only."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / '_internal/DeepFaceLab'))
from DFLIMG import DFLIMG
from vision_assets import atomic_json, canonical_digest, file_sha256
from vision_masks import FACE_PARTS, MaskCandidate

PROFILE = 'bisenet-face-preserve-v1'
SEMANTICS = {
    'target': 'face-only union of BiSeNet classes 1..8,10..13; includes eyes, eyeglasses, mouth and lips',
    'excluded': 'background, earrings, neck, necklace, clothing, hair, hat',
    'eyeMouthPolicy': 'eyes/mouth/lips remain foreground; no erosion and no landmark-based forced hole filling',
    'occlusionPolicy': 'model class predictions only; dark areas/occluders may be misclassified and require visual review',
    'humanPolygons': 'preserved unchanged; no predicted mask converted to human polygons',
    'groundTruth': None, 'iou': None, 'dice': None, 'reviewRequired': True,
}


def metadata_value(value):
    if isinstance(value, np.ndarray):
        return metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {'bytesHex': value.hex()}
    if isinstance(value, dict):
        return {key: metadata_value(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [metadata_value(item) for item in value]
    return value


def metadata_digest(dfl):
    return canonical_digest(metadata_value({key: value for key, value in dfl.get_dict().items() if key not in ('xseg_mask', 'mask_assist')}))


def inspect_aligned(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg'):
        raise ValueError('Only plain DFL aligned JPEG files are supported')
    dfl = DFLIMG.load(path)
    if dfl is None or not dfl.has_data():
        raise ValueError('DFL aligned metadata is required; raw/FFHQ images are not accepted')
    points = np.asarray(dfl.get_landmarks(), dtype=np.float32)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError('Actual aligned 68-point coordinates required; 98-point arrays are not accepted')
    if dfl.get_face_type() not in ('whole_face', 'wf'):
        raise ValueError('This profile requires DFL whole-face (WF) aligned images')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] != image.shape[1] or not 64 <= image.shape[0] <= 4096:
        raise ValueError('Square aligned image dimensions 64..4096 required')
    if np.any(points < -image.shape[0] * .1) or np.any(points > image.shape[0] * 1.1):
        raise ValueError('Landmarks do not use the aligned image coordinate domain')
    return dfl, image, points


def feature_coverage(mask, points):
    result = {}
    for name, indexes in [('leftEye', range(36, 42)), ('rightEye', range(42, 48)), ('mouth', range(48, 60))]:
        region = np.zeros(mask.shape, np.uint8)
        hull = cv2.convexHull(np.round(points[list(indexes)]).astype(np.int32))
        cv2.fillConvexPoly(region, hull, 1)
        count = int(region.sum())
        result[name] = {'landmarkRegionPixels': count,
                        'foregroundFraction': float(mask[region.astype(bool)].mean()) if count else None,
                        'qualification': 'predicted aligned landmark window, not human mask truth'}
    return result


def prepare_draft(input_root, names, output, assets, device='cuda', candidate=None):
    input_root, output = Path(input_root), Path(output)
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError('Plain aligned directory required')
    input_root = input_root.resolve()
    if output.exists() or output.resolve().is_relative_to(input_root):
        raise ValueError('Use a fresh draft directory outside the original aligned dataset')
    if not isinstance(names, list) or not 1 <= len(names) <= 500 or not all(isinstance(name, str) for name in names) or len(set(name.casefold() for name in names)) != len(names):
        raise ValueError('Select 1..500 unique aligned filenames per pass')
    sources = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Plain aligned filenames required')
        source = input_root / name
        inspect_aligned(source)
        sources.append((source, file_sha256(source)))
    output.mkdir(parents=True)
    for directory in ('original', 'copies', 'overlay', 'masks', 'labels'):
        (output / directory).mkdir()
    candidate = candidate or MaskCandidate('bisenet-celebamaskhq', assets, device)
    report = {'schemaVersion': 1, 'profile': PROFILE, 'status': 'ready', 'semantics': SEMANTICS,
              'provenance': candidate.provenance, 'entries': [], 'sourcePolicy': 'original files never overwritten',
              'coordinateDomain': 'actual DFL WF aligned pixel coordinates; model logits resized back to this same grid'}
    for index, (source, original_hash) in enumerate(sources):
        if file_sha256(source) != original_hash:
            raise RuntimeError('Source changed before mask inference')
        dfl, image, points = inspect_aligned(source)
        prediction = candidate.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        labels = prediction['labels']
        if labels.shape != image.shape[:2] or not np.isin(labels, range(19)).all():
            raise ValueError('BiSeNet output must use the actual aligned pixel grid and 19-class domain')
        mask = np.isin(labels, FACE_PARTS)
        if not 0 < int(mask.sum()) < mask.size:
            raise ValueError('Mask is empty/full; candidate requires manual review instead of publication')
        original = output / 'original' / source.name
        result = output / 'copies' / source.name
        shutil.copyfile(source, original)
        shutil.copyfile(source, result)
        result_dfl = DFLIMG.load(result)
        preserved_digest = metadata_digest(dfl)
        # DFL-compatible embedded inference mask only; never fabricates polygons.
        result_dfl.set_xseg_mask(mask.astype(np.float32)[:, :, None])
        identity = candidate.provenance.get('identity') or {}
        result_dfl.get_dict()['mask_assist'] = {
            'schemaVersion': 1, 'profile': PROFILE, 'model': 'bisenet-celebamaskhq',
            'inputSha256': original_hash, 'modelRevision': identity.get('revision'),
            'weightSha256': (identity.get('weights') or [{}])[0].get('sha256'),
            'sourceListSha256': canonical_digest(identity.get('sourceFiles', [])),
            'licenseSha256': (identity.get('licenseFiles') or [{}])[0].get('sha256'),
            'semanticTarget': SEMANTICS['target'], 'coordinateDomain': 'DFL WF aligned pixels',
            'humanGroundTruth': False, 'humanReviewed': False,
            'priorAssistProvenance': dfl.get_dict().get('mask_assist'),
        }
        result_dfl.save()
        loaded, pixels, _ = inspect_aligned(result)
        if metadata_digest(loaded) != preserved_digest or not np.array_equal(pixels, image):
            raise RuntimeError('Copy changed decoded pixels or non-mask DFL metadata')
        if not np.array_equal(loaded.get_xseg_mask()[:, :, 0], mask.astype(np.float32)):
            raise RuntimeError('Embedded mask readback differs from the generated binary mask')
        if file_sha256(source) != original_hash or file_sha256(original) != original_hash:
            raise RuntimeError('Source changed during generation; draft is not reviewable')
        stem = f'{index:04d}'
        tint = np.zeros_like(image)
        tint[:, :, 1] = 210
        overlay = image.copy()
        overlay[mask] = cv2.addWeighted(image, .65, tint, .35, 0)[mask]
        cv2.drawContours(overlay, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)
        overlay_path, mask_path, labels_path = output / 'overlay' / (stem + '.png'), output / 'masks' / (stem + '.png'), output / 'labels' / (stem + '.png')
        for path, data in ((overlay_path, overlay), (mask_path, mask.astype(np.uint8) * 255), (labels_path, labels)):
            if not cv2.imwrite(str(path), data):
                raise RuntimeError('Mask preview output failed')
        entry = {'file': source.name, 'inputSha256': original_hash, 'copySha256': file_sha256(result),
                 'overlaySha256': file_sha256(overlay_path), 'maskSha256': file_sha256(mask_path),
                 'labelsSha256': file_sha256(labels_path), 'preservedMetadataSha256': preserved_digest,
                 'priorEmbeddedMaskPresent': dfl.has_xseg_mask(), 'humanPolygonsPreserved': True,
                 'decodedPixelsExact': True, 'embeddedMaskReadbackExact': True, 'faceType': dfl.get_face_type(),
                 'dimensions': [image.shape[1], image.shape[0]], 'foregroundFraction': float(mask.mean()),
                 'featureCoverage': feature_coverage(mask, points), 'needsReview': True,
                 'paths': {'original': 'original/' + source.name, 'copy': 'copies/' + source.name,
                           'overlay': 'overlay/' + stem + '.png', 'mask': 'masks/' + stem + '.png'}}
        report['entries'].append(entry)
        print(json.dumps({'progress': {'completed': index + 1, 'total': len(sources), 'step': 'mask-inference'}}), flush=True)
    for source, expected in sources:
        if file_sha256(source) != expected:
            raise RuntimeError('Source changed before draft completion')
    atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    arguments = parser.parse_args()
    names = json.loads(sys.stdin.read())['names']
    prepare_draft(arguments.input_root, names, arguments.output, arguments.assets, arguments.device)
