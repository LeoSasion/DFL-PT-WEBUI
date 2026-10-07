"""Bounded full CLI extraction + metadata/geometry acceptance, no training.

The supplied read-only manifest contains 1-3 samples with image/id/sha256.
Private inputs and outputs must stay in ignored evaluation folders.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webui/python'))
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from vision_assets import atomic_json, file_sha256, prepare_evaluation_output
from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor


def run(manifest, output, detector, gpu):
    records = json.loads(Path(manifest).read_text(encoding='utf-8'))['samples']
    if not 1 <= len(records) <= 3:
        raise ValueError('Use 1-3 explicitly selected source images; total CLI inputs stay at most ten')
    for sample in records:
        if file_sha256(sample['image']) != sample['sha256']:
            raise ValueError('Read-only source image hash mismatch')
    output = prepare_evaluation_output(output)
    source = output / 'input'
    source.mkdir()
    for index, sample in enumerate(records):
        shutil.copyfile(sample['image'], source / f'sample{index}{Path(sample["image"]).suffix}')
    atomic_json(output / 'selected-inputs.json', records)
    results = []
    baseline = {}
    for model, face_type in [('tufa', 'whole_face'), ('fan', 'head'), ('tufa', 'head')]:
        label = model + '-' + face_type
        target = output / label
        command = [str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / '_internal/DeepFaceLab/main.py'), 'extract',
                   '--input-dir', str(source), '--output-dir', str(target), '--detector', detector,
                   '--landmark-model', model, '--face-type', face_type, '--image-size', '256',
                   '--jpeg-quality', '95', '--max-faces-from-image', '1', '--no-output-debug', '--force-gpu-idxs', gpu]
        with (output / (label + '.log')).open('w', encoding='utf-8') as log:
            import os
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180,
                                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if completed.returncode:
            raise RuntimeError('Full extraction failed: ' + label)
        files = sorted(target.glob('*.jpg'))
        if len(files) != len(records):
            raise RuntimeError('Incomplete real extraction coverage: ' + label)
        for path in files:
            dfl = DFLJPG.load(path)
            landmarks = dfl.get_landmarks()
            original = dfl.get_source_landmarks()
            mat = np.asarray(dfl.get_image_to_face_mat())
            if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
                raise RuntimeError('Legacy68 geometry corrupted')
            # DFL stores lists, while both production extractors provide float32.
            # Reconstruct the actual extraction dtype for the legacy estimator.
            projected = LandmarksProcessor.transform_points(original.astype(np.float32), mat)
            regenerated = LandmarksProcessor.get_transform_mat(original.astype(np.float32), 256, FaceType.fromString(face_type))
            error = float(np.max(np.abs(projected - landmarks)))
            matrix_error = float(np.max(np.abs(regenerated - mat)))
            if error > 1e-4 or matrix_error > 1e-6:
                raise RuntimeError('DFL geometry roundtrip changed')
            provenance = dfl.get_dict()['landmark_provenance']
            if model == 'fan':
                baseline[path.name] = (original, mat)
            else:
                native = dfl.get_dict()['native_landmarks98']
                sidecar = json.loads(Path(str(path) + '.landmarks.json').read_text(encoding='utf-8'))
                if sidecar['aligned_sha256'] != file_sha256(path) or native['point_definition']['count'] != 98:
                    raise RuntimeError('Native98 identity/point contract corrupted')
                np.testing.assert_allclose(LandmarksProcessor.transform_points(native['points_original'], mat),
                                           native['points_aligned'], atol=1e-4)
                annotated = cv2.imread(str(path))
                for x, y in landmarks:
                    cv2.circle(annotated, (round(float(x)), round(float(y))), 1, (0, 255, 0), -1)
                for x, y in native['points_aligned']:
                    cv2.circle(annotated, (round(float(x)), round(float(y))), 1, (0, 170, 255), -1)
                cv2.imwrite(str(path.with_suffix('.audit.png')), annotated)
                if face_type == 'head':
                    previous, previous_mat = baseline[path.name]
                    np.testing.assert_array_equal(original, previous)
                    np.testing.assert_array_equal(mat, previous_mat)
                    if provenance['alignmentModel'] != 'fan3d':
                        raise RuntimeError('HEAD lost real FAN3D provenance')
            results.append({'run': label, 'aligned': str(path), 'source68_shape': list(original.shape),
                            'landmark_provenance': provenance, 'geometryRoundtripMaxPx': error,
                            'matrixRegenerationMaxAbs': matrix_error, 'headBaselineExactlyEqual': face_type == 'head' and model == 'tufa',
                            'native98Present': 'native_landmarks98' in dfl.get_dict()})
    # A PNG container copy validates restoration source naming without claiming
    # that a restoration network was run by this extraction acceptance.
    restored = output / 'source-map-input'
    restored.mkdir()
    first = sorted(source.iterdir())[0]
    png = restored / 'restored.png'
    cv2.imwrite(str(png), cv2.imread(str(first)))
    mapping = output / 'source-map.json'
    atomic_json(mapping, {'outputs': [{'name': png.name, 'sourceName': first.name,
        'inputSha256': file_sha256(first), 'outputSha256': file_sha256(png)}]})
    target = output / 'source-map-aligned'
    command = [str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / '_internal/DeepFaceLab/main.py'), 'extract',
               '--input-dir', str(restored), '--output-dir', str(target), '--detector', detector,
               '--landmark-model', 'tufa', '--source-map', str(mapping), '--face-type', 'whole_face',
               '--image-size', '256', '--jpeg-quality', '95', '--max-faces-from-image', '1', '--no-output-debug', '--force-gpu-idxs', gpu]
    with (output / 'source-map.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180,
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if completed.returncode or len(list(target.glob('*.jpg'))) != 1:
        raise RuntimeError('Restored source-map extraction failed')
    dfl = DFLJPG.load(next(target.glob('*.jpg')))
    if dfl.get_source_filename() != first.name or dfl.get_dict()['restoration_provenance']['processedSha256'] != file_sha256(png):
        raise RuntimeError('Restoration source provenance roundtrip failed')
    for sample in records:
        if file_sha256(sample['image']) != sample['sha256']:
            raise RuntimeError('Original media changed')
    atomic_json(output / 'acceptance.json', {'ok': True, 'totalCliInputImages': 3 * len(records) + 1,
        'detector': detector, 'records': results, 'sourceMapRoundtrip': True, 'sourceMapFixtureIsContainerCopy': True,
        'originalMediaUnchanged': True, 'visualQualityReview': 'pending independent reviewer; overlays are not scores'})
    print(json.dumps({'ok': True, 'report': str(output / 'acceptance.json'), 'records': len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--detector', default='yolo26s-face')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    run(args.input_manifest, args.output, args.detector, args.gpu)
