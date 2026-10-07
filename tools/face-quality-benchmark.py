"""Read-only actual-face and controlled-degradation ranking benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.imagelib import estimate_sharpness
from core.imagelib.face_quality import face_detail_signals, metadata_coverage
from DFLIMG import DFLJPG
from facelib import LandmarksProcessor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.is_relative_to(ROOT / 'workspace/.vision-evaluation') or output.exists():
        raise ValueError('Use a new isolated ignored evaluation directory')
    files = sorted(args.aligned.glob('*.jpg'))
    files = [files[index] for index in sorted(set((0, len(files) // 2, len(files) - 1)))]
    before = {str(path): sha(path) for path in files}
    output.mkdir(parents=True)
    blind = output / 'blind'; blind.mkdir()
    controlled = output / 'controlled-aligned'; controlled.mkdir()
    entries, key, scores = [], {}, []
    rng = np.random.default_rng(20261007)
    cases = []
    for group_index, path in enumerate(files):
        original = cv2.imread(str(path))
        dfl = DFLJPG.load(path)
        points = dfl.get_landmarks()
        mask = LandmarksProcessor.get_image_hull_mask(original.shape, points)[..., 0]
        variants = [('original', original),
                    ('blur-moderate', cv2.GaussianBlur(original, (0, 0), 2.0)),
                    ('blur-strong', cv2.GaussianBlur(original, (0, 0), 4.5)),
                    ('blur-noise', np.uint8(np.clip(cv2.GaussianBlur(original, (0, 0), 2.0).astype(np.float32)
                                                   + rng.normal(0, 20, original.shape), 0, 255)))]
        for variant, image in variants:
            pose = LandmarksProcessor.estimate_pitch_yaw_roll(points, size=original.shape[1])
            rect = np.asarray(dfl.get_source_rect(), dtype=float)
            area = float(max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1]))
            detail = face_detail_signals(image, points, foreground=mask)
            cpbd = float(estimate_sharpness(np.uint8(image * mask[..., None])))
            index = len(cases)
            target = controlled / f'controlled-{index:02d}.jpg'
            if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise RuntimeError('Cannot write controlled aligned')
            copy = DFLJPG.load(target); copy.set_dict(dfl.get_dict().copy()); copy.save()
            cases.append({'image': image, 'index': index, 'group': group_index + 1, 'variant': variant,
                          'input': str(path), 'inputSha256': before[str(path)], 'controlled': str(target),
                          'signals': detail, 'coverage': metadata_coverage(points, pose, canvas_size=original.shape[1]),
                          'scores': {**detail['scores'], 'legacy_cpbd': cpbd, 'source_box_area': area}})
    aliases = list(range(len(cases))); random.Random(4107).shuffle(aliases)
    for index, case_index in enumerate(aliases):
        case = cases[case_index]
        alias = f'Q{index + 1:02d}'
        image = case['image']
        canvas = np.full((552, 768, 3), 243, np.uint8)
        cv2.putText(canvas, alias, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2)
        canvas[40:552, :512] = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        points = DFLJPG.load(Path(case['input'])).get_landmarks()
        for crop_index, selected in enumerate((points[36:48], points[48:68])):
            x, y = np.mean(selected, axis=0).astype(int)
            radius = 55 if crop_index == 0 else 50
            crop = image[max(0, y-radius):min(image.shape[0], y+radius), max(0, x-radius):min(image.shape[1], x+radius)]
            canvas[40+crop_index*256:296+crop_index*256, 512:] = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        path = blind / f'{alias}.png'
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError('Cannot write blind panel')
        entries.append({'id': alias, 'file': path.name, 'sha256': sha(path)})
        key[alias] = {k: v for k, v in case.items() if k not in ('image', 'signals', 'coverage', 'scores')}
        scores.append({'id': alias, **{k: v for k, v in case.items() if k not in ('image', 'index')}})
    write(blind / 'manifest.json', {'schemaVersion': 1, 'entries': entries,
          'instructions': 'Score each unlabeled face 0–5 for useful eye/mouth detail and natural clarity; separately note noise, blur and artifacts. Order all IDs best to worst, allowing ties. Images are actual aligned faces with possible controlled degradation, not identity or landmark ground truth. Do not inspect files outside this blind directory before locking scores.'})
    write(output / 'blind-key.json', key)
    write(output / 'proxy-scores.json', scores)
    unchanged = all(sha(Path(path)) == digest for path, digest in before.items())
    write(output / 'acceptance.json', {'schemaVersion': 1, 'sourceSha256': before,
          'originalsUnchanged': unchanged, 'cases': len(entries), 'newTraining': False,
          'accuracy': None, 'blind': str(blind), 'batchLimit': 500, 'pairedLimit': 250})
    if not unchanged:
        raise RuntimeError('Evaluation originals changed')
    print(json.dumps({'blind': str(blind), 'cases': len(entries), 'originalsUnchanged': unchanged}))


if __name__ == '__main__':
    main()
