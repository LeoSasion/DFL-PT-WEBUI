"""Bounded inference-only verification on an already prepared isolated clip.

Requires genuine SHA-bound frames and DFL aligned files. Copies a checkpoint;
never trains, edits the supplied checkpoint, or publishes a user mask batch.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
sys.path.insert(0, str(ROOT / 'webui/python'))
from core.media_timeline import write_json
from DFLIMG import DFLIMG
from merger.quality_merge import sha256, points_digest
from mask_assist import prepare_draft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--parent-video', type=Path, required=True)
    args = parser.parse_args()
    base = args.prepared.resolve()
    if not base.is_relative_to(ROOT / 'workspace/.vision-evaluation') or (base / 'acceptance.json').exists():
        raise ValueError('Fresh isolated .vision-evaluation batch required')
    frames, aligned = base / 'frames', base / 'aligned'
    names = sorted(path.name for path in aligned.glob('*.jpg'))
    if not 3 <= len(names) <= 12:
        raise ValueError('This bounded verification requires 3..12 actual extracted faces')
    checkpoint_hash = sha256(args.checkpoint)
    model = base / 'model' / 'verification'; model.mkdir(parents=True)
    shutil.copyfile(args.checkpoint, model / 'me.pt')
    report = prepare_draft(aligned, names, base / 'mask-draft', ROOT / 'workspace/.vision-models/masks')
    reviewed = base / 'reviewed'; reviewed.mkdir()
    entries = []
    for entry in report['entries']:
        path = base / 'mask-draft/copies' / entry['file']
        dfl = DFLIMG.load(path)
        frame = frames / Path(dfl.get_source_filename()).name
        pixels = cv2.imread(str(frame))
        output = reviewed / path.name; shutil.copyfile(path, output)
        entries.append({'file': path.name, 'assisted': True, 'sourceSha256': sha256(aligned/path.name),
                        'outputSha256': sha256(output), 'sourceFrame': {
                            'file': frame.name, 'sha256': sha256(frame), 'width': pixels.shape[1], 'height': pixels.shape[0],
                            'alignedCanvasWh': list(reversed(dfl.get_shape()[:2])),
                            'sourceToAlignedAffine': dfl.get_image_to_face_mat().tolist(),
                            'sourceLandmarksSha256': points_digest(dfl.get_source_landmarks()),
                            'alignedLandmarksSha256': points_digest(dfl.get_landmarks())}})
    write_json(reviewed / 'mask-assist-provenance.json', {
        'schemaVersion': 1, 'side': 'dst', 'status': 'published', 'reviewed': True, 'entries': entries,
        'qualification': 'ISOLATED FUNCTIONAL FIXTURE ACKNOWLEDGMENT; no claim of human review or ground truth; not a user batch'})
    candidates = {
        'rct': {}, 'robust-lab': {'colorTransfer': 'robust-lab'}, 'lab-quantile': {'colorTransfer': 'lab-quantile'},
        'distance': {'blendMode': 'distance'}, 'multiband': {'blendMode': 'multiband'},
        'geometry': {'geometryMode': 'bounded-center-scale', 'geometryStrength': 60},
        'reviewed-dst': {'maskMode': 11},
    }
    acceptance = {'schemaVersion': 1, 'scope': '8-frame actual ME prediction/CLI merging; no training',
                  'parentVideoSha256': sha256(args.parent_video), 'clipSha256': sha256(base/'source.nut'),
                  'clipParentExtraction': {'seekSeconds': 52, 'durationSeconds': .32, 'working': 'FFmpeg decoded RGB8, re-encoded FFV1'},
                  'checkpointInputSha256': checkpoint_hash, 'checkpointCopySha256': sha256(model/'me.pt'), 'runs': {}}
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'
    for name, options in candidates.items():
        config = {'mode': 'overlay', 'maskMode': 4, 'colorTransfer': 'rct', 'workers': 1,
                  'randomSeed': 121, 'blurMask': 16, **options}
        env['DFL_WEB_MERGE_CONFIG'] = json.dumps(config)
        out, masks = base/'merged'/name, base/'merged-masks'/name
        command = [sys.executable, str(ROOT/'_internal/DeepFaceLab/main.py'), 'merge', '--input-dir', str(frames),
                   '--output-dir', str(out), '--output-mask-dir', str(masks), '--aligned-dir', str(reviewed if name == 'reviewed-dst' else aligned),
                   '--model-dir', str(base/'model'), '--model', 'ME', '--force-model-name', 'verification', '--force-gpu-idxs', '0',
                   '--xseg-dir', str(ROOT/'_internal/model_generic_xseg')]
        result = subprocess.run(command, env=env, capture_output=True, timeout=180)
        (base/(name+'.out.log')).write_bytes(result.stdout); (base/(name+'.err.log')).write_bytes(result.stderr)
        audit_path = out/'merge.audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else {}
        acceptance['runs'][name] = {'returnCode': result.returncode, 'config': config, 'audit': str(audit_path.relative_to(base)),
                                    'outputCount': len(list(out.glob('*.png'))), 'status': audit.get('status')}
        write_json(base/'acceptance.json', acceptance)
        if result.returncode or audit.get('status') != 'complete' or len(list(out.glob('*.png'))) != len(names):
            raise RuntimeError('Actual CLI merge failed: '+name)
        print('Verified actual CLI merge:', name, flush=True)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Original checkpoint changed during inference acceptance')
    acceptance['originalCheckpointRetained'] = True
    acceptance['originalFramesRetained'] = all(sha256(frames/e['sourceFrame']['file']) == e['sourceFrame']['sha256'] for e in entries)
    blind = base/'blind'; blind.mkdir()
    rng = random.Random(735092)
    groups = {'color': ['rct', 'robust-lab', 'lab-quantile'], 'blend': ['rct', 'distance', 'multiband'], 'geometry': ['rct', 'geometry']}
    key, manifest = {}, {'schemaVersion': 1, 'qualification': 'Visual scores, not accuracy/AP/NME/IoU; a single bounded interview clip, no occlusion or glasses guarantee', 'groups': {}}
    for group, methods in groups.items():
        ordered = methods.copy(); rng.shuffle(ordered); key[group] = {chr(65+i): name for i, name in enumerate(ordered)}
        paths = []
        for index, name in enumerate(names):
            dfl = DFLIMG.load(aligned/name); frame_name = Path(dfl.get_source_filename()).name
            source = cv2.imread(str(frames/frame_name)); points = dfl.get_source_landmarks()
            x0, y0 = np.floor(points.min(axis=0)-100).astype(int); x1, y1 = np.ceil(points.max(axis=0)+100).astype(int)
            x0, y0, x1, y1 = max(0,x0), max(0,y0), min(source.shape[1],x1), min(source.shape[0],y1)
            tiles = []
            for label, image in [('Source', source)] + [(chr(65+i), cv2.imread(str(base/'merged'/method/frame_name))) for i, method in enumerate(ordered)]:
                tile = cv2.resize(image[y0:y1,x0:x1], (384, 384), interpolation=cv2.INTER_AREA)
                tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
                cv2.putText(tile, label, (12,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA); tiles.append(tile)
            output = blind / f'{group}-{index+1:02d}.png'; cv2.imwrite(str(output), np.concatenate(tiles,axis=1))
            paths.append({'file': output.name, 'sha256': sha256(output), 'sourceFrameSha256': sha256(frames/frame_name)})
        # Storyboard rows preserve temporal order; labels stay randomized per group.
        storyboard = blind/(group+'-storyboard.png')
        cv2.imwrite(str(storyboard), np.concatenate([cv2.imread(str(blind/p['file'])) for p in paths],axis=0))
        manifest['groups'][group] = {'images': paths, 'storyboard': {'file': storyboard.name, 'sha256': sha256(storyboard)}}
    write_json(base/'blind-key.json', key); write_json(blind/'manifest.json', manifest)
    acceptance['blindManifestSha256'] = sha256(blind/'manifest.json')
    write_json(base/'acceptance.json', acceptance)
    print(json.dumps({'acceptance': str(base/'acceptance.json'), 'blind': str(blind/'manifest.json')}))


if __name__ == '__main__':
    main()
