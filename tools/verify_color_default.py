"""Add the actual WebUI no-color baseline to the existing isolated merge QA.

Reuses a copied checkpoint and locked input clip; no training or user-data edits.
The blind key is stored outside the blind folder and must remain hidden until
the requested independent visual reviewer locks scores.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'_internal/DeepFaceLab'))
from core.media_timeline import write_json
from DFLIMG import DFLIMG
from merger.quality_merge import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    base = args.prepared.resolve()
    if not base.is_relative_to(ROOT/'workspace/.vision-evaluation'):
        raise ValueError('Use the isolated merge evaluation root')
    scope = base/'color-default-v2'
    scope.mkdir()  # Never overwrite earlier evaluations, low scores or failures.
    frames, aligned = base/'frames', base/'aligned'
    model = base/'model/verification/me.pt'
    before = sha256(model)
    old_acceptance = json.loads((base/'acceptance.json').read_text(encoding='utf-8'))
    if before != old_acceptance['checkpointCopySha256']:
        raise ValueError('Copied checkpoint differs from the earlier candidate runs')
    methods = ['none', 'rct', 'robust-lab', 'lab-quantile']
    config = {**old_acceptance['runs']['rct']['config'], 'colorTransfer': 'none'}
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'; env['DFL_WEB_MERGE_CONFIG'] = json.dumps(config)
    command = [sys.executable, str(ROOT/'_internal/DeepFaceLab/main.py'), 'merge',
               '--input-dir', str(frames), '--output-dir', str(scope/'merged/none'),
               '--output-mask-dir', str(scope/'merged-masks/none'), '--aligned-dir', str(aligned),
               '--model-dir', str(base/'model'), '--model', 'ME', '--force-model-name', 'verification',
               '--force-gpu-idxs', '0', '--xseg-dir', str(ROOT/'_internal/model_generic_xseg')]
    result = subprocess.run(command, env=env, capture_output=True, timeout=180)
    (scope/'none.out.log').write_bytes(result.stdout); (scope/'none.err.log').write_bytes(result.stderr)
    audit_file = scope/'merged/none/merge.audit.json'
    audit = json.loads(audit_file.read_text(encoding='utf-8')) if audit_file.is_file() else {}
    if result.returncode or audit.get('status') != 'complete' or sha256(model) != before:
        raise RuntimeError('No-color actual CLI merge did not complete or checkpoint changed')
    folders = {'none': scope/'merged/none', **{method: base/'merged'/method for method in methods[1:]}}
    existing_audits = {method: json.loads((folder/'merge.audit.json').read_text(encoding='utf-8')) for method, folder in folders.items()}
    names = sorted(path.name for path in aligned.glob('*.jpg'))
    if len(names) != 8 or len(list(folders['none'].glob('*.png'))) != len(names):
        raise ValueError('The locked eight-frame cohort is required')
    for method, source_audit in existing_audits.items():
        if source_audit['status'] != 'complete' or [item['sourceSha256'] for item in source_audit['frames']] != [item['sourceSha256'] for item in audit['frames']]:
            raise ValueError('Candidate inputs differ: '+method)
        for item in source_audit['frames']:
            output = folders[method]/Path(item['file']).with_suffix('.png').name
            if sha256(output) != item['outputSha256'] or sha256(frames/item['file']) != item['sourceSha256']:
                raise ValueError('Candidate/source changed after its previous audit: '+method)
    ordered = methods.copy(); random.Random(908742).shuffle(ordered)
    key = {chr(65+i): method for i, method in enumerate(ordered)}
    blind = scope/'blind'; blind.mkdir()
    manifest = {'schemaVersion': 1, 'scope': 'same eight actual ME prediction frames; current no-color default plus three color options',
                'noTraining': True, 'checkpointSha256': before,
                'sourceClipSha256': sha256(base/'source.nut'), 'timelineSha256': sha256(frames/'frames.timeline.json'),
                'scoreQualification': 'Visual comparative scores only; source is DST context, not swapped identity truth; no accuracy/AP/NME/IoU. This 0.32 second single interview clip does not prove general temporal, occlusion or glasses quality.',
                'labelPolicy': 'A-D randomized and fixed throughout all eight frames; key outside blind directory',
                'images': []}
    tiles_for_story = []
    for index, name in enumerate(names):
        dfl = DFLIMG.load(aligned/name); source_name = Path(dfl.get_source_filename()).name
        source = cv2.imread(str(frames/source_name)); points = dfl.get_source_landmarks()
        lower = np.floor(points.min(axis=0)-100).astype(int); upper = np.ceil(points.max(axis=0)+100).astype(int)
        x0, y0 = max(0, lower[0]), max(0, lower[1]); x1, y1 = min(source.shape[1], upper[0]), min(source.shape[0], upper[1])
        tiles = []
        for label, image in [('Source', source)] + [(chr(65+i), cv2.imread(str(folders[method]/source_name))) for i, method in enumerate(ordered)]:
            tile = cv2.resize(image[y0:y1, x0:x1], (384,384), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            cv2.putText(tile, label, (12,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
            tiles.append(tile)
        row = np.concatenate(tiles, axis=1); tiles_for_story.append(row)
        target = blind/f'color-default-{index+1:02d}.png'; cv2.imwrite(str(target), row)
        manifest['images'].append({'file': target.name, 'sha256': sha256(target), 'sourceFrame': source_name,
                                   'sourceFrameSha256': sha256(frames/source_name)})
    story = blind/'color-default-storyboard.png'; cv2.imwrite(str(story), np.concatenate(tiles_for_story, axis=0))
    manifest['storyboard'] = {'file': story.name, 'sha256': sha256(story)}
    write_json(blind/'manifest.json', manifest); write_json(scope/'blind-key.json', key)
    write_json(scope/'acceptance.json', {'schemaVersion': 1, 'status': 'complete', 'noTraining': True,
        'newActualCLI': {'colorTransfer': 'none', 'config': config, 'returnCode': result.returncode, 'outputCount': 8,
                        'auditSha256': sha256(audit_file)}, 'reusedCandidateMethods': methods[1:],
        'checkpointBeforeSha256': before, 'checkpointAfterSha256': sha256(model),
        'blindManifestSha256': sha256(blind/'manifest.json'), 'keyIsOutsideBlindDirectory': True})
    print(json.dumps({'blind': str(blind/'manifest.json'), 'status': 'complete', 'imageCount': 8, 'storyboardCount': 1}))


if __name__ == '__main__':
    main()
