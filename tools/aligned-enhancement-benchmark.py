"""Actual aligned and ME-predicted face restoration; no training/ground-truth claims."""
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
from facelib.FaceEnhancement import AlignedFaceEnhancer
from me_backend.model_adapter import MEInferenceModel
from me_backend.data import read_aligned


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.is_relative_to(ROOT / 'workspace/.vision-evaluation') or output.exists():
        raise ValueError('Use a new isolated ignored evaluation output')
    files = sorted(args.aligned.glob('*.jpg'))
    files = [files[i] for i in sorted(set((0, len(files)//2, len(files)-1)))]
    source_hashes = {str(p): sha(p) for p in files}
    checkpoint_hash = sha(args.model / 'me.pt')
    from core.leras import nn
    nn.initialize_main_env()
    model = MEInferenceModel(args.model, cpu_only=False)
    predict, _, _ = model.get_MergerConfig()
    inputs = []
    for index, path in enumerate(files):
        aligned = cv2.imread(str(path)).astype(np.float32) / 255.
        model_input = read_aligned(path, model.engine.config)[0]
        predicted = predict(model_input)[0].astype(np.float32)
        inputs.extend([{'id': f'aligned-{index+1}', 'kind': 'aligned', 'image': aligned},
                       {'id': f'predicted-{index+1}', 'kind': 'ME-predicted', 'image': predicted}])
    del model
    import torch
    torch.cuda.empty_cache()
    output.mkdir(parents=True)
    raw = output / 'raw'; raw.mkdir()
    results = {}
    for model_id in ('mambairv2', 'realesrgan-x4plus'):
        enhancer = AlignedFaceEnhancer(model_id, device='cuda:0')
        for item in inputs:
            result = enhancer.enhance(item['image'], is_tanh=False, preserve_size=True)
            results[(item['id'], model_id)] = np.rint(result * 255).astype(np.uint8)
        del enhancer
        torch.cuda.empty_cache()
    blind = output / 'blind'; blind.mkdir()
    rng = random.Random(20261007)
    keys, entries = {}, []
    families = {'aligned': ['mambairv2', 'realesrgan-x4plus'],
                'ME-predicted': ['mambairv2', 'realesrgan-x4plus']}
    for kind, choices in families.items():
        choices = choices[:]; rng.shuffle(choices)
        keys[kind] = dict(zip('ABCD', choices))
    for item in inputs:
        choices = keys[item['kind']]
        tiles = []
        for label, img in [('INPUT', np.rint(item['image'] * 255).astype(np.uint8)), *[(label, results[(item['id'], key)]) for label, key in choices.items()]]:
            panel = np.full((296, 256, 3), 242, np.uint8)
            panel[40:, :] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 30, 30), 2)
            tiles.append(panel)
        canvas = np.concatenate(tiles, axis=1)
        target = blind / (item['id'] + '.png')
        if not cv2.imwrite(str(target), canvas):
            raise RuntimeError('Blind panel write failed')
        entries.append({'id': item['id'], 'kind': item['kind'], 'file': target.name, 'sha256': sha(target), 'labels': list(choices)})
    write(blind / 'manifest.json', {'schemaVersion': 1, 'entries': entries, 'guidance': 'INPUT is the exact pre-restoration face, not identity ground truth. Same aliases within each kind. Score detail, eye/mouth preservation, skin, artifacts. No temporal or identity accuracy ground truth.'})
    write(output / 'blind-key.json', keys)
    unchanged = all(sha(Path(name)) == digest for name, digest in source_hashes.items()) and sha(args.model / 'me.pt') == checkpoint_hash
    write(output / 'acceptance.json', {'schemaVersion': 1, 'sourceSha256': source_hashes, 'checkpointSha256': checkpoint_hash,
         'originalsUnchanged': unchanged, 'inputs': [{k: v for k,v in item.items() if k != 'image'} for item in inputs],
         'newTraining': False, 'accuracy': None, 'temporalQuality': None, 'review': str(blind)})
    if not unchanged:
        raise RuntimeError('Evaluation sources changed')
    print(json.dumps({'review': str(blind), 'cases': len(entries), 'originalsUnchanged': unchanged}))


if __name__ == '__main__':
    main()
