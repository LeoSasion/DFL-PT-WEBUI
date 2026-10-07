"""Shared six-image frozen mask functionality probe; no accuracy ranking."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'webui' / 'python'))
from vision_assets import atomic_json, file_sha256
from vision_masks import MODELS, MaskCandidate, mask_statistics


def benchmark(inputs, output, assets, device='cuda', models=MODELS):
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    if inputs == output or output.is_relative_to(inputs):
        raise ValueError('Outputs must be separate from shared benchmark inputs')
    if output.exists():
        raise ValueError('Use a fresh mask evaluation directory')
    protocol = json.loads((inputs / 'protocol.json').read_text(encoding='utf-8'))
    cases = protocol['cases'][:6]
    output.mkdir(parents=True)
    import torch
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {'schemaVersion': 1, 'torchVersion': torch.__version__, 'purpose': 'mask functionality and semantics only',
              'protocolSha256': file_sha256(inputs / 'protocol.json'), 'humanGroundTruthAvailable': False,
              'iou': None, 'dice': None, 'winner': None, 'models': [],
              'validPixelDomain': 'shared aligned source-valid eroded mask; reflected border excluded',
              'limitations': ['different mask semantics; no quality/accuracy ranking',
                              'XSeg expects WF; shared FFHQ crops are a cross-alignment functional test',
                              'SAM fixed predicted-alignment-relative prompt is not human mask truth']}
    for model_id in models:
        print('Frozen mask: ' + model_id, flush=True)
        candidate = MaskCandidate(model_id, assets, device)
        entries = []
        for case in cases:
            reference, valid_file = inputs / case['case'] / 'reference.png', inputs / case['case'] / 'valid-mask.png'
            if file_sha256(reference) != case['referenceSha256'] or file_sha256(valid_file) != case['validMaskSha256']:
                raise ValueError('Shared locked image/valid-pixel hash changed')
            image = cv2.cvtColor(cv2.imread(str(reference)), cv2.COLOR_BGR2RGB)
            valid = cv2.imread(str(valid_file), cv2.IMREAD_GRAYSCALE) >= 128
            result = candidate.predict(image)
            directory = output / case['case']
            directory.mkdir(exist_ok=True)
            mask_file = directory / (model_id + '.png')
            if not cv2.imwrite(str(mask_file), result['mask'].astype(np.uint8) * 255):
                raise RuntimeError('Mask output failed')
            item = {key: value for key, value in result.items() if key not in ('mask', 'labels')}
            if 'labels' in result:
                cv2.imwrite(str(directory / (model_id + '-labels.png')), result['labels'])
            item.update(case=case['case'], inputSha256=case['referenceSha256'], maskSha256=file_sha256(mask_file),
                        statistics=mask_statistics(result['mask'], valid))
            entries.append(item)
        report['models'].append({'id': model_id, 'provenance': candidate.provenance, 'cases': entries})
        atomic_json(output / 'report.json', report)
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=PROJECT_ROOT / 'workspace/.vision-models/masks')
    parser.add_argument('--device', default='cuda')
    arguments = parser.parse_args()
    benchmark(arguments.inputs, arguments.output, arguments.assets, arguments.device)
