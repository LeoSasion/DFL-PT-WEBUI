"""Native TUFA topologies with a separate, unchanged DFL 68-point contract."""
import argparse
import copy
from contextlib import redirect_stdout
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAN_HASHES = {
    '2DFAN.npy': 'ca2dc7f0b2aa146842e6de2119fedff1142188b7cff5ab702564952d6cba4624',
    '3DFAN.npy': 'b50d2faf0fd4d6503aba9d19365e7aff06f2ea96bac37fc5f8f25a191a0a63a9',
}


def local_module(name):
    location = ROOT / 'webui/python'
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != location / (name + '.py'):
        raise ValueError('Unexpected project landmark module')
    return module


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fan_identity(head):
    filename = '3DFAN.npy' if head else '2DFAN.npy'
    # Match the existing FAN constructor's search order, including its legacy
    # local compatibility directory. Never verify one file and execute another.
    path = ROOT / '_internal/DeepFaceLab/DeepFaceLab-master/facelib' / filename
    if not path.exists():
        path = Path(__file__).parent / filename
    path = local_module('vision_assets').resolve_asset_path(ROOT, path.relative_to(ROOT).as_posix())
    if sha256(path) != FAN_HASHES[filename]:
        raise ValueError('FAN landmark asset differs from pinned release identity')
    return {'model': 'fan3d' if head else 'fan68', 'file': filename,
            'sha256': FAN_HASHES[filename], 'output': 'IBUG68 XY; depth is not returned'}


def validate_landmark_assets(model='fan', face_type=None, assets_root=None):
    if model not in ('fan', 'tufa'):
        raise ValueError('Unsupported landmark model')
    head = face_type is None or str(face_type).lower() in ('head', 'head_no_align')
    result = {'model': model, 'geometryProtocol': 'dfl-native-68-v1', 'assets': []}
    if model == 'fan' or head:
        result['assets'].append(fan_identity(head))
        if model == 'fan' and face_type is None:
            result['assets'].append(fan_identity(False))
    if model == 'tufa':
        module = local_module('vision_landmarks')
        directory = local_module('vision_resource_paths').resource_group('tufa', module.DEFAULT_ASSETS / 'TUFA') if assets_root is None else local_module('vision_assets').resolve_asset_path(assets_root, 'TUFA')
        result['assets'].append(module.verified_identity(directory, expected_model='TUFA'))
        # Verify executable optional dependencies before aligned deletion.
        for dependency in ('torchvision', 'timm'):
            importlib.import_module(dependency)
    return result


class TufaExtractor:
    def __init__(self, device='cpu'):
        module = local_module('vision_landmarks')
        self.predictor = module.LandmarkPredictor('tufa68', device=device)
        # Both native prompts use the same verified official network. Sharing
        # the network avoids duplicating its weights in each extraction worker.
        self.audit_predictor = copy.copy(self.predictor)
        self.audit_predictor.count = 98
        self.audit_predictor.model_id = 'tufa98'
        self.audit_predictor.identity = copy.deepcopy(self.predictor.identity)
        prompt_path = self.predictor.directory / 'source/Prompt/shape_98.npz'
        self.audit_predictor.prompt = module.torch.from_numpy(np.load(prompt_path)['offset'] / 256).float().to(self.predictor.device)[None]
        self.audit_predictor.identity['prompt'] = {'file': 'Prompt/shape_98.npz', 'sha256': sha256(prompt_path)}
        self.identity = self.predictor.identity

    def extract(self, image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        rgb = np.ascontiguousarray(image[:, :, ::-1] if is_bgr else image)
        points = [np.asarray(self.predictor.predict(rgb, rect)['points_original'], dtype=np.float32) for rect in rects]
        if any(p.shape != (68, 2) or not np.isfinite(p).all() for p in points):
            raise ValueError('Native TUFA68 output violates the DFL geometry contract')
        return points

    def audit(self, image, rects):
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        # The legacy rotation mapping may store descending rectangle corners.
        # Diagnose in the actual original canvas with a canonical box, without
        # changing its historical 68-point mapping or alignment implementation.
        return [self.audit_predictor.predict(rgb, [min(r[0], r[2]), min(r[1], r[3]),
                max(r[0], r[2]), max(r[1], r[3])]) for r in rects]


def load_source_map(path, input_path):
    if path is None:
        return {}
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(record, dict) or not isinstance(record.get('outputs'), list):
        raise ValueError('source-map must contain an outputs array')
    output = {}
    for entry in record['outputs']:
        for key in ('name', 'sourceName'):
            name = entry.get(key)
            if not isinstance(name, str) or not name or Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
                raise ValueError('source-map names must be basenames')
        if entry['name'] in output:
            raise ValueError('Duplicate restored source-map filename')
        image = local_module('vision_assets').resolve_asset_path(input_path, entry['name'])
        if entry.get('inputSize') is not None and entry.get('outputSize') is not None and entry['inputSize'] != entry['outputSize']:
            raise ValueError('Restored image must preserve the original canvas size')
        digest = sha256(image)
        if entry.get('outputSha256') and entry['outputSha256'] != digest:
            raise ValueError('Restored image hash differs from source-map')
        if entry.get('inputSha256') and (not isinstance(entry['inputSha256'], str) or len(entry['inputSha256']) != 64 or any(c not in '0123456789abcdef' for c in entry['inputSha256'])):
            raise ValueError('Invalid declared original source hash')
        output[entry['name']] = {'processedFilename': entry['name'], 'sourceFilename': entry['sourceName'],
            'processedSha256': digest, 'declaredInputSha256': entry.get('inputSha256'),
            'originalHashVerified': False, 'coordinatePolicy': 'same-size original canvas required',
            'restoration': {key: entry[key] for key in ('model', 'modelId', 'assetIdentity', 'inputSize', 'outputSize') if key in entry}}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-assets', action='store_true', required=True)
    parser.add_argument('--model', choices=('fan', 'tufa'), default='fan')
    parser.add_argument('--face-type', default=None)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            result = validate_landmark_assets(args.model, args.face_type)
        print(json.dumps({'ok': True, **result}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False))
        sys.exit(1)
