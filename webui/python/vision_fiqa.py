"""Pinned Efficient-FIQA student, isolated from TUFA's older timm contract.

The worker uses this project's Python and PyTorch. Its fixed pure-Python timm
wheel is added only to the worker's import path. Inference never downloads.
Scores express perceptual image quality, not identity or training accuracy.
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import json
import math
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = 'efficient-fiqa'
REVISION = '68f4c9eb90faa6a474c635f0f34df64304397595'
SOURCE = f'https://github.com/sunwei925/Efficient-FIQA/tree/{REVISION}'
ASSETS = {
    'EdgeNeXt_XXS_checkpoint.pt': ('29eec850648acff953eae3721487ff3532305f7b885ed79851b092f6d22cef69', 4772402),
    'FIQA_model.py': ('c7b95e43db95c2d404335e58e2622938b6701108ecb98b2a71444e736841dc95', 2998),
    'LICENSE': ('c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4', 11357),
    'test.py': ('f820800f2b21d31ff0f27da411bc1e4e017be293d7b2569d0a98117300501fad', 3943),
    'timm-1.0.19-py3-none-any.whl': ('c07b56c32f3d3226c656f75c1b5479c08eb34eefed927c82fd8751a852f47931', 2497950),
}
PREPROCESS = 'RGB/PIL; Resize(short-side=352,bilinear); CenterCrop(352); ImageNet-normalize; FP32'
COMPARISON = {'date': '2026-10-07', 'status': 'no-clear-benefit',
              'decision': 'retain-current-aggregate-quality', 'sourceGroups': 8, 'cases': 64,
              'currentMeanWithinSourceSpearman': .5592, 'efficientMeanWithinSourceSpearman': .5185,
              'assessor': 'gpt-6-astra/low; subjective exploratory evidence',
              'report': 'docs/EFFICIENT_FIQA_COMPARISON_20261007.md'}


def verified_assets(assets_root=None):
    if assets_root is None:
        from vision_resource_paths import resource_group
        assets_root = resource_group(MODEL_ID, ROOT / 'workspace/.vision-models' / MODEL_ID)
    root = Path(assets_root).resolve()
    for name, (digest, size) in ASSETS.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
            raise ValueError(f'Missing pinned Efficient-FIQA asset: {name}; run tools/prepare-efficient-fiqa.py')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError(f'Efficient-FIQA checksum mismatch: {name}')
    return root


def availability():
    try:
        verified_assets()
        return {'available': True, 'model': MODEL_ID, 'admission': 'comparison-only',
                'default': False, 'comparison': COMPARISON,
                'runtime': 'project-PyTorch/isolated-timm-wheel'}
    except (OSError, ValueError) as exc:
        return {'available': False, 'model': MODEL_ID, 'admission': 'comparison-only',
                'default': False, 'comparison': COMPARISON, 'reason': str(exc)}


def _lines(stream, inbox):
    try:
        for line in stream:
            inbox.put(line)
    finally:
        inbox.put(None)


class EfficientFIQAScorer:
    """A bounded JSON worker; close it after a batch, or use a context manager."""
    def __init__(self, *, device='cpu', assets_root=None, timeout=90):
        if device != 'cpu' and not (device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('Efficient-FIQA device must be cpu or cuda[:index]')
        self.assets_root = verified_assets(assets_root)
        self.timeout = float(timeout)
        if not 1 <= self.timeout <= 300:
            raise ValueError('FIQA timeout must be between 1 and 300 seconds')
        self.stderr = tempfile.TemporaryFile(mode='w+b')
        self.inbox = queue.Queue()
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker', '--assets-root', str(self.assets_root), '--device', device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding='utf-8', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        threading.Thread(target=_lines, args=(self.process.stdout, self.inbox), daemon=True).start()
        atexit.register(self.close)
        try:
            result = self._receive()
            if result.get('ready') is not True:
                raise RuntimeError('FIQA worker did not initialize')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            line = self.inbox.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError('Efficient-FIQA worker exceeded its bounded timeout') from exc
        if line is None:
            self.stderr.seek(0)
            detail = self.stderr.read()[-1800:].decode('utf-8', errors='replace')
            raise RuntimeError('Efficient-FIQA worker stopped: ' + detail)
        result = json.loads(line)
        if result.get('error'):
            raise RuntimeError('Efficient-FIQA inference failed: ' + result['error'])
        return result

    def __call__(self, image_bgr, metadata=None):
        import cv2
        import numpy as np
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
            raise ValueError('FIQA needs a uint8 HWC BGR face image of at least 16 pixels')
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Cannot encode FIQA input')
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError('Efficient-FIQA scorer is closed')
            self.process.stdin.write(json.dumps({'png': base64.b64encode(encoded).decode('ascii')}) + '\n')
            self.process.stdin.flush()
            raw = float(self._receive()['rawScore'])
        if not math.isfinite(raw):
            raise ValueError('Non-finite FIQA output')
        return {'model': MODEL_ID, 'score': min(1.0, max(0.0, raw)) * 100.0,
                'minimumAcceptable': 0.0,
                'provenance': {'modelId': MODEL_ID, 'architecture': 'EdgeNeXt-XXS-student',
                    'source': SOURCE, 'sourceRevision': REVISION,
                    'weightSha256': ASSETS['EdgeNeXt_XXS_checkpoint.pt'][0],
                    'architectureWheelSha256': ASSETS['timm-1.0.19-py3-none-any.whl'][0],
                    'preprocessing': PREPROCESS, 'rawScore': raw,
                    'displayScale': '100*clip(raw,0,1); not a percentage accuracy',
                    'cutoffStatus': 'uncalibrated; existing hard-filter governs eligibility',
                    'admission': 'comparison-only; no automatic promotion'}}

    def close(self):
        process = getattr(self, 'process', None)
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
        self.stderr.close()
        atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_quality_scorer(device='cpu', *, assets_root=None):
    return EfficientFIQAScorer(device=device, assets_root=assets_root)


def _worker(assets_root, device):
    import ast
    import io
    import torch
    from PIL import Image
    from torchvision import transforms
    root = verified_assets(assets_root)
    sys.path.insert(0, str(root / 'timm-1.0.19-py3-none-any.whl'))
    import timm
    if timm.__version__ != '1.0.19':
        raise RuntimeError('FIQA worker loaded a different timm version')
    # Execute only the two verbatim official student classes, never the teachers.
    source = ast.parse((root / 'FIQA_model.py').read_text(encoding='utf-8'))
    nodes = [node for node in source.body if isinstance(node, ast.ClassDef)
             and node.name in ('Identity', 'FIQA_EdgeNeXt_XXS')]
    if len(nodes) != 2:
        raise ValueError('Pinned student definition is incomplete')
    namespace = {'torch': torch, 'nn': torch.nn, 'timm': timm}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-efficient-fiqa-student>', 'exec'), namespace)
    model = namespace['FIQA_EdgeNeXt_XXS'](is_pretrained=False)
    state = torch.load(root / 'EdgeNeXt_XXS_checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device).eval()
    transform = transforms.Compose([transforms.Resize(352), transforms.CenterCrop(352),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    print(json.dumps({'ready': True}), flush=True)
    inbox = queue.Queue()
    threading.Thread(target=_lines, args=(sys.stdin, inbox), daemon=True).start()
    while True:
        try:
            line = inbox.get(timeout=60)
        except queue.Empty:
            return  # Idle workers cannot remain orphaned indefinitely.
        if line is None:
            return
        try:
            request = json.loads(line)
            payload = base64.b64decode(request['png'], validate=True)
            if len(payload) > 64 * 1024 * 1024:
                raise ValueError('FIQA input is too large')
            image = Image.open(io.BytesIO(payload)).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                raw = float(model(tensor).item())
            if not math.isfinite(raw):
                raise ValueError('Non-finite model output')
            print(json.dumps({'rawScore': raw}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': str(exc)[:1200]}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--assets-root', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.worker:
        parser.error('This entry point is an internal inference worker')
    _worker(args.assets_root, args.device)
