"""Compare the complete pinned network using portable vs official serial scan."""
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webui/python'))
sys.path.insert(0, str(ROOT / 'tools/tests'))
from test_mamba_scan import official_reference
from vision_mambairv2 import MambaRestoration

model = MambaRestoration('cuda:0')
generator = torch.Generator().manual_seed(519)
data = torch.rand(1, 3, 16, 16, generator=generator).cuda()
def run():
    with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(10); torch.cuda.manual_seed(10)
        return model.network(data).cpu()
started = time.monotonic()
portable = run()
reference = official_reference()
for block in model.network.modules():
    if type(block).__name__ == 'Selective_Scan':
        block.selective_scan = reference
serial = run()
torch.testing.assert_close(portable, serial, rtol=2e-5, atol=2e-5)
report = {'schemaVersion': 1, 'model': model.provenance, 'scope': 'full official Large x4 weights, 16x16 RGB random input, FP32 CUDA',
          'maxAbsError': float((portable-serial).abs().max()), 'meanAbsError': float((portable-serial).abs().mean()),
          'outputShape': list(portable.shape), 'passed': True, 'elapsedSeconds': time.monotonic()-started, 'newTraining': False}
output = ROOT / '.validation/all-upgrade-20261007/mambairv2-scan-equivalence.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
