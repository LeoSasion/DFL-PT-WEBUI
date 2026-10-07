import ast
import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webui/python'))
from mamba_scan import selective_scan_fn


def official_reference():
    path = ROOT / 'workspace/.vision-models/restoration/mambairv2/selective_scan_interface.reference.py'
    if not path.is_file():
        pytest.skip('Pinned official scan reference is an optional validation resource')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'a570f4f15a2eebabf62024a9ffb0d1dde04280300405e1632f43809f448bfb1a'
    node = next(node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef) and node.name == 'selective_scan_ref')
    space = {'torch': torch, 'F': F, 'rearrange': rearrange, 'repeat': repeat}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), space)
    return space['selective_scan_ref']


@pytest.mark.parametrize('shape', [(1, 4, 17), (2, 8, 513), (1, 12, 1537)])
@pytest.mark.parametrize('coefficients', ['static', 'variable', 'grouped'])
def test_scan_matches_official_serial_reference_across_chunks(shape, coefficients):
    batch, channels, length = shape
    generator = torch.Generator().manual_seed(113)
    make = lambda *size: torch.randn(*size, generator=generator)
    u, delta, A = make(*shape), make(*shape), -make(channels, 4).abs()
    sizes = (channels, 4) if coefficients == 'static' else (batch, 4, length) if coefficients == 'variable' else (batch, 2, 4, length)
    B, C, D, z, bias = make(*sizes), make(*sizes), make(channels), make(*shape), make(channels)
    args = (u, delta, A, B, C, D, z, bias, True, True)
    expected, state = official_reference()(*args)
    actual, last = selective_scan_fn(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-5)
    torch.testing.assert_close(last, state, rtol=1e-5, atol=3e-5)


def test_scan_rejects_training_and_complex_coefficients():
    u = torch.zeros(1, 4, 8, requires_grad=True)
    with pytest.raises(ValueError, match='inference only'):
        selective_scan_fn(u, torch.zeros_like(u), torch.zeros(4, 2), torch.ones(4, 2), torch.ones(4, 2))
    with torch.inference_mode(), pytest.raises(ValueError, match='Real FP32'):
        selective_scan_fn(u, torch.zeros_like(u), torch.zeros(4, 2, dtype=torch.complex64), torch.ones(4, 2), torch.ones(4, 2))
