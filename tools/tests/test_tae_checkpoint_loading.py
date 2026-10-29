"""TAE checkpoint interoperability and shared safe loading; synthetic tensors only."""
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))
from core.torch_checkpoint import load_weights_checkpoint
from tae.checkpoint import FORMAT, VERSION, read_checkpoint
from tae.engine import TAEEngine
from test_tae_engine import TinyFrozenVAE, batch, config


@pytest.mark.parametrize('mmap', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('source_kind', ['str', 'path', 'bytesio', 'file'])
def test_tae_serialization_and_file_source_compatibility(tmp_path, monkeypatch, mmap, legacy, source_kind):
    path = tmp_path / 'tae.pt'
    expected = torch.arange(32, dtype=torch.float32).view(4, 8)
    torch.save({'format': FORMAT, 'version': VERSION, 'model': {'example': expected}}, path,
               _use_new_zipfile_serialization=not legacy)
    calls, load = [], torch.load

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', record)
    context = (nullcontext(str(path) if source_kind == 'str' else path)
               if source_kind in {'str', 'path'} else
               BytesIO(path.read_bytes()) if source_kind == 'bytesio' else path.open('rb'))
    with context as source:
        result = read_checkpoint(source, mmap=mmap)
    torch.testing.assert_close(result['model']['example'], expected, rtol=0, atol=0)
    mapped = mmap and source_kind in {'str', 'path'}
    assert [call['mmap'] for call in calls] == ([True, False] if mapped and legacy else [mapped])
    assert all(call['weights_only'] is True and call['map_location'] == 'cpu' for call in calls)


@pytest.mark.parametrize('mmap', [False, True])
@pytest.mark.parametrize('kind', ['unsafe', 'corrupt-zip'])
def test_shared_loader_does_not_retry_security_rejection_or_damaged_zip(tmp_path, monkeypatch, mmap, kind):
    path = tmp_path / 'tae.pt'
    if kind == 'unsafe':
        torch.save({'blocked': object()}, path)
        expected = pickle.UnpicklingError
    else:
        torch.save({'broken': torch.zeros(16)}, path)
        path.write_bytes(path.read_bytes()[:-64])
        expected = RuntimeError
    calls, load = [], torch.load

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', record)
    with pytest.raises(expected):
        load_weights_checkpoint(path, mmap=mmap)
    assert len(calls) == 1
    assert calls[0]['weights_only'] is True


def test_legacy_mmap_fallback_still_rejects_unsafe_objects(tmp_path, monkeypatch):
    path = tmp_path / 'unsafe-legacy.pt'
    torch.save({'blocked': object()}, path, _use_new_zipfile_serialization=False)
    calls, load = [], torch.load

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', record)
    with pytest.raises(pickle.UnpicklingError):
        load_weights_checkpoint(path, mmap=True)
    assert [call['mmap'] for call in calls] == [True, False]
    assert all(call['weights_only'] is True for call in calls)


@pytest.mark.parametrize('payload', [[], torch.zeros(1), {}, {'format': 'me-pytorch', 'version': VERSION},
                                     {'format': FORMAT, 'version': VERSION + 1}])
def test_tae_reader_rejects_invalid_container_or_foreign_format(tmp_path, payload):
    path = tmp_path / 'wrong.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='TAE v1 checkpoint'):
        read_checkpoint(path, mmap=True)


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('source_kind', ['path', 'bytesio', 'file'])
def test_tae_inference_legacy_and_file_sources_match_trained_engine(tmp_path, legacy, source_kind):
    torch.set_num_threads(2)
    worker = TAEEngine(config(), 'cpu', seed=48, vae=TinyFrozenVAE())
    worker.train_step(batch(44), batch(45))
    path = tmp_path / 'tae.pt'
    worker.save(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
    expected = worker.predict(batch(46)[0])
    context = (nullcontext(path) if source_kind == 'path' else
               BytesIO(path.read_bytes()) if source_kind == 'bytesio' else path.open('rb'))
    with context as source:
        inference = TAEEngine.load(source, 'cpu', vae=TinyFrozenVAE(), training=False)
    for actual, reference in zip(inference.predict(batch(46)[0]), expected):
        np.testing.assert_array_equal(actual, reference)
    assert inference.iteration == worker.iteration == 1
    assert inference.optimizer is None
    # Retaining the mapped payload would block this replacement on Windows.
    worker.save(path)
