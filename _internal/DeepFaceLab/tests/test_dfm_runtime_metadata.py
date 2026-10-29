"""Static engine metadata reuse with fresh dynamic output buffers."""
from collections import Counter
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm_export.runtime import TensorRTRunner, build_engine


def fake_runner(tmp_path, monkeypatch):
    names = ('in_face:0', 'out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0')
    queries, tensors = Counter(), {}
    real_as_tensor, real_empty = torch.as_tensor, torch.empty

    def as_tensor(value, *, device, dtype):
        tensor = real_as_tensor(value, dtype=dtype)
        tensors[tensor.data_ptr()] = tensor
        return tensor

    def empty(shape, *, device, dtype):
        tensor = real_empty(shape, dtype=dtype)
        tensors[tensor.data_ptr()] = tensor
        return tensor

    class Context:
        unresolved = False
        execute_ok = True
        addresses = {}

        def set_input_shape(self, name, shape):
            assert name == names[0]
            self.shape = shape
            return len(shape) == 4 and 1 <= shape[0] <= 4 and shape[1:] == (16, 16, 3)

        def get_tensor_shape(self, name):
            queries['shape'] += 1
            return (-1 if self.unresolved else self.shape[0], 16, 16, 3 if name == names[2] else 1)

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer

        def execute_async_v3(self, stream):
            if not self.execute_ok:
                return False
            x = tensors[self.addresses[names[0]]]
            for name, value in zip(names[1:], (x[..., :1] * .25 + .125, x * .5, x[..., :1] * .125 + .0625)):
                tensors[self.addresses[name]].copy_(value)
            return True

    class Engine:
        num_io_tensors = 4
        context = Context()

        def create_execution_context(self): return self.context
        def get_tensor_name(self, index):
            queries['name'] += 1
            return names[index]
        def get_tensor_mode(self, name):
            queries['mode'] += 1
            return 'in' if name == names[0] else 'out'
        def get_tensor_dtype(self, name):
            queries['dtype'] += 1
            return np.float32

    class Logger:
        WARNING = 1
        def __init__(self, level): pass

    engine = Engine()
    trt = SimpleNamespace(Logger=Logger, TensorIOMode=SimpleNamespace(OUTPUT='out'),
        Runtime=lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda data: engine), nptype=lambda dtype: dtype)
    stream = SimpleNamespace(device=torch.device('cuda:0'), cuda_stream=0, synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, 'tensorrt', trt)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'Stream', lambda device: stream)
    monkeypatch.setattr(torch, 'as_tensor', as_tensor)
    monkeypatch.setattr(torch, 'empty', empty)
    path = tmp_path / 'synthetic.engine'
    path.write_bytes(b'isolated test engine; no published model bundle')
    return TensorRTRunner(path), engine.context, queries


def test_runner_reads_immutable_metadata_once_and_keeps_dynamic_results_independent(tmp_path, monkeypatch):
    runner, _, queries = fake_runner(tmp_path, monkeypatch)
    assert queries == {'name': 4, 'mode': 4, 'dtype': 3}
    first = runner(np.full((1, 16, 16, 3), .5, np.float32))
    retained = [value.copy() for value in first]
    second = runner(np.zeros((3, 16, 16, 3), np.float32))
    assert queries == {'name': 4, 'mode': 4, 'dtype': 3, 'shape': 6}
    for value, channels, expected in zip(second, (1, 3, 1), (.125, 0., .0625)):
        assert value.shape == (3, 16, 16, channels)
        np.testing.assert_array_equal(value, expected)
    for value, saved, expected in zip(first, retained, (.25, .25, .125)):
        np.testing.assert_array_equal(value, saved)
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize('failure', ['profile', 'unresolved-shape', 'execution', 'nonfinite'])
def test_runner_preserves_dynamic_shape_execution_and_finite_failures(tmp_path, monkeypatch, failure):
    runner, context, _ = fake_runner(tmp_path, monkeypatch)
    data = np.zeros((1, 16, 16, 3), np.float32)
    if failure == 'profile': data = np.zeros((5, 16, 16, 3), np.float32)
    if failure == 'unresolved-shape': context.unresolved = True
    if failure == 'execution': context.execute_ok = False
    if failure == 'nonfinite': data.fill(np.nan)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        runner(data)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('tensorrt') is None,
                    reason='requires CUDA and the optional TensorRT runtime')
def test_real_runner_waits_for_current_producer_stream_and_preserves_batch_outputs(tmp_path):
    class Contract(torch.nn.Module):
        def forward(self, x):
            return x[..., :1] * .25 + .125, x * .5, x[..., :1] * .125 + .0625

    graph = tmp_path / 'synthetic.dfm'
    with torch.no_grad():
        torch.onnx.export(Contract(), torch.zeros(1, 16, 16, 3), str(graph),
            input_names=['in_face:0'], output_names=['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0'],
            dynamic_axes={name: {0: 'batch'} for name in
                ('in_face:0', 'out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0')},
            opset_version=17, dynamo=False)
    engine, _, _ = build_engine(graph, 'cuda:0')
    runner = TensorRTRunner(engine, 'cuda:0')
    for batch in (1, 2, 4, 1):
        data = np.full((batch, 16, 16, 3), .5, np.float32)
        for actual, expected in zip(runner(data), (.25, .25, .125)):
            np.testing.assert_array_equal(actual, expected)
    # Warm same-shape allocations before delaying the producer, so allocator
    # synchronization cannot conceal a missing stream dependency.
    value = torch.zeros(1, 16, 16, 3, device='cuda:0')
    torch.cuda.synchronize()
    producer = torch.cuda.Stream(device='cuda:0')
    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        value.fill_(.5)
        outputs = runner(value)
    for actual, expected in zip(outputs, (.25, .25, .125)):
        np.testing.assert_array_equal(actual, expected)
