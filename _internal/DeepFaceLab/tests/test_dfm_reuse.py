"""Synthetic export reuse regressions; mocked TRT tests publish no fake bundle."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm_export import exporter, quantization
from me_backend.config import MEConfig
from tae.model import LIAEModel
from me_fixtures import make_aligned


class CalibrationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        value = self.conv(image.permute(0, 3, 1, 2))
        return value[:, :1], value, value[:, :1]


@pytest.mark.parametrize('retain', [False, True])
def test_calibration_retains_at_most_one_owned_cpu_input(monkeypatch, retain):
    network = CalibrationNetwork()
    image = torch.full((1, 8, 8, 3), .125)
    first = image.clone()

    def inputs(*args):
        for index in range(8):
            image.fill_(.125 + index / 16)
            yield image, {'name': str(index), 'sha256': str(index) * 64}

    monkeypatch.setattr(quantization, 'calibration_faces', inputs)
    maxima, records, sample = quantization._calibrate(network, None, None, 8, 'cpu',
        {'conv': 'fp8'}, retain_sample=retain)
    assert network.calls == 8 and len(records) == 8
    assert maxima == {'conv': float(image.max())}
    assert not network.conv._forward_pre_hooks
    assert all(parameter.grad is None for parameter in network.parameters())
    if retain:
        assert sample.device.type == 'cpu' and sample.data_ptr() != image.data_ptr()
        torch.testing.assert_close(sample, first, rtol=0, atol=0)
    else:
        assert sample is None


def test_calibration_nonfinite_output_removes_observers(monkeypatch):
    network = CalibrationNetwork()
    with torch.no_grad():
        network.conv.weight.fill_(float('nan'))
    monkeypatch.setattr(quantization, 'calibration_faces', lambda *args:
        iter([(torch.ones(1, 8, 8, 3), {'name': 'synthetic'})]))
    with pytest.raises(FloatingPointError, match='Nonfinite FP16 calibration output'):
        quantization._calibrate(network, None, None, 8, 'cpu', {'conv': 'fp8'}, retain_sample=True)
    assert not network.conv._forward_pre_hooks


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_calibration_invalid_activation_is_reported_before_loading_another_face(monkeypatch, invalid):
    class SanitizingNetwork(CalibrationNetwork):
        def forward(self, image):
            return tuple(torch.nan_to_num(value) for value in super().forward(image))

    network = SanitizingNetwork()
    loaded = []

    def inputs(*args):
        loaded.append(1)
        yield torch.full((1, 8, 8, 3), invalid), {'name': 'invalid'}
        raise AssertionError('invalid activation must fail before the next sample')

    monkeypatch.setattr(quantization, 'calibration_faces', inputs)
    with pytest.raises(FloatingPointError, match='Nonfinite calibration activation: conv'):
        quantization._calibrate(network, None, None, 8, 'cpu', {'conv': 'fp8'}, retain_sample=True)
    assert loaded == [1]
    assert not network.conv._forward_pre_hooks


class IdentityNetwork(nn.Module):
    def predict(self, image):
        return image, image[:, :1], image[:, :1]


def test_extreme_smoke_reuses_first_calibrated_snapshot_without_another_scan(tmp_path, monkeypatch):
    """Exercise orchestration on CPU; GPU gates remain in the real exporter."""
    directory = tmp_path / 'model'
    directory.mkdir()
    checkpoint = directory / 'me.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity; model loading is mocked')
    faces = make_aligned(tmp_path / 'faces', count=8, size=64)
    config = MEConfig(resolution=64)
    monkeypatch.setattr(exporter, 'load_network', lambda *args:
        (IdentityNetwork(), config, 7, checkpoint))
    scans = []
    calibration_faces = quantization.calibration_faces

    def once(*args):
        scans.append(args)
        assert len(scans) == 1, 'smoke must reuse calibration, not reread the faceset'
        yield from calibration_faces(*args)

    monkeypatch.setattr(quantization, 'calibration_faces', once)
    observed = []

    def cpu_calibration(module, config, example, graph_path, directory, limit, device,
                        threshold, *, capture_calibration_sample):
        _, records, sample = quantization._calibrate(module, config, directory, limit,
            'cpu', {}, retain_sample=True)
        observed.append(sample.clone())
        capture_calibration_sample(sample)
        graph, parameters = exporter.trace_graph(module, example, graph_path)
        sidecars = exporter.attach_weights(graph, parameters, graph_path, threshold=threshold)
        return graph, sidecars, {'recipe': 'test-only-no-acceleration', 'calibration': records}

    monkeypatch.setattr(quantization, 'export_extreme', cpu_calibration)
    from dfm_export import runtime

    def build_stub(graph, device):
        # The source may move while a long engine build runs. The calibrated
        # input and its receipt must still describe the same earlier snapshot.
        faces.rename(tmp_path / 'moved-faces')
        engine, layers = Path(str(graph) + '.engine'), Path(str(graph) + '.layers.json')
        engine.write_bytes(b'test-only stub, never published')
        layers.write_text('{}')
        return engine, layers, {'test_only': True}

    class SmokeStub:
        def __init__(self, *args):
            pass

        def __call__(self, image):
            np.testing.assert_array_equal(image, np.repeat(observed[0].numpy(), len(image), 0))
            with torch.no_grad():
                return [value.numpy() for value in exporter.DFMNetwork(IdentityNetwork())(
                    torch.from_numpy(image))]

    monkeypatch.setattr(runtime, 'build_engine', build_stub)
    monkeypatch.setattr(runtime, 'TensorRTRunner', SmokeStub)
    published = []

    def inspect_without_publishing(output, files):
        import json
        receipt = json.loads(files[-1].read_text(encoding='utf8'))
        assert receipt['precision']['engine_smoke']['max_abs_error'] == [0., 0., 0.]
        assert receipt['precision']['calibration'][0]['name'] == '000.jpg'
        published.append(receipt)

    monkeypatch.setattr(exporter, 'publish_bundle', inspect_without_publishing)
    output = exporter.export_dfm(directory, mode='extreme', calibration_dir=faces,
        calibration_samples=8, device='cpu')
    assert len(scans) == len(observed) == len(published) == 1
    assert checkpoint.read_bytes() == b'synthetic checkpoint identity; model loading is mocked'
    assert not output.exists(), 'mocked engine output must never be published'


class TinyRGBA(nn.Module):
    """Test-only RGBA codec; validates routing, not native TAE weight quality."""
    latent_channels = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.AvgPool2d(16), nn.Conv2d(4, 4, 1))
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=16, mode='nearest'),
                                     nn.Conv2d(4, 4, 1), nn.Sigmoid())
        self.requires_grad_(False)

    def assert_frozen(self):
        assert all(not parameter.requires_grad for parameter in self.parameters())

    def encode(self, image):
        return self.encoder(torch.cat((image, torch.ones_like(image[:, :1])), 1))

    def decode(self, latent):
        return self.decoder(latent)[:, :3]


@pytest.mark.parametrize('failure', ['batch-row', 'shape', 'nonfinite'])
def test_engine_smoke_rejects_invalid_dynamic_batches(failure):
    calls = []
    reference = [np.full((1, 4, 4, channels), .3, np.float32) for channels in (1, 3, 1)]

    def runner(image):
        batch = len(image)
        calls.append(batch)
        values = [np.repeat(value, batch, 0) for value in reference]
        if batch > 1:
            if failure == 'batch-row': values[0][1] = .5
            elif failure == 'shape': values[0] = values[0][:1]
            else: values[0][1] = np.nan
        return values

    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        exporter._engine_smoke(runner, np.ones((1, 4, 4, 3), np.float32), reference)
    assert calls == [1, 2]


def test_engine_smoke_records_max_batch_and_single_batch_replay():
    calls = []
    reference = [np.full((1, 4, 4, channels), .3, np.float32) for channels in (1, 3, 1)]

    def runner(image):
        calls.append(len(image))
        return [np.repeat(value, len(image), 0) for value in reference]

    report = exporter._engine_smoke(runner, np.ones((1, 4, 4, 3), np.float32), reference)
    assert calls == [1, 2, 4, 1]
    assert report['finite'] and report['max_abs_error'] == [0., 0., 0.]
    assert report['batch_consistency']['absolute_tolerance'] == .02
    assert [value['batch'] for value in report['batch_consistency']['batches']] == calls


@pytest.mark.parametrize('version,capability,dynamic,fp4,expected', [
    ('10.16.1.11', (12, 0), True, True, 0),
    ('10.16.1.11', (10, 0), True, True, 3),
    ('11.0.0', (12, 0), True, True, 3),
    ('10.16.1.11', (12, 0), False, True, 3),
    ('10.16.1.11', (12, 0), True, False, 3),
])
def test_builder_policy_only_disables_affected_nvfp4_fusion(version, capability, dynamic, fp4, expected):
    from dfm_export.runtime import _builder_optimization_level
    trt = SimpleNamespace(__version__=version, LayerType=SimpleNamespace(DYNAMIC_QUANTIZE='dynamic'), fp4='fp4')
    layer = SimpleNamespace(type='dynamic' if dynamic else 'other',
        get_output=lambda index: SimpleNamespace(dtype='fp4' if fp4 else 'fp8'))
    network = SimpleNamespace(num_layers=1, get_layer=lambda index: layer)
    assert _builder_optimization_level(network, trt, capability) == expected


def test_tae_dfm_reuses_inter_ab_and_external_graph_runs_dynamic_batches(tmp_path):
    torch.manual_seed(27)
    network = LIAEModel(TinyRGBA(), inter_dims=8, hidden_dims=8, resolution=64).eval()
    module = exporter.DFMNetwork(network, family='TAE').eval()
    calls = []
    handle = network.inter_AB.register_forward_hook(lambda *args: calls.append(1))
    data = torch.rand(2, 64, 64, 3)
    with torch.no_grad():
        actual = module(data)
    handle.remove()
    assert calls == [1]
    rgb = data.permute(0, 3, 1, 2)[:, [2, 1, 0]]
    with torch.no_grad():
        src, dst = network.reconstruct(rgb, 'src'), network.reconstruct(rgb, 'dst')
    expected = (dst['mask'], src['image'][:, [2, 1, 0]].clamp(0, 1), src['mask'])
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference.permute(0, 2, 3, 1), rtol=0, atol=0)
    graph_path = tmp_path / 'synthetic.dfm'
    graph, parameters = exporter.trace_graph(module, data[:1], graph_path)
    sidecars = exporter.attach_weights(graph, parameters, graph_path, threshold=1)
    assert len(sidecars) == 1
    onnx.save_model(graph, str(graph_path))
    onnx.checker.check_model(str(graph_path))
    session = ort.InferenceSession(str(graph_path), providers=['CPUExecutionProvider'])
    for batch in [1, 2]:
        outputs = session.run(None, {exporter.INPUT: data[:batch].numpy()})
        for value, reference in zip(outputs, actual):
            np.testing.assert_allclose(value, reference[:batch].numpy(), atol=2e-6, rtol=2e-4)
