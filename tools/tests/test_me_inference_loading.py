"""Inference owns only ME generator weights; checkpoint training stays intact."""
from dataclasses import replace
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
from me_backend import engine as engine_module
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.model_adapter import MEInferenceModel


def settings(**options):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                    d_mask_dims=16, batch_size=1, **options)


def sample(seed):
    rng = np.random.default_rng(seed)
    image = rng.random((1, 3, 64, 64), dtype=np.float32)
    mask = np.ones((1, 1, 64, 64), dtype=np.float32)
    return image, image.copy(), mask, mask.copy()


def exact(left, right):
    if isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            exact(left[key], right[key])
    elif isinstance(right, (tuple, list)):
        assert len(left) == len(right)
        for actual, expected in zip(left, right):
            exact(actual, expected)
    else:
        assert left == right


@pytest.mark.parametrize('options', [
    {},
    {'archi': 'df-ud', 'gan_power': .1, 'gan_dims': 4, 'true_face_power': .1},
    {'gan_power': .1, 'gan_dims': 4, 'retraining_samples': True, 'retraining_every': 1,
     'retraining_capacity': 2, 'optimizer_on_cpu': True},
])
def test_inference_predictions_match_and_do_not_construct_training_state(tmp_path, monkeypatch, options):
    torch.set_num_threads(2)
    source = MEEngine(settings(**options), 'cpu', seed=61)
    batch = sample(12)
    source.train_step(batch, batch)
    checkpoint = tmp_path / 'me.pt'
    source.save(checkpoint, {'test_bookmark': 1})
    expected = source.predict(batch[1])

    def unexpected(*args, **kwargs):
        raise AssertionError('inference constructed training-only state')

    monkeypatch.setattr(engine_module, 'MEOptimizer', unexpected)
    monkeypatch.setattr(engine_module, 'HardSampleReplay', unexpected)
    monkeypatch.setattr(MEEngine, '_build_runtime', unexpected)
    monkeypatch.setattr(MEEngine, '_build_adversarial', unexpected)
    torch.manual_seed(881)
    caller_rng = torch.get_rng_state()
    inference = MEEngine.load(checkpoint, 'cpu', training=False)
    torch.testing.assert_close(torch.get_rng_state(), caller_rng, rtol=0, atol=0)
    for actual, ref in zip(inference.predict(batch[1]), expected):
        np.testing.assert_array_equal(actual, ref)
    assert inference.iteration == source.iteration
    assert inference.optimizer_updates == source.optimizer_updates
    assert inference.data_state is None
    assert inference.optimizer is inference.adversarial is inference.adversarial_optimizer is None
    assert inference.scaler is inference.replay is inference.forward_network is None
    assert not inference.network.training
    assert all(not parameter.requires_grad for parameter in inference.network.parameters())
    for operation in [lambda: inference.train_step(batch, batch),
                      lambda: inference.reconfigure(replace(inference.config, lr=.0001)),
                      lambda: inference.save(tmp_path / 'forbidden/me.pt')]:
        with pytest.raises(RuntimeError, match='inference-only|training=True'):
            operation()
    assert not (tmp_path / 'forbidden').exists()


def test_live_inference_does_not_pin_or_follow_replaced_checkpoint(tmp_path):
    torch.set_num_threads(2)
    source = MEEngine(settings(), 'cpu', seed=7)
    batch = sample(17)
    source.train_step(batch, batch)
    checkpoint = tmp_path / 'me.pt'
    source.save(checkpoint)
    inference = MEEngine.load(checkpoint, 'cpu', training=False)
    expected = inference.predict(batch[1])
    resumed = MEEngine.load(checkpoint, 'cpu')
    before = resumed.train_step(batch, batch)
    # A mapped optimizer/parameter retained by inference would make this atomic
    # replacement fail on Windows, or make its prediction follow new file bytes.
    resumed.save(checkpoint)
    assert resumed.iteration == 2
    for actual, ref in zip(inference.predict(batch[1]), expected):
        np.testing.assert_array_equal(actual, ref)
    loaded = MEEngine.load(checkpoint, 'cpu')
    state = torch.get_rng_state()
    expected_metrics = resumed.train_step(batch, batch)
    torch.set_rng_state(state)
    assert loaded.train_step(batch, batch) == expected_metrics
    exact(loaded.network.state_dict(), resumed.network.state_dict())
    exact(loaded.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert before['iteration'] == 2


@pytest.mark.parametrize('kind', ['format', 'config', 'iteration', 'updates', 'missing-weight', 'nan-weight'])
def test_inference_retains_strict_generator_checkpoint_validation(tmp_path, kind):
    torch.set_num_threads(2)
    checkpoint = tmp_path / 'me.pt'
    MEEngine(settings(), 'cpu', seed=13).save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    if kind == 'format':
        payload['format'] = 'tae-pytorch'
    elif kind == 'config':
        payload['config']['unrecognized_option'] = True
    elif kind == 'iteration':
        payload['iteration'] = -1
    elif kind == 'updates':
        payload['optimizer_updates'] = True
    elif kind == 'missing-weight':
        payload['model'].pop(next(iter(payload['model'])))
    else:
        payload['model'][next(iter(payload['model']))].fill_(float('nan'))
    torch.save(payload, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        MEEngine.load(checkpoint, 'cpu', training=False)


def test_inference_explicitly_ignores_training_auxiliary_state(tmp_path):
    checkpoint = tmp_path / 'me.pt'
    MEEngine(settings(), 'cpu', seed=7).save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload['optimizer'] = {'unrecognized_future_state': torch.ones(16)}
    payload['adversarial'] = {'unused': torch.ones(16)}
    payload['replay'] = {'unused': torch.ones(16)}
    torch.save(payload, checkpoint)
    inference = MEEngine.load(checkpoint, 'cpu', training=False)
    assert all(np.isfinite(value).all() for value in inference.predict(sample(2)[1]))
    with pytest.raises(ValueError, match='optimizer'):
        MEEngine.load(checkpoint, 'cpu')


def test_merger_adapter_requests_inference_only_loading(tmp_path):
    directory = tmp_path / 'model'
    source = MEEngine(settings(), 'cpu', seed=8)
    source.save(directory / 'me.pt')
    adapter = MEInferenceModel(directory, cpu_only=True)
    predictor, shape, _ = adapter.get_MergerConfig()
    face = sample(3)[1][0].transpose(1, 2, 0)
    outputs = predictor(face)
    assert shape == (64, 64, 3)
    assert all(np.isfinite(value).all() for value in outputs)
    assert adapter.engine._inference_only
    assert adapter.engine.optimizer is None


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('source_kind', ['path', 'bytesio', 'file'])
def test_inference_retains_legacy_serialization_and_file_like_support(tmp_path, monkeypatch, legacy, source_kind):
    torch.set_num_threads(2)
    source = MEEngine(settings(), 'cpu', seed=47)
    batch = sample(18)
    source.train_step(batch, batch)
    checkpoint = tmp_path / 'me.pt'
    source.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    torch.save(payload, checkpoint, _use_new_zipfile_serialization=not legacy)
    expected = source.predict(batch[1])
    calls = []
    load = torch.load

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', record)
    context = (nullcontext(checkpoint) if source_kind == 'path'
               else BytesIO(checkpoint.read_bytes()) if source_kind == 'bytesio'
               else checkpoint.open('rb'))
    with context as value:
        inference = MEEngine.load(value, 'cpu', training=False)
    for actual, reference in zip(inference.predict(batch[1]), expected):
        np.testing.assert_array_equal(actual, reference)
    expected_calls = ([True, False] if legacy else [True]) if source_kind == 'path' else [False]
    assert [call['mmap'] for call in calls] == expected_calls
    assert all(call['weights_only'] is True for call in calls)
    source.save(checkpoint)


@pytest.mark.parametrize('kind', ['unsafe', 'corrupt-zip'])
def test_inference_does_not_retry_weights_only_rejection_or_corrupt_zip(tmp_path, monkeypatch, kind):
    checkpoint = tmp_path / 'me.pt'
    if kind == 'unsafe':
        torch.save({'blocked': object()}, checkpoint)
        expected = pickle.UnpicklingError
    else:
        torch.save({'broken': torch.zeros(16)}, checkpoint)
        checkpoint.write_bytes(checkpoint.read_bytes()[:-64])
        expected = RuntimeError
    calls = []
    load = torch.load

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', record)
    with pytest.raises(expected):
        MEEngine.load(checkpoint, 'cpu', training=False)
    assert len(calls) == 1
    assert calls[0]['weights_only'] is True and calls[0]['mmap'] is True
