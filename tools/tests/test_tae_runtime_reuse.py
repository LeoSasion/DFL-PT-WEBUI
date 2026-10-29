"""TAE reuse preserves loss gradients, updates and inference caller state."""
from dataclasses import replace
from pathlib import Path
import random
import sys
from unittest import mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))
from tae.engine import TAEEngine
from tae import losses
from tae.masking import image_loss_mask
from test_tae_engine import TinyFrozenVAE, batch, config


def reference_loss(image, predicted_mask, values, settings):
    """Original independent masked reductions, used as a numerical reference."""
    target, mask, roi = values['target'], values['mask'], values['eyes_mouth']
    coverage = values.get('image_mask', mask)
    error = torch.sqrt((image.float() - target.float()).square() + 1e-6) - .001
    terms = dict(rgb=losses.masked_mean(error, coverage),
        dssim=losses.masked_mean(losses.dssim_map(image, target), coverage),
        edge=losses.edge_loss(image, target, coverage),
        eyes_mouth=losses.masked_mean(error, roi * mask) + losses.edge_loss(image, target, roi * mask),
        mask=F.binary_cross_entropy(predicted_mask.float(), mask.float()))
    return sum(getattr(settings, name + '_weight') * value for name, value in terms.items()), terms


def exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            exact(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize('mode', ['face', 'expanded_feather', 'full'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_loss_shared_denominator_preserves_values_and_output_gradients(mode, dtype):
    torch.set_num_threads(2)
    torch.manual_seed(85)
    mask = torch.rand(2, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1
    values = dict(target=torch.rand(2, 3, 64, 64), mask=mask,
        eyes_mouth=torch.rand_like(mask), image_mask=torch.from_numpy(image_loss_mask(mask.numpy(), mode)))
    settings = replace(config(), training_mask_mode=mode)
    image = torch.rand(2, 3, 64, 64).to(dtype).requires_grad_()
    predicted_mask = torch.rand_like(mask).to(dtype).requires_grad_()
    other_image, other_mask = image.detach().clone().requires_grad_(), predicted_mask.detach().clone().requires_grad_()
    expected, expected_terms = reference_loss(other_image, other_mask, values, settings)
    actual, terms = losses.reconstruction_loss(image, predicted_mask, values, settings)
    exact(terms, expected_terms)
    exact(actual, expected)
    expected.backward()
    actual.backward()
    exact(image.grad, other_image.grad)
    exact(predicted_mask.grad, other_mask.grad)


@pytest.mark.parametrize('mode', ['frozen', 'tail', 'full'])
def test_loss_reuse_preserves_three_complete_updates_and_rng(monkeypatch, mode):
    torch.set_num_threads(2)
    settings = replace(config(), decoder_mode=mode)
    expected = TAEEngine(settings, 'cpu', seed=12, vae=TinyFrozenVAE())
    actual = TAEEngine(settings, 'cpu', seed=12, vae=TinyFrozenVAE())
    import tae.engine as engine_module
    for step in range(3):
        source, destination = batch(21 + step), batch(31 + step)
        monkeypatch.setattr(engine_module, 'reconstruction_loss', reference_loss)
        old = expected.train_step(source, destination)
        old_rng = torch.get_rng_state().clone()
        monkeypatch.setattr(engine_module, 'reconstruction_loss', losses.reconstruction_loss)
        new = actual.train_step(source, destination)
        exact(new, old)
        exact(actual.network.state_dict(), expected.network.state_dict())
        exact(actual.optimizer.state_dict(), expected.optimizer.state_dict())
        exact(torch.get_rng_state(), old_rng)


@pytest.mark.parametrize('mode', ['frozen', 'tail', 'full'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_inference_load_omits_adam_and_preserves_rng_contracts_and_weights(tmp_path, mode, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(2)
    worker = TAEEngine(replace(config(), decoder_mode=mode), device, seed=18, vae=TinyFrozenVAE())
    worker.train_step(batch(25), batch(26))
    path = tmp_path / 'tae.pt'
    worker.save(path)
    resumed = TAEEngine.load(path, device, vae=TinyFrozenVAE())
    expected = resumed.predict(batch(28)[0])
    codec = TinyFrozenVAE()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    python_rng = random.getstate()
    with mock.patch('torch.optim.AdamW', side_effect=AssertionError('inference constructed AdamW')):
        inference = TAEEngine.load(path, device, vae=codec, training=False)
    exact(torch.get_rng_state(), cpu_rng)
    exact(torch.cuda.get_rng_state_all() if cuda_rng else [], cuda_rng)
    assert random.getstate() == python_rng
    assert inference.optimizer is None and inference.data_state is None
    assert not inference.network.training
    assert inference.iteration == resumed.iteration == 1
    assert inference.metadata() == resumed.metadata()
    inference.assert_frozen()
    for left, right in zip(inference.predict(batch(28)[0]), expected):
        np.testing.assert_array_equal(left, right)
    saved_bytes = path.read_bytes()
    for call in [lambda: inference.save(path), lambda: inference.train_step(batch(25), batch(26)),
                 lambda: inference.reconfigure_decoder(replace(inference.config, decoder_mode='tail'))]:
        with pytest.raises(RuntimeError, match='inference|training=True'):
            call()
        assert path.read_bytes() == saved_bytes


@pytest.mark.parametrize('training', [None, 0, 1, 'yes'])
def test_training_flag_rejected_before_model_or_file_access(training):
    with pytest.raises(ValueError, match='boolean'):
        TAEEngine(config(), 'cpu', training=training)
    with pytest.raises(ValueError, match='boolean'):
        TAEEngine.load('does-not-exist.pt', 'cpu', training=training)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('damage', ['model', 'stage-list', 'stage-scalar'])
def test_malformed_checkpoint_containers_are_rejected(tmp_path, training, damage):
    worker = TAEEngine(config(), 'cpu', vae=TinyFrozenVAE())
    path = tmp_path / 'tae.pt'
    worker.save(path)
    payload = torch.load(path, weights_only=True)
    if damage == 'model':
        payload['model'] = []
    else:
        payload['stage_history'][0] = [] if damage == 'stage-list' else 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='checkpoint tensor|stage history'):
        TAEEngine.load(path, 'cpu', vae=TinyFrozenVAE(), training=training)
