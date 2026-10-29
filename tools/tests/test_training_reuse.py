"""Synthetic equivalence for omitted work, including full optimizer/RNG state."""
import copy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal' / 'DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))

from me_backend.config import MEConfig
from me_backend.engine import MEEngine, TrainingForward
from tae.engine import TAEEngine
from tae.model import LIAEModel
from test_tae_engine import TinyFrozenVAE, batch, config


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


def small_me(**changes):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                    d_mask_dims=16, batch_size=1, **changes)


@pytest.mark.parametrize('options', [
    {}, {'archi': 'df-ud'}, {'random_warp': False}, {'use_rg': True},
    {'gan_power': .1, 'gan_dims': 4, 'gan_noise': .05, 'lr_dropout': 'y'},
    {'archi': 'df-ud', 'true_face_power': .1},
    {'archi': 'df-ud', 'face_style_power': .1}, {'bg_style_power': .1},
])
def test_me_step_matches_full_swap_forward_gradients_moments_and_rng(options):
    torch.set_num_threads(2)
    settings = small_me(**options)
    expected, actual = (MEEngine(settings, 'cpu', seed=29) for _ in range(2))
    expected.forward_network = TrainingForward(expected.network, False, include_swap=True)
    source, destination = [tuple(np.ascontiguousarray(v[..., :64, :64]) for v in batch(seed))
                           for seed in (51, 52)]
    for step in range(2):
        state = torch.get_rng_state()
        before = expected.train_step(source, destination)
        after_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        after = actual.train_step(source, destination)
        exact(after, before)
        exact(actual.network.state_dict(), expected.network.state_dict())
        exact(actual.optimizer.state_dict(), expected.optimizer.state_dict())
        exact(actual.adversarial.state_dict(), expected.adversarial.state_dict())
        if actual.adversarial_optimizer:
            exact(actual.adversarial_optimizer.state_dict(), expected.adversarial_optimizer.state_dict())
        exact(torch.get_rng_state(), after_rng)
        for left, right in zip(actual.network.parameters(), expected.network.parameters()):
            exact(left.grad, right.grad)


def test_me_swap_policy_uses_effective_pretraining_and_reconfiguration():
    settings = small_me(archi='df-ud', pretrain=True, face_style_power=.1,
                        bg_style_power=.1, gan_power=.1, true_face_power=.1)
    worker = MEEngine(settings, 'cpu', seed=29)
    assert worker.forward_network.include_swap is False
    assert worker.adversarial.enabled is False
    worker.reconfigure(replace(settings, pretrain=False))
    assert worker.forward_network.include_swap is True
    worker.reconfigure(replace(settings, pretrain=False, face_style_power=0., bg_style_power=0.))
    assert worker.forward_network.include_swap is False
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert 'swap' in worker.network(image, image), 'public forward remains backward compatible'
    with pytest.raises(ValueError, match='requires the swap branch'):
        worker.network(image, image, face_style=True, include_swap=False)


def legacy_forward(model, source, destination):
    result = {**model.reconstruct_src(source), **model.reconstruct_dst(destination)}
    latent = model._encode(destination)
    swapped = model._decode(model._features(latent, 'swap'), destination.shape[-2:])
    return {**result, 'swap': swapped['image'], 'swap_mask': swapped['mask']}


def legacy_predict(model, destination):
    latent = model._encode(destination)
    swapped = model._decode(model._features(latent, 'swap'), destination.shape[-2:])
    return {'swap': swapped['image'].clamp(0, 1), 'swap_mask': swapped['mask'],
            'dst_mask': model._mask(model._features(latent, 'dst'), destination.shape[-2:])}


@pytest.mark.parametrize('mode', ['frozen', 'tail', 'full'])
def test_tae_inference_reuses_dst_encoding_and_inter_without_changing_outputs(mode):
    torch.set_num_threads(2)
    torch.manual_seed(91)
    codec = TinyFrozenVAE()
    codec.set_decoder_mode(mode)
    model = LIAEModel(codec, 8, 8, 64).eval()
    source, destination = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    calls = []
    handle = model.inter_AB.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with torch.inference_mode():
            expected = legacy_forward(model, source, destination)
            assert len(calls) == len(codec.encoded_inputs) == 3
            calls.clear(); codec.encoded_inputs.clear()
            actual = model(source, destination, include_swap=True)
            assert len(calls) == len(codec.encoded_inputs) == 2
            exact(actual, expected)
            calls.clear(); codec.encoded_inputs.clear()
            expected = legacy_predict(model, destination)
            assert len(calls) == 2
            calls.clear()
            actual = model.predict(destination)
            assert len(calls) == 1
            exact(actual, expected)
    finally:
        handle.remove()


def test_tae_gradient_enabled_swap_keeps_independent_branch_accumulation():
    torch.set_num_threads(2)
    torch.manual_seed(92)
    expected = LIAEModel(TinyFrozenVAE(), 8, 8, 64)
    actual = copy.deepcopy(expected)
    source, destination = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    original = legacy_forward(expected, source, destination)
    optimized = actual(source, destination, include_swap=True)
    exact(optimized, original)
    sum(value.square().mean() for value in original.values()).backward()
    sum(value.square().mean() for value in optimized.values()).backward()
    for left, right in zip(actual.parameters(), expected.parameters()):
        exact(left.grad, right.grad)


@pytest.mark.parametrize('mask_mode', ['face', 'expanded_feather', 'full'])
def test_tae_preview_preparation_skips_unused_masks_and_keeps_inputs_checked(monkeypatch, mask_mode):
    torch.set_num_threads(2)
    settings = replace(config(), training_mask_mode=mask_mode)
    worker = TAEEngine(settings, 'cpu', seed=27, vae=TinyFrozenVAE())
    source, destination = batch(28), batch(29)
    with torch.inference_mode():
        src, dst = worker._batch(source), worker._batch(destination)
        expected = legacy_forward(worker.network.eval(), src['target'], dst['target'])
        expected = {key: (value[:, [2, 1, 0]].clamp(0, 1) if value.shape[1] == 3 else value).float()
                    for key, value in expected.items()}
    import tae.engine as module
    monkeypatch.setattr(module, 'image_loss_mask', lambda *_: pytest.fail('preview does not need an image-loss mask'))
    exact(worker.preview_outputs(source, destination), expected)
    for index in range(4):
        damaged = [value.copy() for value in source]
        damaged[index].flat[0] = np.nan
        with pytest.raises(FloatingPointError, match='Nonfinite DFL batch'):
            worker.preview_outputs(damaged, destination)
    damaged = [value.copy() for value in source]
    damaged[3].flat[0] = 4
    with pytest.raises(ValueError, match='encoded eyes/mouth'):
        worker.preview_outputs(damaged, destination)
    if mask_mode == 'expanded_feather':
        for count in (1, 2):
            damaged = [np.repeat(value, count, axis=0) for value in source]
            damaged[2][-1].fill(.49)
            with pytest.raises(ValueError, match=r'Expanded training mask needs a face region with weight >= 0\.5'):
                worker.preview_outputs(damaged, destination)
