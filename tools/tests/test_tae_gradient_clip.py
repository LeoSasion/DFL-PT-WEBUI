"""The clipping norm is also the successful-step gradient finite check."""
import copy
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))
import tae.engine as module
from test_tae_engine import batch, engine
from test_tae_engine import TinyFrozenVAE, config
from dataclasses import replace


def committed(worker):
    return dict(weights={name: value.detach().clone() for name, value in worker.network.state_dict().items()},
        iteration=worker.iteration, updates=worker.optimizer_updates, stage=worker.stage_updates,
        learning_rates=[group['lr'] for group in worker.optimizer.param_groups],
        optimizer=copy.deepcopy(worker.optimizer.state_dict()),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        rng=torch.get_rng_state().clone())


def unchanged_after_failure(worker, before):
    assert worker.iteration == before['iteration']
    assert worker.optimizer_updates == before['updates']
    assert worker.stage_updates == before['stage']
    assert [group['lr'] for group in worker.optimizer.param_groups] == before['learning_rates']
    actual_optimizer = worker.optimizer.state_dict()
    assert actual_optimizer['param_groups'] == before['optimizer']['param_groups']
    assert actual_optimizer['state'].keys() == before['optimizer']['state'].keys()
    for index, entry in actual_optimizer['state'].items():
        for name, value in entry.items():
            torch.testing.assert_close(value, before['optimizer']['state'][index][name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in worker.network.parameters())
    torch.testing.assert_close(torch.get_rng_state(), before['rng'], rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if before['cuda_rng'] else [], before['cuda_rng']):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in worker.network.state_dict().items():
        torch.testing.assert_close(value, before['weights'][name], rtol=0, atol=0)


def test_valid_step_uses_one_clipping_check_without_duplicate_gradient_scan():
    torch.set_num_threads(2)
    worker = engine()
    clip = torch.nn.utils.clip_grad_norm_
    with mock.patch.object(module, 'gradients_are_finite', side_effect=AssertionError('duplicate scan')), \
         mock.patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=clip) as clipped:
        result = worker.train_step(batch(51), batch(52))
    assert result['iteration'] == result['optimizer_updates'] == 1
    assert np.isfinite(result['grad_norm'])
    clipped.assert_called_once()
    assert clipped.call_args.kwargs == {'error_if_nonfinite': True}


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('prior_steps', [0, 1])
def test_nonfinite_gradient_keeps_error_type_and_rejected_step_is_recoverable(invalid, prior_steps):
    worker = engine()
    for _ in range(prior_steps):
        worker.train_step(batch(51), batch(52))
    before = committed(worker)
    hook = worker.network.inter_AB.compress.weight.register_hook(lambda value: torch.full_like(value, invalid))
    try:
        with mock.patch.object(module, 'gradients_are_finite', wraps=module.gradients_are_finite) as checked:
            with pytest.raises(FloatingPointError, match='nonfinite or disconnected'):
                worker.train_step(batch(53), batch(54))
            checked.assert_called_once()
    finally:
        hook.remove()
    unchanged_after_failure(worker, before)
    assert worker.train_step(batch(53), batch(54))['iteration'] == prior_steps + 1


def test_finite_large_gradient_keeps_original_norm_overflow_error_and_state():
    worker = engine()
    before = committed(worker)
    hook = worker.network.inter_AB.compress.weight.register_hook(lambda value: torch.full_like(value, 1e38))
    try:
        with mock.patch.object(module, 'gradients_are_finite', wraps=module.gradients_are_finite) as checked:
            with pytest.raises(RuntimeError, match='total norm.*non-finite'):
                worker.train_step(batch(55), batch(56))
            checked.assert_called_once()
    finally:
        hook.remove()
    unchanged_after_failure(worker, before)
    assert worker.train_step(batch(55), batch(56))['iteration'] == 1


@pytest.mark.parametrize('disconnected', [False, True])
def test_missing_or_empty_gradients_reject_before_clipping(disconnected):
    worker = engine()
    parameters = worker.parameters
    if disconnected:
        parameters[0].requires_grad_(False)
    else:
        worker.parameters = []
    before = committed(worker)
    with mock.patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=AssertionError('clip without gradients')):
        with pytest.raises(FloatingPointError, match='disconnected'):
            worker.train_step(batch(57), batch(58))
    unchanged_after_failure(worker, before)
    worker.parameters = parameters
    parameters[0].requires_grad_(True)
    assert worker.train_step(batch(57), batch(58))['iteration'] == 1


def test_zero_gradients_are_valid_and_do_not_trigger_failure_scan():
    worker = engine()
    hooks = [parameter.register_hook(torch.zeros_like) for parameter in worker.parameters]
    try:
        with mock.patch.object(module, 'gradients_are_finite', side_effect=AssertionError('duplicate scan')):
            result = worker.train_step(batch(59), batch(60))
    finally:
        for hook in hooks:
            hook.remove()
    assert result['iteration'] == 1 and result['grad_norm'] == 0


def test_unrelated_clipping_error_is_not_caught_or_rescanned():
    worker = engine()
    before = committed(worker)
    error = RuntimeError('unrelated clipping failure')
    with mock.patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=error), \
         mock.patch.object(module, 'gradients_are_finite', side_effect=AssertionError('unrelated failure scan')):
        with pytest.raises(RuntimeError) as caught:
            worker.train_step(batch(61), batch(62))
        assert caught.value is error
    unchanged_after_failure(worker, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf'), 1e38])
def test_cuda_clipping_rejection_preserves_prior_adam_and_all_rng_then_recovers(invalid):
    worker = module.TAEEngine(replace(config(), precision='bf16', decoder_mode='tail'),
        'cuda:0', seed=17, vae=TinyFrozenVAE())
    worker.train_step(batch(71), batch(72))
    before = committed(worker)
    hook = worker.network.inter_AB.compress.weight.register_hook(lambda value: torch.full_like(value, invalid))
    error = RuntimeError if np.isfinite(invalid) else FloatingPointError
    try:
        with pytest.raises(error):
            worker.train_step(batch(73), batch(74))
    finally:
        hook.remove()
    unchanged_after_failure(worker, before)
    assert worker.train_step(batch(73), batch(74))['iteration'] == 2
