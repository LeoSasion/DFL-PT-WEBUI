"""CPU-only checks of offload, DFL update equations and optimizer continuity."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from me_backend.optimizer import MEOptimizer, optimizer_settings


def config(**changes):
    values = dict(lr=5e-5, adabelief=True, clipgrad=False,
                  lr_dropout='n', optimizer_on_cpu=False)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('adabelief', [True, False])
@pytest.mark.parametrize('offload', [True, False])
@pytest.mark.parametrize('dropout', ['n', 'y', 'cpu'])
def test_fp32_updates_match_existing_equations_exactly(adabelief, offload, dropout):
    parameter = torch.nn.Parameter(torch.linspace(-.2, .3, 37))
    expected = parameter.detach().clone()
    optimizer = MEOptimizer([parameter], config(adabelief=adabelief,
        optimizer_on_cpu=offload, lr_dropout=dropout))
    variance, momentum = torch.zeros_like(parameter), torch.zeros_like(parameter)
    for iteration in range(4):
        gradient = torch.linspace(-.4, .7, 37) * (iteration+1)
        parameter.grad = gradient.clone()
        if adabelief:
            momentum.mul_(.9).add_(gradient, alpha=.1)
            variance.mul_(.999).addcmul_(gradient-momentum, gradient-momentum, value=.001)
            delta = momentum / (variance.sqrt()+1e-6)
        else:
            variance.mul_(.9).addcmul_(gradient, gradient, value=.1)
            delta = gradient / (variance.sqrt()+1e-6)
        lr = 5e-5
        if dropout != 'n':
            lr *= (math.cos(iteration*2*math.pi/500)+1)/2
            torch.manual_seed(400+iteration)
            delta *= torch.rand_like(parameter) < .3
        expected.add_(delta, alpha=-lr)
        torch.manual_seed(400+iteration)
        optimizer.step()
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        torch.testing.assert_close(optimizer.state[parameter]['variance'], variance, atol=0, rtol=0)
        optimizer.validate_state(iteration+1, require_complete=True)
        if adabelief:
            torch.testing.assert_close(optimizer.state[parameter]['momentum'], momentum, atol=0, rtol=0)
        assert all(value.device.type == 'cpu' for value in optimizer.state[parameter].values())


def test_cpu_dropout_explicitly_uses_cpu_random(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(12))
    optimizer = MEOptimizer([parameter], config(lr_dropout='cpu'))
    parameter.grad = torch.ones_like(parameter)
    calls = []
    original = torch.rand

    def observed(*args, **kwargs):
        calls.append(kwargs['device'])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'rand', observed)
    monkeypatch.setattr(torch, 'rand_like', lambda *_args, **_kwargs: pytest.fail('CPU dropout used parameter-device randomness'))
    optimizer.step()
    assert calls == ['cpu']


@pytest.mark.parametrize('adabelief', [True, False])
def test_cpu_offload_exact_resume_and_legacy_groups(adabelief):
    parameter = torch.nn.Parameter(torch.zeros(101))
    optimizer = MEOptimizer([parameter], config(adabelief=adabelief,
        lr_dropout='cpu', optimizer_on_cpu=True))
    torch.manual_seed(31)
    parameter.grad = torch.linspace(-1., 1., 101)
    optimizer.step()
    weights = parameter.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()
    optimizer.step()
    expected = parameter.detach().clone()
    resumed_parameter = torch.nn.Parameter(weights)
    resumed = MEOptimizer([resumed_parameter], config(adabelief=adabelief))
    resumed.load_state_dict(state)
    torch.set_rng_state(rng)
    resumed_parameter.grad = parameter.grad.clone()
    resumed.step()
    torch.testing.assert_close(resumed_parameter, expected, atol=0, rtol=0)
    resumed.validate_state(2, require_complete=True)
    assert all(value.device.type == 'cpu' for value in resumed.state[resumed_parameter].values())
    # Existing checkpoints did not have offload/dropout placement fields.
    legacy = copy.deepcopy(state)
    legacy['param_groups'][0].pop('optimizer_on_cpu')
    legacy['param_groups'][0].pop('dropout_on_cpu')
    resumed.load_state_dict(legacy)
    assert not resumed.param_groups[0]['optimizer_on_cpu']


def test_configure_preserves_moments_steps_and_rng():
    parameter = torch.nn.Parameter(torch.zeros(8))
    optimizer = MEOptimizer([parameter], config())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()
    optimizer.configure(config(lr=1e-4, lr_dropout='cpu', optimizer_on_cpu=True, clipgrad=True))
    assert torch.equal(rng, torch.get_rng_state())
    assert optimizer.param_groups[0]['iteration'] == 1
    for name, value in before['state'][0].items():
        torch.testing.assert_close(optimizer.state[parameter][name], value, atol=0, rtol=0)
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert optimizer.param_groups[0]['dropout_on_cpu']
    assert optimizer.param_groups[0]['optimizer_on_cpu']
    assert optimizer.param_groups[0]['clipnorm'] == 1.
    with pytest.raises(ValueError, match='explicit optimizer reset'):
        optimizer.configure(config(adabelief=False))
    assert optimizer.param_groups[0]['adabelief']
    assert optimizer.param_groups[0]['iteration'] == 1


@pytest.mark.parametrize('change', [dict(optimizer_on_cpu=1), dict(lr_dropout='gpu'), dict(lr=float('nan')), dict(adabelief=1)])
def test_invalid_settings_are_rejected_without_mutation(change):
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = MEOptimizer([parameter], config())
    before = copy.deepcopy(optimizer.param_groups[0])
    with pytest.raises(ValueError):
        optimizer.configure(config(**change))
    assert {k:v for k,v in optimizer.param_groups[0].items() if k != 'params'} == {k:v for k,v in before.items() if k != 'params'}


@pytest.mark.parametrize('invalid', ['shape', 'nonfinite', 'negative', 'dtype', 'device', 'keys'])
def test_invalid_state_rejected(invalid):
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = MEOptimizer([parameter], config(optimizer_on_cpu=True))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    if invalid == 'shape':
        state['variance'] = torch.zeros(4)
    elif invalid == 'nonfinite':
        state['variance'][0] = float('nan')
    elif invalid == 'negative':
        state['variance'][0] = -1
    elif invalid == 'dtype':
        state['variance'] = state['variance'].double()
    elif invalid == 'device':
        state['variance'] = torch.zeros(3, device='meta')
    else:
        state['extra'] = torch.zeros(3)
    with pytest.raises(ValueError, match='optimizer|variance'):
        optimizer.validate_state(1)


def test_previously_frozen_parameters_may_have_no_state():
    active = torch.nn.Parameter(torch.zeros(3))
    frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    optimizer = MEOptimizer([active, frozen], config())
    active.grad = torch.ones_like(active)
    optimizer.step()
    optimizer.validate_state(1, require_complete=True)
    frozen.requires_grad_(True)
    optimizer.validate_state(1)
    with pytest.raises(ValueError, match='Incomplete optimizer'):
        optimizer.validate_state(1, require_complete=True)


def test_settings_helper_rejects_config_errors():
    assert optimizer_settings(config(lr_dropout='cpu'))['dropout_on_cpu']
    with pytest.raises(ValueError, match='lr_dropout'):
        optimizer_settings(config(lr_dropout='unknown'))


@pytest.mark.parametrize('invalid', ['dtype', 'unknown', 'duplicate', 'group', 'nonfinite'])
def test_loaded_state_is_checked_before_torch_cast_or_mutation(invalid):
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = MEOptimizer([parameter], config())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    raw = copy.deepcopy(optimizer.state_dict())
    if invalid == 'dtype':
        raw['state'][0]['variance'] = raw['state'][0]['variance'].double()
    elif invalid == 'unknown':
        raw['state'][999] = raw['state'].pop(0)
    elif invalid == 'duplicate':
        raw['param_groups'][0]['params'] = [0,0]
    elif invalid == 'group':
        raw['param_groups'][0]['dropout_on_cpu'] = 1
    else:
        raw['state'][0]['variance'][0] = float('inf')
    original = optimizer.state[parameter]['variance'].clone()
    with pytest.raises(ValueError):
        optimizer.load_state_dict(raw)
    assert optimizer.param_groups[0]['iteration'] == 1
    torch.testing.assert_close(optimizer.state[parameter]['variance'], original, atol=0, rtol=0)
