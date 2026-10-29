"""CPU-only safety/equivalence checks; synthetic batches and an explicit tiny codec."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal' / 'DeepFaceLab'))
sys.path.insert(0, str(Path(__file__).parent))

from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.losses import me_losses
from me_backend.tensor_checks import all_finite, gradients_are_finite, scalar_metrics
from tae.engine import TAEEngine
from tae.losses import reconstruction_loss
from test_tae_engine import TinyFrozenVAE, batch as tae_batch, config as tae_config

torch.set_num_threads(2)


def make_me(**changes):
    return MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                             d_mask_dims=16, batch_size=1, **changes), 'cpu', seed=7)


def make_tae(**changes):
    return TAEEngine(replace(tae_config(), **changes), 'cpu', seed=7, vae=TinyFrozenVAE())


def sample(family, seed):
    values = tae_batch(seed)
    return values if family == 'tae' else tuple(np.ascontiguousarray(value[..., :64, :64]) for value in values)


def weights(engine):
    return {name: value.detach().clone() for name, value in engine.network.state_dict().items()}


def assert_weights(engine, expected):
    for name, value in engine.network.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def assert_rejected(engine, before):
    assert engine.iteration == engine.optimizer_updates == 0
    assert not engine.optimizer.state
    assert all(parameter.grad is None for parameter in engine.network.parameters())
    assert_weights(engine, before)


def test_finite_reduction_and_metrics_preserve_nonfinite_and_scalar_values():
    values = [torch.arange(24, dtype=torch.float32).view(4, 6)[:, ::2], torch.tensor(-0.)]
    assert bool(all_finite(values))
    for invalid in [float('nan'), float('inf'), -float('inf')]:
        assert not bool(all_finite([*values, torch.tensor(invalid)]))
    parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    assert not gradients_are_finite(parameters)
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    assert gradients_are_finite(parameters), 'zero gradients are valid, missing gradients are not'
    parameters[1].grad = None
    assert not gradients_are_finite(parameters)
    assert not gradients_are_finite([])
    metrics = {'tiny': torch.tensor(1e-30), 'negative': torch.tensor(-.25, dtype=torch.float64),
               'half': torch.tensor(.125, dtype=torch.float16)}
    assert scalar_metrics(metrics) == {name: float(value) for name, value in metrics.items()}


@pytest.mark.parametrize('archi,random_warp', [('liae-ud', True), ('liae-ud', False), ('df-ud', True)])
def test_me_step_matches_direct_fp32_backward_and_optimizer(archi, random_warp):
    source, destination = sample('me', 1), sample('me', 2)
    expected = make_me(archi=archi, random_warp=random_warp)
    actual = make_me(archi=archi, random_warp=random_warp)
    src, dst = expected.tensors(source), expected.tensors(destination)
    outputs = expected.network(src[0], dst[0])
    src_loss, dst_loss = me_losses(expected.training_config, src, dst, outputs)
    (src_loss + dst_loss).mean().backward()
    expected.optimizer.step()
    result = actual.train_step(source, destination)
    assert result == {'iteration': 1, 'src_loss': float(src_loss.mean().detach()),
                      'dst_loss': float(dst_loss.mean().detach())}
    assert_weights(actual, weights(expected))


@pytest.mark.parametrize('decoder_mode', ['frozen', 'tail', 'full'])
def test_tae_step_preserves_paired_loss_metrics_clipping_and_adam(decoder_mode):
    source, destination = tae_batch(3), tae_batch(4)
    expected, actual = make_tae(decoder_mode=decoder_mode), make_tae(decoder_mode=decoder_mode)
    expected.network.train()
    metrics = {}
    lr = expected.config.learning_rate / expected.config.warmup_steps
    for group in expected.optimizer.param_groups:
        group['lr'] = lr * group['lr_scale']
    for domain, values in [('src', source), ('dst', destination)]:
        image, target, mask, encoded = (torch.as_tensor(value) for value in values)
        prepared = {'input': image[:, [2, 1, 0]], 'target': target[:, [2, 1, 0]],
                    'mask': mask, 'image_mask': mask, 'eyes_mouth': (encoded - 1).clamp(0, 1)}
        output = expected.network.reconstruct(prepared['input'], domain=domain)
        loss, terms = reconstruction_loss(output['image'], output['mask'], prepared, expected.config)
        (loss * .5).backward()
        metrics[domain + '_loss'] = float(loss.detach())
        metrics.update({domain + '_' + name: float(value.detach()) for name, value in terms.items()})
    metrics['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(
        expected.parameters, expected.config.grad_clip, error_if_nonfinite=True))
    expected.optimizer.step()
    result = actual.train_step(source, destination)
    assert {name: result[name] for name in metrics} == metrics
    assert result['iteration'] == result['optimizer_updates'] == result['stage_updates'] == 1
    assert result['amp_retries'] == 0 and result['encoder_frozen']
    assert_weights(actual, weights(expected))
    expected.assert_frozen()
    actual.assert_frozen()
    for parameter, reference in zip(actual.parameters, expected.parameters):
        for key, value in actual.optimizer.state[parameter].items():
            torch.testing.assert_close(value, expected.optimizer.state[reference][key], rtol=0, atol=0)


@pytest.mark.parametrize('as_tensor', [False, True])
@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_tae_rejects_nonfinite_cpu_input_before_upload(monkeypatch, as_tensor, index, invalid):
    engine = make_tae()
    values = list(tae_batch(5))
    values[index].flat[0] = invalid
    if as_tensor:
        values = [torch.from_numpy(value) for value in values]
    monkeypatch.setattr(engine, 'tensors', lambda _: pytest.fail('invalid input reached device transfer'))
    with pytest.raises(FloatingPointError, match='Nonfinite DFL batch'):
        engine._batch(values)


@pytest.mark.parametrize('index,invalid,message', [
    (0, -0.01, r'\[0,1\]'), (1, 1.01, r'\[0,1\]'), (2, -0.01, r'\[0,1\]'),
    (2, 1.01, r'\[0,1\]'), (3, -0.01, 'encoded'), (3, 3.002, 'encoded'),
])
def test_tae_rejects_out_of_range_cpu_input_before_upload(monkeypatch, index, invalid, message):
    engine = make_tae()
    values = list(tae_batch(6))
    values[index].flat[0] = invalid
    monkeypatch.setattr(engine, 'tensors', lambda _: pytest.fail('invalid input reached device transfer'))
    with pytest.raises(ValueError, match=message):
        engine._batch(values)


def test_tae_cpu_validation_uses_the_same_fp32_conversion_and_prepared_masks():
    engine = make_tae(training_mask_mode='expanded_feather')
    arrays = list(tae_batch(7))
    arrays[0] = arrays[0].astype(np.float64)
    arrays[0].flat[0] = 1 + 1e-9  # rounds to valid FP32, exactly as device conversion did.
    actual = engine._batch(arrays)
    tensors = [torch.from_numpy(value) for value in arrays]
    expected = engine._batch(tensors)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert float(actual['input'][0, 2, 0, 0]) == 1.
    arrays[0].flat[0] = np.finfo(np.float64).max
    with pytest.raises(FloatingPointError, match='Nonfinite DFL batch'):
        engine._batch(arrays)


def test_tae_shape_errors_are_rejected_before_device_transfer(monkeypatch):
    engine = make_tae()
    monkeypatch.setattr(engine, 'tensors', lambda _: pytest.fail('invalid shape reached device transfer'))
    for index in range(4):
        values = list(tae_batch(7))
        values[index] = values[index][..., :-1]
        with pytest.raises(ValueError, match='NCHW shape'):
            engine._batch(values)
    with pytest.raises(ValueError, match='Expected DFL'):
        engine._batch(tae_batch(7)[:3])


@pytest.mark.parametrize('family', ['me', 'tae'])
@pytest.mark.parametrize('failure', ['missing', 'nan', 'inf'])
def test_bad_gradients_never_update_and_next_valid_step_recovers(family, failure):
    engine = make_tae() if family == 'tae' else make_me()
    source, destination = sample(family, 8), sample(family, 9)
    before = weights(engine)
    parameter = next(parameter for parameter in engine.network.parameters() if parameter.requires_grad)
    if failure == 'missing':
        hook = parameter.register_post_accumulate_grad_hook(lambda value: setattr(value, 'grad', None))
    else:
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float(failure)))
    try:
        with pytest.raises(FloatingPointError, match='gradients'):
            engine.train_step(source, destination)
    finally:
        hook.remove()
    assert_rejected(engine, before)
    assert engine.train_step(source, destination)['iteration'] == 1


@pytest.mark.parametrize('failure', ['missing', 'nan', 'inf'])
def test_me_bad_discriminator_gradients_reject_both_updates_and_recover(failure):
    engine = make_me(gan_power=.1, gan_dims=4, gan_patch_size=8)
    before = weights(engine)
    discriminator_before = {name: value.detach().clone() for name, value in engine.adversarial.state_dict().items()}
    parameter = next(engine.adversarial.parameters())
    if failure == 'missing':
        hook = parameter.register_post_accumulate_grad_hook(lambda value: setattr(value, 'grad', None))
    else:
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float(failure)))
    try:
        with pytest.raises(FloatingPointError, match='gradients'):
            engine.train_step(sample('me', 8), sample('me', 9))
    finally:
        hook.remove()
    assert_rejected(engine, before)
    assert not engine.adversarial_optimizer.state
    for name, value in engine.adversarial.state_dict().items():
        torch.testing.assert_close(value, discriminator_before[name], rtol=0, atol=0)
    assert all(parameter.requires_grad and parameter.grad is None for parameter in engine.adversarial.parameters())
    assert engine.train_step(sample('me', 8), sample('me', 9))['iteration'] == 1


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
@pytest.mark.parametrize('output_name', ['image', 'mask'])
def test_tae_bad_destination_output_clears_source_gradients_and_recovers(monkeypatch, invalid, output_name):
    engine = make_tae()
    before = weights(engine)
    original = engine.network.reconstruct

    def corrupt(value, *, domain):
        result = original(value, domain=domain)
        if domain == 'dst':
            result[output_name] = torch.full_like(result[output_name], invalid)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(engine.network, 'reconstruct', corrupt)
        with pytest.raises(FloatingPointError, match='Nonfinite loss input'):
            engine.train_step(tae_batch(10), tae_batch(11))
    assert_rejected(engine, before)
    assert engine.train_step(tae_batch(10), tae_batch(11))['iteration'] == 1


@pytest.mark.parametrize('name', ['image', 'predicted_mask', 'target', 'mask', 'eyes_mouth'])
def test_tae_loss_rejects_each_nonfinite_input_and_empty_individual_masks(name):
    values = {key: torch.full((2, 3 if key in ('image', 'target') else 1, 16, 16), .5)
              for key in ('image', 'predicted_mask', 'target', 'mask', 'eyes_mouth')}
    values[name][0, 0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='Nonfinite loss input'):
        reconstruction_loss(values['image'], values['predicted_mask'], values, tae_config())
    values[name][0, 0, 0, 0] = .5
    values['mask'][1].zero_()
    with pytest.raises(ValueError, match='nonempty'):
        reconstruction_loss(values['image'], values['predicted_mask'], values, tae_config())


def test_me_cpu_amp_overflow_retries_same_pair_without_partial_update():
    # Exercise the real GradScaler retry protocol on CPU; this is not CUDA/FP16 validation.
    engine, reference = make_me(), make_me()
    source, destination = sample('me', 12), sample('me', 13)
    engine.scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=8.)
    parameter = next(parameter for parameter in engine.network.parameters() if parameter.requires_grad)
    calls = []

    def one_overflow(gradient):
        calls.append(1)
        return torch.full_like(gradient, float('inf')) if len(calls) == 1 else gradient

    hook = parameter.register_hook(one_overflow)
    try:
        result = engine.train_step(source, destination)
    finally:
        hook.remove()
    expected = reference.train_step(source, destination)
    assert engine.iteration == engine.optimizer_updates == 1
    assert engine.scaler.get_scale() == 4.
    assert len(calls) == 2
    assert result == expected
    assert_weights(engine, weights(reference))
