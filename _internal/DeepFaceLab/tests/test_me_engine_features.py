import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from me_backend.config import MEConfig
from me_backend.data import AlignedDataset
from me_backend.engine import MEEngine, resolve_devices, validate_scaler_state
from me_backend.web_bridge import load_training_engine
from tests.me_fixtures import make_aligned

torch.set_num_threads(2)


def small(**options):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16, batch_size=1, **options)


def batch(tmp_path, config):
    data = AlignedDataset(make_aligned(tmp_path / 'src'), config, True, 12)
    try:
        return data.batch()
    finally:
        data.close()


def assert_state_equal(expected, actual):
    assert set(expected) == set(actual)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_resume_configuration_is_checked_and_changes_are_explicit(tmp_path):
    config = small()
    engine = MEEngine(config, 'cpu', 3)
    model = tmp_path / 'model'
    engine.save(model / 'me.pt')
    update = tmp_path / 'fine-tune.json'
    update.write_text(json.dumps(dict(lr=.0001, random_warp=False, use_rg=True)))
    args = SimpleNamespace(model=model, config=update, resume=True, device='cpu', allow_config_change=False)
    with pytest.raises(ValueError, match='Resume config differs'):
        load_training_engine(args)
    args.allow_config_change = True
    resumed = load_training_engine(args)
    assert resumed.config.lr == .0001 and resumed.config.use_rg
    assert all(not parameter.requires_grad for parameter in resumed.network.inter_AB.parameters())
    assert_state_equal(engine.network.state_dict(), resumed.network.state_dict())
    update.write_text(json.dumps(dict(resolution=128)))
    with pytest.raises(ValueError, match='network fields'):
        load_training_engine(args)
    args.config = None
    args.allow_config_change = False
    args.batch_size = 2
    with pytest.raises(ValueError, match='Resume config differs'):
        load_training_engine(args)


def test_fine_tune_preserves_moments_and_unfreezes_intermediate(tmp_path):
    config = small(random_warp=False)
    engine = MEEngine(config, 'cpu', 1)
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    engine.reconfigure(replace(config, random_warp=True, lr=.00001, optimizer_on_cpu=True))
    assert all(parameter.requires_grad for parameter in engine.network.inter_AB.parameters())
    engine.save(tmp_path / 'changed.pt')
    loaded = MEEngine.load(tmp_path / 'changed.pt', 'cpu')
    result = loaded.train_step(sample, sample)
    assert result['iteration'] == 2
    assert all(value.device.type == 'cpu' for state in loaded.optimizer.state.values() for value in state.values())
    with pytest.raises(ValueError, match='reset-optimizer'):
        loaded.reconfigure(replace(loaded.config, adabelief=False))
    loaded.reconfigure(replace(loaded.config, adabelief=False), reset_optimizer=True)
    assert loaded.iteration == 2 and loaded.optimizer_updates == 0
    loaded.save(tmp_path / 'reset.pt')
    assert MEEngine.load(tmp_path / 'reset.pt', 'cpu').train_step(sample, sample)['iteration'] == 3


def test_hard_sample_replay_and_exact_resume(tmp_path):
    config = small(retraining_samples=True, retraining_every=2, retraining_capacity=4, lr_dropout='cpu')
    engine = MEEngine(config, 'cpu', 13)
    sample = batch(tmp_path, config)
    first = engine.train_step(sample, sample)
    assert not first['replayed']
    engine.save(tmp_path / 'replay.pt')
    expected = engine.train_step(sample, sample)
    weights = copy.deepcopy(engine.network.state_dict())
    resumed = MEEngine.load(tmp_path / 'replay.pt', 'cpu')
    assert expected == resumed.train_step(sample, sample)
    assert expected['replayed'] and resumed.optimizer_updates == 3
    assert_state_equal(weights, resumed.network.state_dict())
    resumed.save(tmp_path / 'replay-next.pt')
    assert MEEngine.load(tmp_path / 'replay-next.pt', 'cpu').optimizer_updates == 3


@pytest.mark.parametrize('archi,true_face', [('liae-ud', 0.), ('df-ud', .1)])
def test_adversarial_checkpoint_exact_resume(tmp_path, archi, true_face):
    config = small(archi=archi, use_rg=True, gan_power=.1, gan_dims=4, gan_patch_size=8,
                   gan_noise=.1, true_face_power=true_face)
    engine = MEEngine(config, 'cpu', 5)
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    engine.save(tmp_path / 'adversarial.pt')
    expected = engine.train_step(sample, sample)
    network = copy.deepcopy(engine.network.state_dict())
    discriminator = copy.deepcopy(engine.adversarial.state_dict())
    resumed = MEEngine.load(tmp_path / 'adversarial.pt', 'cpu')
    assert expected == resumed.train_step(sample, sample)
    assert_state_equal(network, resumed.network.state_dict())
    assert_state_equal(discriminator, resumed.adversarial.state_dict())


def test_pretraining_disables_identity_adversaries_and_supports_transition(tmp_path):
    config = small(archi='df-ud', pretrain=True, true_face_power=.1, gan_power=.1, gan_dims=4, face_style_power=.1)
    engine = MEEngine(config, 'cpu')
    assert not engine.adversarial.enabled and engine.training_config.face_style_power == 0
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    engine.reconfigure(replace(config, pretrain=False))
    assert engine.adversarial.enabled
    assert np.isfinite(engine.train_step(sample, sample)['discriminator_loss'])


def test_optimizer_reset_preserves_trained_discriminator_weights(tmp_path):
    config = small(gan_power=.1, gan_dims=4)
    engine = MEEngine(config, 'cpu')
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    discriminator = copy.deepcopy(engine.adversarial.state_dict())
    engine.reconfigure(replace(config, adabelief=False), reset_optimizer=True)
    assert_state_equal(discriminator, engine.adversarial.state_dict())
    assert not engine.adversarial_optimizer.state


def test_policy_changes_preserve_training_rng(tmp_path):
    config = small(gan_power=.1, gan_dims=4, lr_dropout='cpu')
    engine = MEEngine(config, 'cpu', 17)
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    rng = torch.get_rng_state().clone()
    engine.reconfigure(replace(config, lr=.00001, use_rg=True, optimizer_on_cpu=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_adding_trueface_preserves_existing_gan_weights_and_moments(tmp_path):
    config = small(archi='df-ud', gan_power=.1, gan_dims=4)
    engine = MEEngine(config, 'cpu')
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    weights = copy.deepcopy(engine.adversarial.gan.state_dict())
    parameter = next(engine.adversarial.gan.parameters())
    moments = copy.deepcopy(engine.adversarial_optimizer.state[parameter])
    engine.reconfigure(replace(config, true_face_power=.1))
    assert_state_equal(weights, engine.adversarial.gan.state_dict())
    new_parameter = next(engine.adversarial.gan.parameters())
    assert_state_equal(moments, engine.adversarial_optimizer.state[new_parameter])
    engine.save(tmp_path / 'new-trueface.pt')
    assert np.isfinite(MEEngine.load(tmp_path / 'new-trueface.pt', 'cpu').train_step(sample, sample)['src_loss'])


def test_checkpoint_rejects_inconsistent_settings_and_corrupt_replay(tmp_path):
    config = small(retraining_samples=True)
    engine = MEEngine(config, 'cpu')
    sample = batch(tmp_path, config)
    engine.train_step(sample, sample)
    engine.save(tmp_path / 'valid.pt')
    payload = torch.load(tmp_path / 'valid.pt', weights_only=True)
    payload['optimizer']['param_groups'][0]['lr'] = .001
    torch.save(payload, tmp_path / 'bad-settings.pt')
    with pytest.raises(ValueError, match='settings differ'):
        MEEngine.load(tmp_path / 'bad-settings.pt', 'cpu')
    payload = torch.load(tmp_path / 'valid.pt', weights_only=True)
    payload['replay']['source'][0][1][0].resize_(1, 64, 64)
    torch.save(payload, tmp_path / 'bad-replay.pt')
    with pytest.raises(ValueError, match='replay shape'):
        MEEngine.load(tmp_path / 'bad-replay.pt', 'cpu')


def test_invalid_devices_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        resolve_devices('cuda:0,0')
    with pytest.raises(ValueError, match='must be'):
        resolve_devices('mps')


@pytest.mark.parametrize('key,value', [('scale', float('nan')), ('scale', 0.), ('growth_factor', 1.),
                                    ('backoff_factor', 1.), ('growth_interval', 0), ('_growth_tracker', -1)])
def test_invalid_amp_scaler_checkpoint_is_rejected(key, value):
    state = dict(scale=1024., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=0)
    state[key] = value
    with pytest.raises(ValueError, match='AMP scaler'):
        validate_scaler_state(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for real FP16 updates')
def test_fp16_rg_cpu_optimizer_real_cuda_update_and_resume(tmp_path):
    config = small(use_fp16=True, use_rg=True, optimizer_on_cpu=True, lr_dropout='cpu', gan_power=.1, gan_dims=4)
    engine = MEEngine(config, 'cuda:0', 42)
    sample = batch(tmp_path, config)
    before = copy.deepcopy(engine.network.state_dict())
    result = engine.train_step(sample, sample)
    assert np.isfinite(result['src_loss']) and engine.optimizer_updates == 1
    assert any(not torch.equal(before[key], value) for key, value in engine.network.state_dict().items())
    assert all(value.device.type == 'cpu' for state in engine.optimizer.state.values() for value in state.values())
    assert all(parameter.dtype == torch.float32 for parameter in engine.network.parameters())
    engine.save(tmp_path / 'amp.pt')
    resumed = MEEngine.load(tmp_path / 'amp.pt', 'cuda:0')
    assert resumed.scaler.state_dict() == engine.scaler.state_dict()
    assert np.isfinite(resumed.train_step(sample, sample)['src_loss'])
    cpu = MEEngine.load(tmp_path / 'amp.pt', 'cpu')
    assert np.isfinite(cpu.predict(sample[1])[0]).all()
    with pytest.raises(ValueError, match='requires CUDA'):
        cpu.train_step(sample, sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP overflow recovery')
def test_amp_overflow_retries_at_scale_one_and_keeps_scaler_on_config_change(tmp_path):
    config = small(use_fp16=True, use_rg=True)
    engine = MEEngine(config, 'cuda:0', 9)
    sample = batch(tmp_path, config)
    engine.scaler.load_state_dict(dict(scale=8., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=0))
    calls = []
    def inject_overflow(gradient):
        calls.append(1)
        return torch.full_like(gradient, float('inf')) if len(calls) <= 3 else gradient
    hook = next(engine.network.parameters()).register_hook(inject_overflow)
    result = engine.train_step(sample, sample)
    hook.remove()
    assert len(calls) == 4 and result['iteration'] == 1 and engine.optimizer_updates == 1
    assert engine.scaler.get_scale() == 1.
    scaler = engine.scaler.state_dict()
    engine.reconfigure(replace(config, lr=.0001))
    assert scaler == engine.scaler.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP failure recovery')
def test_amp_can_recover_after_discriminator_failure_before_step(tmp_path, monkeypatch):
    config = small(use_fp16=True, gan_power=.1, gan_dims=4)
    engine = MEEngine(config, 'cuda:0', 8)
    sample = batch(tmp_path, config)
    original = engine.adversarial.discriminator_loss
    with monkeypatch.context() as patch:
        patch.setattr(engine.adversarial, 'discriminator_loss', lambda *args: torch.tensor(float('nan'), device='cuda:0'))
        with pytest.raises(FloatingPointError, match='discriminator'):
            engine.train_step(sample, sample)
    assert engine.iteration == engine.optimizer_updates == 0
    assert np.isfinite(engine.train_step(sample, sample)['src_loss'])
    engine.save(tmp_path / 'valid-amp.pt')
    payload = torch.load(tmp_path / 'valid-amp.pt', weights_only=True, map_location='cpu')
    payload['scaler']['scale'] = float('nan')
    torch.save(payload, tmp_path / 'bad-amp.pt')
    with pytest.raises(ValueError, match='AMP scaler'):
        MEEngine.load(tmp_path / 'bad-amp.pt', 'cpu')
