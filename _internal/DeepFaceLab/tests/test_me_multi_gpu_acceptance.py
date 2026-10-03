"""Real hardware gate; one-card hosts run the baseline and skip the second card."""
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import me_backend.multi_gpu_acceptance as acceptance
from me_backend.config import MEConfig
from me_backend.data import AlignedDataset
from me_backend.engine import MEEngine
from me_backend.multi_gpu_acceptance import _assess_dual_run, _optimizer_digest, _selection, run_acceptance
from tests.me_fixtures import make_aligned


torch.set_num_threads(2)


def _small():
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16,
                    d_mask_dims=16, batch_size=2)


def test_gpu_selection_and_batch_size_fail_early(tmp_path):
    with pytest.raises(ValueError, match='distinct'):
        _selection('cuda:0,0')
    with pytest.raises(ValueError, match='two GPU indexes'):
        _selection('cpu')
    with pytest.raises(ValueError, match='batch_size'):
        run_acceptance(MEConfig(batch_size=1), tmp_path, tmp_path)


def test_optimizer_digest_survives_cpu_checkpoint_roundtrip(tmp_path):
    config = _small()
    with AlignedDataset(make_aligned(tmp_path / 'source'), config, True, 3) as data:
        sample = data.batch()
    engine = MEEngine(config, 'cpu', 5)
    engine.train_step(sample, sample)
    before = _optimizer_digest(engine.optimizer)
    path = tmp_path / 'me.pt'
    engine.save(path)
    resumed = MEEngine.load(path, 'cpu')
    assert before == _optimizer_digest(resumed.optimizer)
    assert _optimizer_digest(engine.adversarial_optimizer) is None


@pytest.mark.parametrize('fault,failed_check', [
    ('zero_secondary_samples', 'scatter_both_gpus'),
    ('no_secondary_backward', 'backward_both_gpus'),
    ('no_secondary_memory', 'secondary_cuda_allocations'),
    ('no_primary_memory', 'gpu_memory_accounting'),
    ('no_secondary_reserved_memory', 'gpu_memory_accounting'),
    ('optimizer_restore_mismatch', 'dual_gpu_resume'),
    ('scaler_restore_mismatch', 'dual_gpu_resume'),
    ('incomplete_loss_history', 'finite_training_outputs'),
    ('missing_adversarial_loss', 'finite_training_outputs'),
    ('loss_drift', 'single_dual_numerical_parity'),
    ('weight_drift', 'single_dual_numerical_parity'),
    ('prediction_drift', 'single_dual_numerical_parity'),
    ('nonfinite_earlier_loss', 'finite_training_outputs'),
    ('nonfinite_measured_generator_loss', 'finite_training_outputs'),
    ('nonfinite_measured_discriminator_loss', 'finite_training_outputs'),
    ('nonfinite_probe_discriminator_loss', 'finite_training_outputs'),
    ('nonfinite_continued_discriminator_loss', 'dual_gpu_resume'),
    ('zero_throughput', 'finite_throughput'),
    ('nonfinite_throughput', 'finite_throughput'),
    ('zero_step_time', 'finite_throughput'),
    ('nonfinite_step_time', 'finite_throughput'),
    ('incomplete_step_times', 'finite_throughput'),
    ('inconsistent_throughput', 'finite_throughput'),
])
def test_dual_gpu_acceptance_gate_rejects_false_positives(fault, failed_check):
    """Check the verdict logic on one-card hosts; this does not emulate CUDA."""
    base = dict(checkpoint_bytes=100, loaded_exact_weights=True,
                loaded_exact_optimizer=True, loaded_exact_adversarial_optimizer=True,
                loaded_exact_scaler=True,
                iteration=3, loaded_iteration=3, optimizer_updates=3,
                loaded_optimizer_updates=3, continued_finite=True,
                resume_iteration=4, resume_optimizer_updates=4,
                continued_step={'src_loss': 1., 'dst_loss': 1.,
                                'generator_loss': 1., 'discriminator_loss': 1.,
                                'replayed': False}, steps=2, step_ms=[100., 100.],
                median_step_ms=100., images_per_second=20.,
                cuda_peak_allocated_bytes={'0': 100},
                cuda_peak_reserved_bytes={'0': 200},
                losses=[{'src_loss': 1., 'dst_loss': 1.,
                         'generator_loss': 1., 'discriminator_loss': 1.},
                        {'src_loss': 1., 'dst_loss': 1.,
                         'generator_loss': 1., 'discriminator_loss': 1.}])
    single = dict(base)
    dual = dict(base, losses=[dict(loss) for loss in base['losses']],
                continued_step=dict(base['continued_step']),
                step_ms=list(base['step_ms']),
                cuda_peak_allocated_bytes={'0': 100, '1': 100},
                cuda_peak_reserved_bytes={'0': 200, '1': 200},
                scatter_reduce_probe={
                    'forwards': [{'device': 0, 'samples': 1}, {'device': 1, 'samples': 1}],
                    'backwards': [{'device': 0, 'abs_sum': 1.}, {'device': 1, 'abs_sum': 1.}],
                    'primary_gradient_abs_sum': 2.,
                    'step': {'src_loss': 1., 'dst_loss': 1.,
                             'generator_loss': 1., 'discriminator_loss': 1.},
                })
    single_weights = {'weight': torch.ones(4)}
    dual_weights = {'weight': torch.ones(4)}
    single_prediction = (np.zeros((2, 1, 1, 3), dtype=np.float32),)
    dual_prediction = (np.zeros((2, 1, 1, 3), dtype=np.float32),)
    if fault == 'zero_secondary_samples':
        dual['scatter_reduce_probe']['forwards'] = [
            {'device': 0, 'samples': 2}, {'device': 1, 'samples': 0}]
    elif fault == 'no_secondary_backward':
        dual['scatter_reduce_probe']['backwards'][1]['abs_sum'] = 0.
    elif fault == 'no_secondary_memory':
        dual['cuda_peak_allocated_bytes']['1'] = 0
    elif fault == 'no_primary_memory':
        dual['cuda_peak_allocated_bytes']['0'] = 0
    elif fault == 'no_secondary_reserved_memory':
        dual['cuda_peak_reserved_bytes']['1'] = 0
    elif fault == 'optimizer_restore_mismatch':
        dual['loaded_exact_optimizer'] = False
    elif fault == 'scaler_restore_mismatch':
        dual['loaded_exact_scaler'] = False
    elif fault == 'incomplete_loss_history':
        dual['losses'].pop()
    elif fault == 'missing_adversarial_loss':
        for step in single['losses'] + dual['losses']:
            step.update(generator_loss=1., discriminator_loss=1.)
        single['continued_step'].update(generator_loss=1., discriminator_loss=1.)
        dual['continued_step'].update(generator_loss=1., discriminator_loss=1.)
        dual['scatter_reduce_probe']['step'].update(generator_loss=1., discriminator_loss=1.)
        dual['losses'][0].pop('discriminator_loss')
    elif fault == 'loss_drift':
        dual['losses'][-1]['src_loss'] += .01
    elif fault == 'weight_drift':
        dual_weights['weight'] += .01
    elif fault == 'prediction_drift':
        dual_prediction[0][:] = .01
    elif fault == 'nonfinite_earlier_loss':
        dual['losses'][0]['src_loss'] = math.nan
    elif fault == 'nonfinite_measured_generator_loss':
        dual['losses'][0]['generator_loss'] = math.nan
    elif fault == 'nonfinite_measured_discriminator_loss':
        dual['losses'][0]['discriminator_loss'] = math.nan
    elif fault == 'nonfinite_probe_discriminator_loss':
        dual['scatter_reduce_probe']['step']['discriminator_loss'] = math.nan
    elif fault == 'nonfinite_continued_discriminator_loss':
        dual['continued_step']['discriminator_loss'] = math.nan
    elif fault == 'zero_throughput':
        single['images_per_second'] = 0.
    elif fault == 'nonfinite_throughput':
        dual['images_per_second'] = math.nan
    elif fault == 'zero_step_time':
        dual['step_ms'][0] = 0.
    elif fault == 'nonfinite_step_time':
        dual['step_ms'][0] = math.nan
    elif fault == 'incomplete_step_times':
        dual['step_ms'].pop()
    elif fault == 'inconsistent_throughput':
        dual['images_per_second'] = 40.
    config = replace(_small(), gan_power=.1) if fault == 'missing_adversarial_loss' else _small()
    checks, _, speedup = _assess_dual_run(config, single, dual, (0, 1),
                                   single_weights, dual_weights,
                                   single_prediction, dual_prediction, 'parity')
    assert checks[failed_check] is False
    assert not all(checks.values())
    if 'throughput' in fault or 'step_time' in fault or fault == 'incomplete_step_times':
        assert speedup is None
    if fault not in ('loss_drift', 'weight_drift', 'prediction_drift'):
        smoke_checks, _, _ = _assess_dual_run(config, single, dual, (0, 1),
                                             single_weights, dual_weights,
                                             single_prediction, dual_prediction, 'smoke')
        if fault.startswith('nonfinite_') and fault not in ('nonfinite_throughput', 'nonfinite_step_time'):
            assert smoke_checks['finite_training_outputs'] is False
        assert not all(smoke_checks.values())


def test_dual_gpu_gate_accepts_valid_observation_and_require_dual_rejects_partial(
        tmp_path, monkeypatch):
    base = dict(checkpoint_bytes=100, loaded_exact_weights=True,
                loaded_exact_optimizer=True, loaded_exact_adversarial_optimizer=True,
                loaded_exact_scaler=True,
                iteration=3, loaded_iteration=3, optimizer_updates=3,
                loaded_optimizer_updates=3, continued_finite=True,
                resume_iteration=4, resume_optimizer_updates=4,
                continued_step={'src_loss': 1., 'dst_loss': 1., 'replayed': False},
                steps=1, step_ms=[100.], median_step_ms=100., images_per_second=20.,
                cuda_peak_allocated_bytes={'0': 100},
                cuda_peak_reserved_bytes={'0': 200},
                losses=[{'src_loss': 1., 'dst_loss': 1.}])
    dual = dict(base, cuda_peak_allocated_bytes={'0': 100, '1': 100},
                cuda_peak_reserved_bytes={'0': 200, '1': 200},
                scatter_reduce_probe={
                    'forwards': [{'device': 0, 'samples': 1}, {'device': 1, 'samples': 1}],
                    'backwards': [{'device': 0, 'abs_sum': 1.}, {'device': 1, 'abs_sum': 1.}],
                    'primary_gradient_abs_sum': 2.,
                    'step': {'src_loss': 1., 'dst_loss': 1.},
                })
    weights = {'weight': torch.ones(4)}
    prediction = (np.zeros((2, 1, 1, 3), dtype=np.float32),)
    checks, _, _ = _assess_dual_run(_small(), base, dual, (0, 1),
                                   weights, weights, prediction, prediction, 'parity')
    assert all(checks.values())
    monkeypatch.setattr(acceptance, 'run_acceptance', lambda *args, **kwargs: {
        'status': 'partial', 'reason': 'Second CUDA card unavailable'})
    output = tmp_path / 'partial.json'
    with pytest.raises(SystemExit) as error:
        acceptance.main(['--src', str(tmp_path), '--dst', str(tmp_path),
                         '--warmup', '1', '--steps', '1', '--require-dual',
                         '--output', str(output)])
    assert error.value.code == 2
    assert '"status": "partial"' in output.read_text(encoding='utf-8')
    monkeypatch.setattr(acceptance, 'run_acceptance', lambda *args, **kwargs: {
        'status': 'failed', 'speedup': math.nan})
    with pytest.raises(SystemExit) as error:
        acceptance.main(['--src', str(tmp_path), '--dst', str(tmp_path),
                         '--warmup', '1', '--steps', '1', '--output', str(output)])
    assert error.value.code == 2
    assert '"speedup": null' in output.read_text(encoding='utf-8')


@pytest.mark.parametrize('fault,failed_check', [
    ('nonfinite_step_time', 'single_gpu_timing'),
    ('zero_primary_memory', 'single_gpu_memory_accounting'),
    ('nonfinite_measured_loss', 'single_gpu_finite_training_outputs'),
    ('incomplete_loss_history', 'single_gpu_finite_training_outputs'),
    ('nonfinite_prediction', 'single_gpu_finite_training_outputs'),
    ('nonfinite_weight', 'single_gpu_finite_training_outputs'),
    ('missing_adversarial_loss', 'single_gpu_finite_training_outputs'),
    ('scaler_restore_mismatch', 'single_gpu_resume'),
])
def test_partial_report_rejects_invalid_single_gpu_measurement(
        tmp_path, monkeypatch, fault, failed_check):
    run = dict(checkpoint_bytes=1, loaded_exact_weights=True,
               loaded_exact_optimizer=True, loaded_exact_adversarial_optimizer=True,
               loaded_exact_scaler=True,
               iteration=1, loaded_iteration=1, optimizer_updates=1,
               loaded_optimizer_updates=1, continued_finite=True,
               continued_step={'src_loss': 1., 'dst_loss': 1.},
               resume_iteration=2, resume_optimizer_updates=2,
               steps=1, step_ms=[100.], median_step_ms=100., images_per_second=20.,
               cuda_peak_allocated_bytes={'0': 100},
               cuda_peak_reserved_bytes={'0': 200},
               losses=[{'src_loss': 1., 'dst_loss': 1.}])
    weights = {'weight': torch.ones(1)}
    prediction = (np.zeros((1,), dtype=np.float32),)
    if fault == 'nonfinite_step_time':
        run['step_ms'][0] = math.nan
    elif fault == 'zero_primary_memory':
        run['cuda_peak_allocated_bytes']['0'] = 0
    elif fault == 'nonfinite_measured_loss':
        run['losses'][0]['src_loss'] = math.nan
    elif fault == 'incomplete_loss_history':
        run['losses'].clear()
    elif fault == 'nonfinite_prediction':
        prediction[0][0] = math.nan
    elif fault == 'nonfinite_weight':
        weights['weight'][0] = math.nan
    elif fault == 'missing_adversarial_loss':
        run['continued_step'].update(generator_loss=1., discriminator_loss=1.)
    elif fault == 'scaler_restore_mismatch':
        run['loaded_exact_scaler'] = False
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'test GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda index: type('GPU', (), {'total_memory': 1000})())
    monkeypatch.setattr(acceptance, '_fixed_batch',
                        lambda *args: (None, None, 'source', 'destination'))
    monkeypatch.setattr(acceptance, '_batch_digest', lambda *args: 'digest')
    monkeypatch.setattr(acceptance, '_train_run',
                        lambda *args, **kwargs: (run, weights, prediction))
    config = replace(_small(), gan_power=.1) if fault == 'missing_adversarial_loss' else _small()
    report = run_acceptance(config, tmp_path, tmp_path)
    assert report['status'] == 'failed'
    assert report['checks'][failed_check] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for hardware baseline')
@pytest.mark.parametrize('use_fp16', [False, True])
def test_one_gpu_baseline_save_reload_continue_and_missing_card(tmp_path, use_fp16):
    source = make_aligned(tmp_path / 'source', offset=3)
    destination = make_aligned(tmp_path / 'destination', offset=11)
    missing_index = torch.cuda.device_count()
    report = run_acceptance(replace(_small(), use_fp16=use_fp16), source, destination,
                            f'cuda:0,{missing_index}',
                            warmup=1, steps=2)
    assert report['status'] == 'partial'
    assert report['checks'] == {'single_gpu_resume': True, 'single_gpu_timing': True,
                                'single_gpu_memory_accounting': True,
                                'single_gpu_finite_training_outputs': True}
    single = report['single_gpu']
    assert single['iteration'] == 3 and single['resume_iteration'] == 4
    assert single['checkpoint_bytes'] > 0 and single['loaded_exact_weights']
    assert single['loaded_exact_optimizer'] and single['loaded_exact_adversarial_optimizer']
    assert single['loaded_exact_scaler']
    assert single['cuda_peak_allocated_bytes']['0'] > 0
    assert math.isfinite(single['images_per_second']) and single['images_per_second'] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two physical CUDA GPUs required')
def test_real_two_gpu_scatter_reduce_parity_memory_and_resume(tmp_path):
    source = make_aligned(tmp_path / 'source', offset=3)
    destination = make_aligned(tmp_path / 'destination', offset=11)
    report = run_acceptance(_small(), source, destination, 'cuda:0,1',
                            warmup=1, steps=2, mode='parity')
    assert report['status'] == 'passed', report['checks']
    assert all(report['checks'].values())
    assert {entry['device'] for entry in report['dual_gpu']['scatter_reduce_probe']['forwards']} == {0, 1}
    assert {entry['device'] for entry in report['dual_gpu']['scatter_reduce_probe']['backwards']} == {0, 1}
    assert report['dual_gpu']['cuda_peak_allocated_bytes']['1'] > 0
    assert math.isfinite(report['speedup']) and report['speedup'] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two physical CUDA GPUs required')
def test_real_two_gpu_rg_fp16_offload_gan_trueface_smoke(tmp_path):
    source = make_aligned(tmp_path / 'source', offset=3)
    destination = make_aligned(tmp_path / 'destination', offset=11)
    config = replace(_small(), archi='df-ud', use_rg=True, use_fp16=True,
                     optimizer_on_cpu=True, gan_power=.1, gan_dims=4,
                     gan_patch_size=8, true_face_power=.1)
    report = run_acceptance(config, source, destination, 'cuda:0,1',
                            warmup=1, steps=2, mode='smoke')
    assert report['status'] == 'passed', report['checks']
    assert all(report['checks'].values())
    assert math.isfinite(report['dual_gpu']['continued_step']['discriminator_loss'])
