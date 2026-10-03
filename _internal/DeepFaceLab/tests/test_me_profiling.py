"""The diagnostic profiler must leave normal ME updates intact."""
from dataclasses import replace
import math

import pytest
import torch

from me_backend.config import MEConfig
from me_backend.data import AlignedDataset
from me_backend.engine import MEEngine
from me_backend.profile import _comparison, _summarize, benchmark
from me_backend.profiling import StepProfiler, optimizer_state_bytes
from tests.me_fixtures import make_aligned


torch.set_num_threads(2)


def _small(**options):
    return MEConfig(resolution=64, ae_dims=32, e_dims=16, d_dims=16, d_mask_dims=16,
                    batch_size=1, **options)


def _batch(tmp_path, config):
    with AlignedDataset(make_aligned(tmp_path / 'src'), config, True, 9) as data:
        return data.batch()


def test_profiled_cpu_step_has_stages_and_identical_update(tmp_path):
    config = _small()
    sample = _batch(tmp_path, config)
    normal = MEEngine(config, 'cpu', 13)
    profiled = MEEngine(config, 'cpu', 13)
    expected = normal.train_step(sample, sample)
    recorder = StepProfiler()
    actual = profiled.train_step(sample, sample, profiler=recorder)
    assert actual == expected
    assert math.isfinite(recorder.result['total_ms']) and recorder.result['total_ms'] > 0
    assert recorder.result['cuda_peak_allocated_bytes'] is None
    assert {'transfer', 'forward', 'loss', 'generator_backward', 'gradient_check', 'optimizer'} <= {
        stage['name'] for stage in recorder.result['stages']}
    assert recorder.result['optimizer_state_bytes']['cpu'] > 0
    for name, expected_weight in normal.network.state_dict().items():
        torch.testing.assert_close(profiled.network.state_dict()[name], expected_weight, rtol=0, atol=0)


def test_profiler_covers_replayed_updates(tmp_path):
    config = _small(retraining_samples=True, retraining_every=2, retraining_capacity=2)
    sample = _batch(tmp_path, config)
    engine = MEEngine(config, 'cpu', 2)
    engine.train_step(sample, sample)
    recorder = StepProfiler()
    result = engine.train_step(sample, sample, profiler=recorder)
    assert result['replayed'] and engine.optimizer_updates == 3
    names = {stage['name'] for stage in recorder.result['stages']}
    assert 'replay_forward' in names and 'replay_optimizer' in names


def test_stage_summary_accounts_for_conditional_replay_steps():
    samples = [
        dict(total_ms=10., stages=[dict(name='forward', elapsed_ms=8., cuda_peak_allocated_bytes=100)],
             optimizer_state_bytes={'cpu': 1, 'cuda': 0}, cuda_peak_allocated_bytes=100),
        dict(total_ms=18., stages=[dict(name='forward', elapsed_ms=8., cuda_peak_allocated_bytes=100),
                                   dict(name='replay_forward', elapsed_ms=8., cuda_peak_allocated_bytes=1000)],
             optimizer_state_bytes={'cpu': 1, 'cuda': 0}, cuda_peak_allocated_bytes=1000),
        dict(total_ms=10., stages=[dict(name='forward', elapsed_ms=8., cuda_peak_allocated_bytes=100)],
             optimizer_state_bytes={'cpu': 1, 'cuda': 0}, cuda_peak_allocated_bytes=100),
    ]
    summary = _summarize(samples)
    assert summary['stages']['forward']['executed_steps'] == 3
    assert summary['stages']['forward']['mean_ms_per_step'] == 8.
    assert summary['stages']['replay_forward']['executed_steps'] == 1
    assert summary['stages']['replay_forward']['median_ms'] == 8.
    assert summary['stages']['replay_forward']['mean_ms_per_step'] == pytest.approx(8. / 3.)
    assert summary['median_cuda_peak_allocated_bytes'] == 100
    assert summary['max_cuda_peak_allocated_bytes'] == 1000
    baseline = dict(summary, max_cuda_peak_allocated_bytes=100)
    comparisons = _comparison([
        dict(variant='baseline', options=dict(use_rg=False), summary=baseline),
        dict(variant='conditional-replay', options=dict(use_rg=True), summary=summary),
    ])
    assert comparisons[1]['median_cuda_peak_allocated_bytes_change_percent'] == 0
    assert comparisons[1]['max_cuda_peak_allocated_bytes_change_percent'] == 900


def test_fixed_batch_benchmark_is_reproducible_and_cpu_safe(tmp_path):
    source = make_aligned(tmp_path / 'source', offset=3)
    destination = make_aligned(tmp_path / 'destination', offset=11)
    config = _small()
    report = benchmark(config, source, destination, 'cpu', seed=5, warmup=1, steps=1)
    assert report['batch_sha256'] and len(report['batch_sha256']) == 64
    assert [run['variant'] for run in report['runs']] == ['rg0-fp160-cpuopt0', 'rg1-fp160-cpuopt0']
    assert all(math.isfinite(run['losses'][0]['src_loss']) for run in report['runs'])
    assert all(run['summary']['median_cuda_peak_allocated_bytes'] is None for run in report['runs'])
    assert all(run['summary']['optimizer_state_bytes']['cpu'] > 0 for run in report['runs'])
    assert report['comparisons'][0]['median_total_ms_change_percent'] == 0
    with pytest.raises(ValueError, match='positive'):
        benchmark(replace(config, use_fp16=True), source, destination, 'cpu', warmup=0)


def test_optimizer_state_byte_placement_ignores_missing_optimizer():
    config = _small()
    engine = MEEngine(config, 'cpu')
    assert optimizer_state_bytes((engine.optimizer, None)) == {'cpu': 0, 'cuda': 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for measured VRAM')
def test_cuda_profiler_measures_real_fp16_rg_offload_step(tmp_path):
    config = _small(use_rg=True, use_fp16=True, optimizer_on_cpu=True)
    sample = _batch(tmp_path, config)
    engine = MEEngine(config, 'cuda:0', 7)
    recorder = StepProfiler()
    result = engine.train_step(sample, sample, profiler=recorder)
    assert math.isfinite(result['src_loss'])
    assert recorder.result['cuda_peak_allocated_bytes'] > recorder.result['cuda_baseline_allocated_bytes'] > 0
    assert recorder.result['cuda_peak_reserved_bytes'] >= recorder.result['cuda_peak_allocated_bytes']
    assert recorder.result['optimizer_state_bytes']['cpu'] > 0
    assert recorder.result['optimizer_state_bytes']['cuda'] == 0
