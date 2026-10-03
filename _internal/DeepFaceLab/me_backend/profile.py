"""Reproducible fixed-batch ME RG / AMP / optimizer-placement benchmark.

Run ``python -m me_backend.profile --help`` from ``_internal/DeepFaceLab``.
The same decoded SRC/DST batch and model seed are reused for every variant.
Training data loading is deliberately outside the timed region.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import torch

from .config import MEConfig
from .data import AlignedDataset, PairDataLoader
from .engine import MEEngine
from .profiling import StepProfiler


def _batch_digest(source, destination):
    digest = hashlib.sha256()
    for side in (source, destination):
        for array in side:
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.dtype.str.encode('ascii'))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _fixed_batch(source_path, destination_path, config, seed):
    # Workers and prefetch would add time outside the step while making the
    # benchmark harder to reproduce. The augmentation policy still applies.
    data_config = replace(config.effective(), data_workers=0)
    with AlignedDataset(source_path, data_config, True, seed) as source_data, \
         AlignedDataset(destination_path, data_config, False, seed + 1) as destination_data:
        with PairDataLoader(source_data, destination_data) as pair:
            source, destination = pair.batch()
            return source, destination, source_data.fingerprint, destination_data.fingerprint


def _variants(device):
    fp16_values = (False, True) if torch.device(device).type == 'cuda' else (False,)
    offload_values = (False, True) if torch.device(device).type == 'cuda' else (False,)
    return itertools.product((False, True), fp16_values, offload_values)


def _summarize(samples):
    stages = {}
    for sample in samples:
        by_name = {}
        for stage in sample['stages']:
            by_name.setdefault(stage['name'], []).append(stage)
        for name, entries in by_name.items():
            total_ms = sum(entry['elapsed_ms'] for entry in entries)
            stage_peak = max((entry.get('cuda_peak_allocated_bytes') or 0) for entry in entries)
            stages.setdefault(name, []).append((total_ms, stage_peak))
    measured_steps = len(samples)
    summary = {
        'median_total_ms': statistics.median(sample['total_ms'] for sample in samples),
        'stages': {name: {'median_ms': statistics.median(value[0] for value in values),
                          'executed_steps': len(values),
                          'mean_ms_per_step': sum(value[0] for value in values) / measured_steps,
                          'median_cuda_peak_allocated_bytes': statistics.median(value[1] for value in values)
                          if any(value[1] for value in values) else None}
                   for name, values in stages.items()},
        'optimizer_state_bytes': samples[-1]['optimizer_state_bytes'],
        'median_cuda_peak_allocated_bytes': statistics.median(
            sample['cuda_peak_allocated_bytes'] for sample in samples)
            if samples[0]['cuda_peak_allocated_bytes'] is not None else None,
        'max_cuda_peak_allocated_bytes': max(sample['cuda_peak_allocated_bytes'] for sample in samples)
            if samples[0]['cuda_peak_allocated_bytes'] is not None else None,
    }
    return summary


def _comparison(runs):
    baseline = next(run for run in runs if not any(run['options'].values()))
    comparisons = []
    for run in runs:
        item = dict(variant=run['variant'])
        for field in ('median_total_ms', 'median_cuda_peak_allocated_bytes',
                      'max_cuda_peak_allocated_bytes'):
            current, original = run['summary'][field], baseline['summary'][field]
            item[field + '_change_percent'] = 100 * (current / original - 1) if current is not None and original else None
        comparisons.append(item)
    return comparisons


def benchmark(config, source_path, destination_path, device='cuda:0', seed=7, warmup=2, steps=6):
    """Return a JSON-safe 2×2×2 comparison on identical aligned tensors."""
    if type(warmup) is not int or warmup < 1 or type(steps) is not int or steps < 1:
        raise ValueError('warmup and steps must be positive integers')
    if not isinstance(config, MEConfig):
        raise TypeError('config must be MEConfig')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA benchmark requested but unavailable')
    source, destination, source_fingerprint, destination_fingerprint = _fixed_batch(
        source_path, destination_path, config, seed)
    digest = _batch_digest(source, destination)
    device_name = torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else 'CPU'
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  device=str(device), device_name=device_name, torch_version=str(torch.__version__),
                  cuda_runtime=torch.version.cuda, seed=seed, warmup=warmup, measured_steps=steps,
                  source=str(Path(source_path).resolve()), destination=str(Path(destination_path).resolve()),
                  source_fingerprint=source_fingerprint, destination_fingerprint=destination_fingerprint,
                  batch_sha256=digest, config=config.to_dict(),
                  method='Fixed decoded batch; fresh seeded model per variant; synchronized stage boundaries; median of measured steps. Data loading excluded.',
                  runs=[])
    for use_rg, use_fp16, optimizer_on_cpu in _variants(device):
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        variant_config = replace(config, use_rg=use_rg, use_fp16=use_fp16,
                                 optimizer_on_cpu=optimizer_on_cpu)
        engine = MEEngine(variant_config, device, seed)
        options = dict(use_rg=use_rg, use_fp16=use_fp16, optimizer_on_cpu=optimizer_on_cpu)
        variant_name = f"rg{int(use_rg)}-fp16{int(use_fp16)}-cpuopt{int(optimizer_on_cpu)}"
        try:
            for _ in range(warmup):
                engine.train_step(source, destination)
            samples = []
            losses = []
            for _ in range(steps):
                profiler = StepProfiler()
                result = engine.train_step(source, destination, profiler=profiler)
                samples.append(profiler.result)
                losses.append(dict(src_loss=result['src_loss'], dst_loss=result['dst_loss']))
            report['runs'].append(dict(variant=variant_name, options=options,
                                       summary=_summarize(samples), samples=samples, losses=losses))
        finally:
            del engine
            gc.collect()
    report['comparisons'] = _comparison(report['runs'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, help='Aligned SRC directory, PAK, or ZIP')
    parser.add_argument('--dst', type=Path, help='Aligned DST directory, PAK, or ZIP')
    parser.add_argument('--config', type=Path, help='ME JSON config; defaults to MEConfig defaults')
    parser.add_argument('--fixture-manifest', type=Path, help='QA fixture JSON with src, dst and legacy_config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True, help='Destination JSON report')
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    manifest = json.loads(args.fixture_manifest.read_text(encoding='utf-8-sig')) if args.fixture_manifest else {}
    source, destination = args.src or manifest.get('src'), args.dst or manifest.get('dst')
    if not source or not destination:
        parser.error('Specify --src and --dst, or --fixture-manifest')
    options = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else manifest.get('legacy_config', {})
    config = MEConfig.from_dict(options)
    report = benchmark(config, source, destination, args.device, args.seed, args.warmup, args.steps)
    report['cpu_threads'] = args.cpu_threads
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for run in report['runs']:
        summary = run['summary']
        print(f"{run['variant']}: {summary['median_total_ms']:.2f} ms, "
              f"median/max step CUDA allocated peak "
              f"{summary['median_cuda_peak_allocated_bytes']}/"
              f"{summary['max_cuda_peak_allocated_bytes']} bytes, "
              f"optimizer state {summary['optimizer_state_bytes']}")
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
