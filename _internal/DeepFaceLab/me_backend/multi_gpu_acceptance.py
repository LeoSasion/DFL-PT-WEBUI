"""ME one/two GPU training, scatter/reduce, memory and resume acceptance.

The one-GPU baseline still runs on a one-card host; the two-card gate is then
reported as skipped. On a two-card host, --require-dual makes it a CI gate.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
from threading import Lock
import time
import traceback

import numpy as np
import torch

from .config import MEConfig
from .engine import MEEngine
from .profile import _batch_digest, _fixed_batch


LOSS_FIELDS = ('src_loss', 'dst_loss', 'generator_loss', 'discriminator_loss')


def _finite_step_losses(step, require_adversarial=False):
    """Require SRC/DST losses and, for adversarial runs, both extra losses."""
    required = LOSS_FIELDS if require_adversarial else LOSS_FIELDS[:2]
    if not isinstance(step, dict) or any(key not in step for key in required):
        return False
    try:
        return all(math.isfinite(step[key]) for key in LOSS_FIELDS if key in step)
    except (TypeError, ValueError):
        return False


def _selection(value):
    if not re.fullmatch(r'cuda:\d+,\d+', value):
        raise ValueError('Use two GPU indexes, for example cuda:0,1')
    first, second = (int(part) for part in value[5:].split(','))
    if first == second:
        raise ValueError('Select two distinct GPUs')
    return first, second


def _sync(indexes):
    for index in indexes:
        torch.cuda.synchronize(index)


def _snapshot(engine):
    return {name: tensor.detach().cpu().clone() for name, tensor in engine.network.state_dict().items()}


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_digest(optimizer):
    if optimizer is None:
        return None
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(state['param_groups'], sort_keys=True).encode('utf-8'))
    for identifier in sorted(state['state']):
        digest.update(str(identifier).encode('ascii'))
        for name, value in sorted(state['state'][identifier].items()):
            digest.update(name.encode('ascii'))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode('ascii'))
                digest.update(memoryview(tensor.numpy()).cast('B'))
            else:
                digest.update(repr(value).encode('ascii'))
    return digest.hexdigest()


def _weights_error(reference, actual):
    if reference.keys() != actual.keys():
        raise ValueError('Single/dual model state keys differ')
    squared_difference = squared_reference = max_abs = 0.0
    for name in reference:
        before, after = reference[name].double(), actual[name].double()
        if before.shape != after.shape:
            raise ValueError(f'Single/dual shape differs for {name}')
        difference = before - after
        squared_difference += float((difference * difference).sum())
        squared_reference += float((before * before).sum())
        max_abs = max(max_abs, float(difference.abs().max()))
    return dict(relative_l2=math.sqrt(squared_difference / max(squared_reference, 1e-30)),
                max_abs=max_abs)


def _scatter_reduce_probe(engine, source, destination, indexes):
    """Observe per-replica inputs and backpropagated output gradients."""
    forwards, backwards = [], []
    lock = Lock()

    def on_forward(_module, args, outputs):
        device = args[0].device.index
        with lock:
            forwards.append(dict(device=device, samples=int(args[0].shape[0])))

        def on_gradient(gradient):
            magnitude = float(gradient.detach().abs().sum())
            with lock:
                backwards.append(dict(device=gradient.device.index, abs_sum=magnitude))
            return gradient

        outputs['src'].register_hook(on_gradient)

    hook = engine.forward_network.module.register_forward_hook(on_forward)
    try:
        result = engine.train_step(source, destination)
        _sync(indexes)
    finally:
        hook.remove()
    primary_gradients = [parameter.grad for parameter in engine.network.parameters() if parameter.grad is not None]
    primary_gradient_sum = sum(float(gradient.detach().abs().sum()) for gradient in primary_gradients)
    return dict(forwards=sorted(forwards, key=lambda item: item['device']),
                backwards=sorted(backwards, key=lambda item: item['device']),
                primary_gradient_abs_sum=primary_gradient_sum,
                step=result)


def _train_run(config, source, destination, indexes, seed, warmup, steps, probe=False):
    device = 'cuda:' + ','.join(str(index) for index in indexes)
    torch.cuda.empty_cache()
    engine = MEEngine(config, device, seed)
    observed = None
    for step in range(warmup):
        if probe and step == 0:
            observed = _scatter_reduce_probe(engine, source, destination, indexes)
        else:
            engine.train_step(source, destination)
    elapsed_ms, losses = [], []
    peaks = {str(index): [] for index in indexes}
    reserved = {str(index): [] for index in indexes}
    for _ in range(steps):
        _sync(indexes)
        for index in indexes:
            torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter_ns()
        result = engine.train_step(source, destination)
        _sync(indexes)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
        losses.append({key: result[key] for key in LOSS_FIELDS if key in result})
        for index in indexes:
            peaks[str(index)].append(torch.cuda.max_memory_allocated(index))
            reserved[str(index)].append(torch.cuda.max_memory_reserved(index))
    iteration = engine.iteration
    optimizer_updates = engine.optimizer_updates
    weights = _snapshot(engine)
    optimizer_digest = _optimizer_digest(engine.optimizer)
    adversarial_optimizer_digest = _optimizer_digest(engine.adversarial_optimizer)
    scaler_state = copy.deepcopy(engine.scaler.state_dict())
    prediction = engine.predict(destination[1])
    with tempfile.TemporaryDirectory(prefix='me-multi-gpu-acceptance-') as temporary:
        checkpoint = Path(temporary) / 'me.pt'
        engine.save(checkpoint)
        checkpoint_hash = _file_digest(checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        del engine
        gc.collect()
        loaded = MEEngine.load(checkpoint, device)
        loaded_exact = all(torch.equal(value, loaded.network.state_dict()[name].detach().cpu())
                           for name, value in weights.items())
        loaded_exact_optimizer = optimizer_digest == _optimizer_digest(loaded.optimizer)
        loaded_exact_adversarial_optimizer = adversarial_optimizer_digest == _optimizer_digest(loaded.adversarial_optimizer)
        loaded_exact_scaler = scaler_state == loaded.scaler.state_dict()
        loaded_iteration = loaded.iteration
        loaded_updates = loaded.optimizer_updates
        continued = loaded.train_step(source, destination)
        continued_finite = _finite_step_losses(
            continued, bool(config.gan_power or config.true_face_power))
        resume_iteration = loaded.iteration
        resume_updates = loaded.optimizer_updates
        del loaded
        gc.collect()
    median_ms = statistics.median(elapsed_ms)
    report = dict(device=device, warmup=warmup, steps=steps, iteration=iteration,
                  optimizer_updates=optimizer_updates, median_step_ms=median_ms,
                  images_per_second=config.batch_size * 1000 / median_ms,
                  step_ms=elapsed_ms, losses=losses,
                  cuda_peak_allocated_bytes={index: max(values) for index, values in peaks.items()},
                  cuda_peak_reserved_bytes={index: max(values) for index, values in reserved.items()},
                  checkpoint_sha256=checkpoint_hash, checkpoint_bytes=checkpoint_bytes,
                  loaded_exact_weights=loaded_exact,
                  loaded_exact_optimizer=loaded_exact_optimizer,
                  loaded_exact_adversarial_optimizer=loaded_exact_adversarial_optimizer,
                  loaded_exact_scaler=loaded_exact_scaler,
                  loaded_iteration=loaded_iteration,
                  loaded_optimizer_updates=loaded_updates, continued_step=continued,
                  continued_finite=continued_finite, resume_iteration=resume_iteration,
                  resume_optimizer_updates=resume_updates)
    if observed is not None:
        report['scatter_reduce_probe'] = observed
    return report, weights, prediction


def _resume_ok(run, require_adversarial=False):
    return (run['checkpoint_bytes'] > 0 and run['loaded_exact_weights']
            and run['loaded_exact_optimizer'] and run['loaded_exact_adversarial_optimizer']
            and run['loaded_exact_scaler']
            and run['loaded_iteration'] == run['iteration']
            and run['loaded_optimizer_updates'] == run['optimizer_updates']
            and run['continued_finite']
            and _finite_step_losses(run['continued_step'], require_adversarial)
            and run['resume_iteration'] == run['iteration'] + 1
            and run['resume_optimizer_updates'] == run['optimizer_updates'] + 1
            + int(run['continued_step'].get('replayed', False)))


def _finite_run_outputs(run, weights, prediction, require_adversarial=False):
    """Require every measured step and exported result to be present and finite."""
    return (len(run['losses']) == run['steps'] > 0
            and all(_finite_step_losses(loss, require_adversarial) for loss in run['losses'])
            and bool(weights)
            and all(bool(torch.isfinite(tensor).all()) for tensor in weights.values())
            and bool(prediction)
            and all(array.size > 0 and bool(np.isfinite(array).all()) for array in prediction))


def _timing_ok(run, batch_size):
    """Reject incomplete or corrupt timing samples, not only their summary."""
    samples = run['step_ms']
    median_ms = run['median_step_ms']
    throughput = run['images_per_second']
    return (len(samples) == run['steps'] > 0
            and all(math.isfinite(value) and value > 0 for value in samples)
            and math.isfinite(median_ms) and median_ms > 0
            and math.isclose(median_ms, statistics.median(samples), rel_tol=1e-9)
            and math.isfinite(throughput) and throughput > 0
            and math.isclose(throughput, batch_size * 1000 / median_ms, rel_tol=1e-9))


def _memory_ok(run, indexes):
    """Both devices must have coherent allocated and reserved CUDA peaks."""
    allocated = run['cuda_peak_allocated_bytes']
    reserved = run['cuda_peak_reserved_bytes']
    return all((isinstance(allocated[str(index)], int)
                and isinstance(reserved[str(index)], int)
                and 0 < allocated[str(index)] <= reserved[str(index)])
               for index in indexes)


def _assess_dual_run(config, single, dual, indexes, single_weights, dual_weights,
                     single_prediction, dual_prediction, mode):
    """Apply the hardware gate to observed runs; also testable without two GPUs."""
    first, second = indexes
    require_adversarial = bool(config.gan_power or config.true_face_power)
    observed = dual['scatter_reduce_probe']
    requested = {first, second}
    forward_devices = {entry['device'] for entry in observed['forwards']}
    backward_devices = {entry['device'] for entry in observed['backwards'] if entry['abs_sum'] > 0}
    checks = {
        'dual_gpu_resume': _resume_ok(dual, require_adversarial),
        'scatter_both_gpus': (forward_devices == requested
            and all(entry['samples'] > 0 for entry in observed['forwards'])
            and sum(entry['samples'] for entry in observed['forwards']) == config.batch_size),
        'backward_both_gpus': (backward_devices == requested
            and all(math.isfinite(entry['abs_sum']) and entry['abs_sum'] >= 0
                    for entry in observed['backwards'])
            and math.isfinite(observed['primary_gradient_abs_sum'])
            and observed['primary_gradient_abs_sum'] > 0),
        'secondary_cuda_allocations': dual['cuda_peak_allocated_bytes'][str(second)] > 0,
        'gpu_memory_accounting': (_memory_ok(single, (first,))
                                  and _memory_ok(dual, indexes)),
        'finite_throughput': all(_timing_ok(run, config.batch_size) for run in (single, dual)),
    }
    losses = {key: abs(single['losses'][-1][key] - dual['losses'][-1][key])
              for key in ('src_loss', 'dst_loss')}
    weights = _weights_error(single_weights, dual_weights)
    prediction_error = max(float(np.max(np.abs(one - two)))
                           for one, two in zip(single_prediction, dual_prediction))
    checks['finite_training_outputs'] = (
        _finite_run_outputs(single, single_weights, single_prediction, require_adversarial)
        and _finite_run_outputs(dual, dual_weights, dual_prediction, require_adversarial)
        and _finite_step_losses(observed.get('step'), require_adversarial)
        and all(_finite_step_losses(run['continued_step'], require_adversarial)
                for run in (single, dual))
        and all(math.isfinite(value) for value in losses.values())
        and all(math.isfinite(value) for value in weights.values())
        and math.isfinite(prediction_error))
    if mode == 'parity':
        # Adam-style updates may amplify tiny cuDNN reduction differences, so
        # this is a practical parity gate rather than bitwise identity.
        checks['single_dual_numerical_parity'] = (
            checks['finite_training_outputs'] and max(losses.values()) <= 5e-3
            and weights['relative_l2'] <= 1e-3 and weights['max_abs'] <= 1e-3
            and prediction_error <= 5e-3)
    comparison = dict(loss_abs=losses, weights=weights,
                      prediction_max_abs=prediction_error)
    speedup = None
    if checks['finite_throughput']:
        measured_speedup = dual['images_per_second'] / single['images_per_second']
        if math.isfinite(measured_speedup) and measured_speedup > 0:
            speedup = measured_speedup
        else:
            checks['finite_throughput'] = False
    return checks, comparison, speedup


def _json_safe(value):
    """Represent failed nonfinite measurements as null in the durable report."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_acceptance(config, source_path, destination_path, devices='cuda:0,1',
                   seed=7, warmup=2, steps=5, mode='parity'):
    """Run a single-card baseline and, when present, a real DataParallel gate."""
    first, second = _selection(devices)
    if type(warmup) is not int or warmup < 1 or type(steps) is not int or steps < 1:
        raise ValueError('warmup and steps must be positive integers')
    if mode not in ('parity', 'smoke'):
        raise ValueError('mode must be parity or smoke')
    if config.batch_size < 2:
        raise ValueError('Dual GPU acceptance requires batch_size at least 2')
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  requested_devices=devices, available_gpu_count=available,
                  gpu_names=[torch.cuda.get_device_name(index) for index in range(available)],
                  gpu_total_memory_bytes=[torch.cuda.get_device_properties(index).total_memory
                                          for index in range(available)],
                  mode=mode, config=config.to_dict(), seed=seed, warmup=warmup, steps=steps,
                  checks={}, status='skipped')
    if first >= available:
        report['reason'] = f'Primary cuda:{first} is unavailable; cannot run the one-card baseline'
        return report
    source, destination, source_fingerprint, destination_fingerprint = _fixed_batch(
        source_path, destination_path, config, seed)
    report.update(source=str(Path(source_path).resolve()), destination=str(Path(destination_path).resolve()),
                  source_fingerprint=source_fingerprint,
                  destination_fingerprint=destination_fingerprint,
                  batch_sha256=_batch_digest(source, destination))
    single, single_weights, single_prediction = _train_run(
        config, source, destination, (first,), seed, warmup, steps)
    report['single_gpu'] = single
    require_adversarial = bool(config.gan_power or config.true_face_power)
    report['checks']['single_gpu_resume'] = _resume_ok(single, require_adversarial)
    report['checks']['single_gpu_timing'] = _timing_ok(single, config.batch_size)
    report['checks']['single_gpu_memory_accounting'] = _memory_ok(single, (first,))
    report['checks']['single_gpu_finite_training_outputs'] = _finite_run_outputs(
        single, single_weights, single_prediction, require_adversarial)
    if second >= available:
        report['status'] = 'partial' if all(report['checks'].values()) else 'failed'
        report['reason'] = f'Secondary cuda:{second} unavailable; real dual-GPU check skipped'
        return report
    dual, dual_weights, dual_prediction = _train_run(
        config, source, destination, (first, second), seed, warmup, steps, probe=True)
    report['dual_gpu'] = dual
    checks, comparison, speedup = _assess_dual_run(config, single, dual, (first, second),
        single_weights, dual_weights, single_prediction, dual_prediction, mode)
    report['checks'].update(checks)
    report['speedup'] = speedup
    report['numerical_comparison'] = comparison
    report['status'] = 'passed' if all(report['checks'].values()) else 'failed'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path)
    parser.add_argument('--dst', type=Path)
    parser.add_argument('--config', type=Path, help='ME JSON config; defaults to MEConfig defaults')
    parser.add_argument('--fixture-manifest', type=Path)
    parser.add_argument('--devices', default='cuda:0,1')
    parser.add_argument('--mode', choices=('parity', 'smoke'), default='parity')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-dual', action='store_true', help='Exit nonzero when the second GPU is absent')
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
    try:
        report = run_acceptance(config, source, destination, args.devices,
                                args.seed, args.warmup, args.steps, args.mode)
    except Exception as error:
        report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                      requested_devices=args.devices, mode=args.mode, config=config.to_dict(),
                      status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    report['cpu_threads'] = args.cpu_threads
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(report), ensure_ascii=False,
                                      indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"ME GPU acceptance: {report['status']}; report: {args.output.resolve()}")
    if report.get('reason'):
        print(report['reason'])
    if report['status'] == 'failed' or (args.require_dual and report['status'] != 'passed'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
