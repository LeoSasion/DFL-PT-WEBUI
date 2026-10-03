"""Opt-in synchronized timing and memory samples for one ME training step.

CUDA synchronization changes throughput. Keep this collector out of normal
training and compare only runs measured with the same collector and batch.
"""
from contextlib import contextmanager
import time

import torch


def optimizer_state_bytes(optimizers):
    """Logical tensor bytes by placement, including discriminator moments."""
    totals = {'cpu': 0, 'cuda': 0}
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    kind = 'cuda' if value.device.type == 'cuda' else 'cpu'
                    totals[kind] += value.numel() * value.element_size()
    return totals


class StepProfiler:
    """Stage boundaries use CUDA synchronization for actual elapsed GPU work."""

    def __init__(self):
        self.device = None
        self._start_ns = None
        self._baseline_allocated = None
        self._stages = []
        self.result = None

    def _cuda_memory(self):
        if self.device.type != 'cuda':
            return None
        torch.cuda.synchronize(self.device)
        return dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device))

    def start(self, device):
        if self._start_ns is not None:
            raise RuntimeError('ME profiler is already measuring a step')
        self.device = torch.device(device)
        self._stages = []
        self.result = None
        memory = self._cuda_memory()
        self._baseline_allocated = memory['allocated_bytes'] if memory else None
        self._start_ns = time.perf_counter_ns()

    @contextmanager
    def stage(self, name):
        if self._start_ns is None:
            raise RuntimeError('Start the ME profiler before measuring stages')
        before = self._cuda_memory()
        if before is not None:
            torch.cuda.reset_peak_memory_stats(self.device)
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            after = self._cuda_memory()
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            stage = dict(name=name, elapsed_ms=elapsed_ms)
            if before is not None:
                stage.update(cuda_allocated_before_bytes=before['allocated_bytes'],
                             cuda_allocated_after_bytes=after['allocated_bytes'],
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            self._stages.append(stage)

    def finish(self, optimizers=()):
        if self._start_ns is None:
            raise RuntimeError('Start the ME profiler before finishing')
        after = self._cuda_memory()
        total_ms = (time.perf_counter_ns() - self._start_ns) / 1e6
        peaks = [entry['cuda_peak_allocated_bytes'] for entry in self._stages if 'cuda_peak_allocated_bytes' in entry]
        reserved = [entry['cuda_peak_reserved_bytes'] for entry in self._stages if 'cuda_peak_reserved_bytes' in entry]
        self.result = dict(total_ms=total_ms, stages=self._stages,
                           optimizer_state_bytes=optimizer_state_bytes(optimizers),
                           cuda_baseline_allocated_bytes=self._baseline_allocated,
                           cuda_end_allocated_bytes=after['allocated_bytes'] if after else None,
                           cuda_peak_allocated_bytes=max(peaks) if peaks else None,
                           cuda_peak_reserved_bytes=max(reserved) if reserved else None)
        self._start_ns = None
        return self.result
