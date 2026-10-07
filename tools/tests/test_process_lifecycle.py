"""Real spawn-process deadlines, cleanup and RPC ownership recovery."""
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '_internal/DeepFaceLab'))
from core.joblib import Subprocessor, MPFunc, MPClassFuncOnDemand
from core.joblib.MPRemote import RPCOwnerStopped, RPCRemoteError
from core.joblib.SubprocessGenerator import SubprocessGenerator
from core.process_identity import process_identity
from core.interact import interact as io


class TestWorker(Subprocessor.Cli):
    __test__ = False
    def on_initialize(self, options):
        self.kind = options['kind']
        if self.kind == 'init_crash': raise ValueError('init fixture crash')
        if self.kind == 'init_exit': os._exit(17)
        if self.kind == 'init_hang': time.sleep(60)

    def process_data(self, data):
        if self.kind == 'process_exit': os._exit(19)
        if self.kind == 'process_hang': time.sleep(60)
        if self.kind == 'process_crash': raise ValueError('process fixture crash')
        return data * 2

    def on_finalize(self):
        if self.kind == 'finalize_hang': time.sleep(60)
        if self.kind == 'finalize_crash': raise ValueError('finalize fixture crash')


class TestPool(Subprocessor):
    __test__ = False
    def __init__(self, kind='ok', serial=False, host_error=False):
        self.pending = deque([3, 4]); self.results = []; self.returned = []; self.host_error = host_error
        self.kind = kind; self.finalized = 0
        super().__init__('fixture-pool', TestWorker, no_response_time_sec=.4,
                         initialize_subprocesses_in_serial=serial,
                         initialize_timeout_sec=.5 if kind.startswith('init_') else 10,
                         finalize_timeout_sec=.4, shutdown_timeout_sec=.5)
    def process_info_generator(self): yield 'worker-0', {}, {'kind': self.kind}
    def get_data(self, host): return self.pending.popleft() if self.pending else None
    def on_data_return(self, host, data): self.returned.append(data); self.pending.appendleft(data)
    def on_result(self, host, data, result):
        if self.host_error: raise ValueError('host fixture failure')
        self.results.append(result)
    def on_tick(self): return not self.pending
    def on_clients_finalized(self): self.finalized += 1
    def get_result(self): return self.results


@pytest.mark.parametrize('kind', ['init_crash', 'init_exit', 'init_hang'])
@pytest.mark.parametrize('serial', [False, True])
def test_initialization_has_deadline_including_serial_worker_failure(kind, serial):
    pool = TestPool(kind, serial=serial)
    started = time.monotonic()
    with pytest.raises(Subprocessor.LifecycleError): pool.run()
    assert time.monotonic() - started < 6
    assert pool.lifecycle_errors and pool.lifecycle_errors[0]['phase'] == 'initialize'


@pytest.mark.parametrize('kind', ['process_exit', 'process_hang', 'process_crash'])
def test_lost_or_hung_worker_returns_data_once_and_never_claims_partial_success(kind):
    pool = TestPool(kind)
    with pytest.raises(Subprocessor.LifecycleError): pool.run()
    assert pool.returned == [3] and pool.finalized == 1
    assert pool.lifecycle_errors[0]['phase'] == 'process'


def test_normal_spawn_results_and_host_callback_exception_clean_all_queues():
    pool = TestPool()
    assert pool.run() == [6, 8] and pool.finalized == 1
    for worker in pool.clis:
        assert not worker.p.is_alive() and worker.exit_confirmed and worker._queues_closed
    broken = TestPool(host_error=True)
    with pytest.raises(ValueError, match='host fixture'): broken.run()
    assert all(not cli.p.is_alive() and cli._queues_closed for cli in broken.clis)
    assert broken.finalized == 1


def test_worker_finalization_hang_is_bounded_and_exception_is_reported():
    pool = TestPool('finalize_hang')
    with pytest.raises(Subprocessor.LifecycleError, match='deadline'): pool.run()
    assert pool.results == [6, 8] and pool.lifecycle_errors[-1]['phase'] == 'finalize'
    assert all(not cli.p.is_alive() for cli in pool.clis)
    broken = TestPool('finalize_crash')
    with pytest.raises(Subprocessor.LifecycleError, match='finalization'): broken.run()
    assert all(not cli.p.is_alive() for cli in broken.clis)


def plus_one(value): return value + 1
def fail_callback(value): raise ValueError(f'callback fixture {value}')
def delayed(value, delay=0): time.sleep(delay); return value


class LazyCalculator:
    def __init__(self, offset=0): self.offset = offset
    def calculate(self, value): return value + self.offset


def rpc_call(remote, output, *args, **kwargs):
    try: output.put(('ok', remote(*args, **kwargs)))
    except BaseException as error: output.put((type(error).__name__, str(error)))


def pump(processes, timeout=15):
    deadline = time.monotonic() + timeout
    while any(process.is_alive() for process in processes):
        if time.monotonic() > deadline: raise TimeoutError('test caller did not stop')
        io.process_messages(.005)
    for process in processes: process.join(timeout=1)


def cleanup_process(process):
    if process.is_alive(): process.terminate(); process.join(timeout=1)
    if process.is_alive(): process.kill(); process.join(timeout=1)


def test_windows_spawn_manager_transport_parallel_private_reply_and_owner_close():
    context = mp.get_context('spawn')
    output = context.Queue()
    remote = MPFunc(plus_one, rpc_timeout_sec=5)
    lazy = MPClassFuncOnDemand(LazyCalculator, 'calculate', offset=7, rpc_timeout_sec=5)
    manager_pid = remote.manager_pid
    assert manager_pid == lazy.manager_pid
    processes = [context.Process(target=rpc_call, args=(remote, output, n)) for n in range(4)]
    processes.append(context.Process(target=rpc_call, args=(lazy, output, 20)))
    try:
        for process in processes: process.start()
        pump(processes)
        values = [output.get(timeout=1) for _ in processes]
        assert all(item[0] == 'ok' for item in values)
        assert sorted(item[1] for item in values) == [1, 2, 3, 4, 27]
        assert remote(9) == 10
        remote.close()
        assert process_identity(manager_pid) is not None
        lazy.close()
        assert process_identity(manager_pid) is None
    finally:
        for process in processes: cleanup_process(process)
        remote.close(); lazy.close()
        output.cancel_join_thread(); output.close()


def test_callback_exception_returns_structured_failure_and_following_call_succeeds():
    context = mp.get_context('spawn'); output = context.Queue()
    remote = MPFunc(fail_callback, rpc_timeout_sec=5)
    first = context.Process(target=rpc_call, args=(remote, output, 3))
    second = None
    try:
        first.start(); pump([first])
        result = output.get(timeout=1)
        assert result[0] == 'RPCRemoteError' and 'callback fixture 3' in result[1]
        remote.func = plus_one
        second = context.Process(target=rpc_call, args=(remote, output, 7))
        second.start(); pump([second]); assert output.get(timeout=1) == ('ok', 8)
    finally:
        cleanup_process(first)
        if second is not None: cleanup_process(second)
        remote.close(); output.cancel_join_thread(); output.close()


def test_timeout_late_response_does_not_poison_next_call_or_hold_global_lock():
    context = mp.get_context('spawn'); output = context.Queue()
    remote = MPFunc(delayed, rpc_timeout_sec=.25)
    first = context.Process(target=rpc_call, args=(remote, output, 'old'), kwargs={'delay': .5})
    second = None
    try:
        first.start(); pump([first])
        assert output.get(timeout=1)[0] == 'TimeoutError'
        remote.rpc_timeout_sec = 5
        second = context.Process(target=rpc_call, args=(remote, output, 'new'))
        second.start(); pump([second]); assert output.get(timeout=1) == ('ok', 'new')
    finally:
        cleanup_process(first)
        if second is not None: cleanup_process(second)
        remote.close(); output.cancel_join_thread(); output.close()


def test_dead_owner_identity_and_closed_owner_fail_without_waiting_default_timeout():
    remote = MPFunc(plus_one)
    try:
        remote.owner_identity = 'not-the-creation-token'
        started = time.monotonic()
        with pytest.raises(RPCOwnerStopped): remote(1)
        assert time.monotonic() - started < 1
        remote.owner_identity = process_identity(os.getpid())
        remote.close()
        with pytest.raises(RPCOwnerStopped): remote(1)
    finally:
        remote.close()


def abruptly_exiting_owner(control, output):
    remote = MPFunc(plus_one, rpc_timeout_sec=300)
    context = mp.get_context('spawn')
    caller = context.Process(target=rpc_call, args=(remote, output, 7))
    caller.daemon = True
    caller.start()
    deadline = time.monotonic() + 10
    while remote.requests.qsize() == 0:
        if time.monotonic() > deadline: raise TimeoutError('Caller did not issue its request')
        time.sleep(.01)
    output.put(('ready', remote.manager_pid, process_identity(remote.manager_pid), caller.pid))
    control.recv()
    os._exit(41)


def test_actual_owner_death_unblocks_caller_and_manager_watchdog_exits():
    context = mp.get_context('spawn'); output = context.Queue()
    control, child_control = context.Pipe()
    owner = context.Process(target=abruptly_exiting_owner, args=(child_control, output))
    manager_pid = None
    try:
        owner.start()
        ready = output.get(timeout=15)
        assert ready[0] == 'ready'
        manager_pid, identity, caller_pid = ready[1:]
        control.send('exit')
        started = time.monotonic()
        owner.join(timeout=3); assert owner.exitcode == 41
        result = output.get(timeout=5)
        assert result[0] == 'RPCOwnerStopped'
        while process_identity(manager_pid) == identity or process_identity(caller_pid) is not None:
            if time.monotonic() - started > 5: raise TimeoutError('Owned RPC descendants did not stop')
            time.sleep(.01)
        assert time.monotonic() - started < 5
    finally:
        cleanup_process(owner)
        control.close(); child_control.close(); output.cancel_join_thread(); output.close()


def test_client_exit_with_pending_request_does_not_prevent_healthy_client():
    context = mp.get_context('spawn'); output = context.Queue()
    remote = MPFunc(plus_one, rpc_timeout_sec=5)
    dead = context.Process(target=rpc_call, args=(remote, output, 11))
    healthy = None
    try:
        dead.start()
        deadline = time.monotonic() + 10
        while remote.requests.qsize() == 0:
            if time.monotonic() > deadline: raise TimeoutError('Caller request did not arrive')
            time.sleep(.01)
        cleanup_process(dead)
        # Process the dead client's isolated reply, then a fresh live caller.
        io.process_messages(.005)
        healthy = context.Process(target=rpc_call, args=(remote, output, 22))
        healthy.start(); pump([healthy])
        assert output.get(timeout=1) == ('ok', 23)
    finally:
        cleanup_process(dead)
        if healthy is not None: cleanup_process(healthy)
        remote.close(); output.cancel_join_thread(); output.close()


def finite_values(count):
    for value in range(count): yield value


def failing_values(kind):
    if kind == 'exit': os._exit(29)
    if kind == 'hang': time.sleep(60)
    if kind == 'error': raise ValueError('generator fixture failure')
    yield 1


def test_finite_generator_and_parallel_start_close_never_restart_closed_workers():
    generators = [SubprocessGenerator(finite_values, 3, start_now=False) for _ in range(2)]
    try:
        SubprocessGenerator.start_in_parallel(generators)
        for generator in generators:
            assert list(generator) == [0, 1, 2]
            assert generator._closed and not generator.p.is_alive()
            pid = generator.p.pid
            with pytest.raises(StopIteration): next(generator)
            assert generator.p.pid == pid
    finally:
        for generator in generators: generator.close()


@pytest.mark.parametrize('kind', ['error', 'exit', 'hang'])
def test_generator_failures_and_response_hangs_stop_without_unbounded_join(kind):
    generator = SubprocessGenerator(failing_values, kind, initialize_timeout_sec=2,
                                    response_timeout_sec=.3, shutdown_timeout_sec=.5)
    try:
        with pytest.raises((RuntimeError, TimeoutError)): next(generator)
        assert generator._closed and not generator.p.is_alive()
    finally:
        generator.close()


def test_parallel_start_exception_is_observable_instead_of_infinite_wait():
    generator = SubprocessGenerator(finite_values, 3, start_now=False)
    generator.generator_func = lambda count: iter(range(count))  # not spawn-picklable
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='startup failed'): SubprocessGenerator.start_in_parallel([generator])
    assert time.monotonic() - started < 3 and generator._closed
