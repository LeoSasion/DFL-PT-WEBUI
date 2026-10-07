import multiprocessing
import queue as Queue
import threading
import time
import traceback
import os


class SubprocessGenerator(object):

    @staticmethod
    def launch_thread(generator):
        try:
            generator._start()
        except BaseException as error:
            generator._start_error = error

    @staticmethod
    def start_in_parallel(generator_list):
        threads = []
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,), daemon=True)
            thread.start(); threads.append(thread)
        deadline = time.monotonic() + max((g.initialize_timeout_sec for g in generator_list), default=120.0)
        try:
            while not all(generator._is_started() for generator in generator_list):
                errors = [g._start_error for g in generator_list if g._start_error is not None]
                if errors: raise RuntimeError(f'SubprocessGenerator startup failed: {errors[0]}') from errors[0]
                if time.monotonic() >= deadline: raise TimeoutError('SubprocessGenerator startup deadline exceeded')
                time.sleep(.005)
        except BaseException:
            for generator in generator_list: generator.close()
            raise
        finally:
            for thread in threads: thread.join(timeout=.1)

    def __init__(self, generator_func, user_param=None, prefetch=2, start_now=True, *,
                 initialize_timeout_sec=120.0, response_timeout_sec=300.0, shutdown_timeout_sec=2.0):
        if type(prefetch) is not int or not 0 <= prefetch <= 1024:
            raise ValueError('prefetch must be an integer from 0 to 1024')
        for value in (initialize_timeout_sec, response_timeout_sec, shutdown_timeout_sec):
            if not isinstance(value, (int, float)) or not 0 < value <= 86400:
                raise ValueError('Generator timeouts must be positive finite durations')
        self.initialize_timeout_sec = float(initialize_timeout_sec)
        self.response_timeout_sec = float(response_timeout_sec)
        self.shutdown_timeout_sec = float(shutdown_timeout_sec)
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        self._closed = False
        self._start_error = None
        self._start_lock = threading.Lock()
        self._received_first = False
        self.owner_pid = os.getpid()
        if start_now:
            try: self._start()
            except BaseException:
                self.close(); raise

    def close(self):
        if self._closed: return
        if not self._start_lock.acquire(timeout=self.shutdown_timeout_sec):
            raise RuntimeError('Generator startup is still active; stop is unconfirmed')
        try:
            process = self.p
            if process is not None and process.pid is not None:
                if process.is_alive(): process.terminate()
                process.join(timeout=self.shutdown_timeout_sec)
                if process.is_alive():
                    process.kill(); process.join(timeout=self.shutdown_timeout_sec)
                if process.is_alive():
                    raise RuntimeError(f'Generator worker {process.pid} exit is unconfirmed')
            for channel in (self.sc_queue, self.cs_queue):
                channel.cancel_join_thread(); channel.close()
            self._closed = True
        finally:
            self._start_lock.release()

    def _start(self):
        with self._start_lock:
            if self._closed: raise StopIteration()
            if self.p is None:
                user_param = self.user_param
                self.user_param = None
                process = multiprocessing.Process(target=self.process_func, args=(user_param,))
                process.daemon = True
                self.p = process
                process.start()
                self.cs_queue._writer.close()
                self.sc_queue._reader.close()

    def _is_started(self):
        return self.p is not None and self.p.pid is not None

    def process_func(self, user_param):
        try:
            seed_env = os.environ.get('DFL_SUBPROC_SEED', None)
            if seed_env is not None and str(seed_env).strip() != '':
                try:
                    base_seed = int(seed_env)
                except Exception:
                    base_seed = None

                if base_seed is not None:
                    # 尽量稳定：用进程名中的序号（Process-1/Process-2...）做偏移。
                    proc_name = multiprocessing.current_process().name
                    proc_idx = 0
                    try:
                        tail = proc_name.split('-')[-1]
                        proc_idx = int(tail)
                    except Exception:
                        proc_idx = 0

                    seed = int(base_seed + proc_idx)

                    try:
                        import random

                        random.seed(seed)
                    except Exception:
                        pass

                    try:
                        import numpy as np

                        np.random.seed(seed % (2**32 - 1))
                    except Exception:
                        pass

                    try:
                        import torch

                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    except Exception:
                        pass

            self.generator_func = self.generator_func(user_param)
            while True:
                while self.prefetch > -1:
                    try:
                        gen_data = next(self.generator_func)
                    except StopIteration:
                        self.cs_queue.put(None)
                        return
                    self.cs_queue.put(gen_data)
                    self.prefetch -= 1
                self.sc_queue.get()
                self.prefetch += 1
        except Exception:
            # Propagate worker exception to parent to avoid deadlocks.
            try:
                self.cs_queue.put({'__error__': traceback.format_exc()})
            except Exception:
                pass
            return

    def __iter__(self):
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state['p'] = None
        state.pop('_start_lock', None)
        state['_start_error'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._start_lock = threading.Lock()

    def __next__(self):
        if self._closed: raise StopIteration()
        self._start()
        wait = self.response_timeout_sec if self._received_first else min(self.response_timeout_sec, self.initialize_timeout_sec)
        deadline = time.monotonic() + wait
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError(f'SubprocessGenerator response deadline exceeded ({wait:g}s)')
            try:
                gen_data = self.cs_queue.get(timeout=min(.2, remaining))
                break
            except Queue.Empty:
                if not self.p.is_alive():
                    self.close()
                    raise RuntimeError(f'SubprocessGenerator worker exited unexpectedly ({self.p.exitcode})')
            except (EOFError, OSError, ValueError) as error:
                self.close()
                raise RuntimeError('SubprocessGenerator response channel closed') from error
        if isinstance(gen_data, dict) and gen_data.get('__error__') is not None:
            error = gen_data['__error__']
            self.close()
            raise RuntimeError(f'SubprocessGenerator worker error:\n{error}')
        if gen_data is None:
            self.close(); raise StopIteration()
        self._received_first = True
        self.sc_queue.put(1)
        return gen_data

    def __del__(self):
        try:
            # The worker's target object may be collected before Queue's
            # exit finalizer flushes its final value/end marker. Only creator
            # teardown should cancel feeder joins and terminate the process.
            if os.getpid() == self.owner_pid: self.close()
        except Exception:
            pass
