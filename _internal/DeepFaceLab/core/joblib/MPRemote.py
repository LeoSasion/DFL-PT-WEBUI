"""Finite owner-thread RPC with a private reply pipe per request.

Callers never hold a shared mutex while waiting. A stdlib Manager queue
accepts complete requests, so an exiting caller cannot leave a Queue writer
lock held. Wrappers on the same owner share one manager.
"""
import multiprocessing
from multiprocessing.managers import SyncManager
import os
import queue
import threading
import time
import traceback
import uuid

from core.interact import interact as io
from core.process_identity import process_identity


class RPCRemoteError(RuntimeError): pass
class RPCOwnerStopped(RuntimeError): pass
class RPCShutdownError(RuntimeError): pass


_pools = {}
_pools_lock = threading.Lock()


def _watch_owner(pid, identity):
    # Manager finalizers cannot run after an owner's abrupt os._exit/crash.
    # The server therefore has its own finite liveness watchdog.
    def watch():
        while True:
            time.sleep(.1)
            current = process_identity(pid)
            if current is None or current != identity: os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def _pool_acquire():
    pid = os.getpid()
    with _pools_lock:
        pool = _pools.get(pid)
        if pool is None:
            manager = SyncManager(ctx=multiprocessing.get_context('spawn'), shutdown_timeout=1.0)
            failures = []
            def start():
                try: manager.start(initializer=_watch_owner, initargs=(pid, process_identity(pid)))
                except BaseException as error: failures.append(error)
            startup = threading.Thread(target=start, daemon=True)
            startup.start(); startup.join(120.0)
            if startup.is_alive():
                process = getattr(manager, '_process', None)
                if process is not None and process.is_alive():
                    process.terminate(); process.join(1.0)
                if process is not None and process.is_alive():
                    process.kill(); process.join(1.0)
                raise TimeoutError('RPC manager initialization deadline exceeded (120s)')
            if failures: raise RPCOwnerStopped(f'RPC manager initialization failed: {failures[0]}') from failures[0]
            pool = {'manager': manager, 'references': 0}
            _pools[pid] = pool
        pool['references'] += 1
        return pool['manager']


def _pool_release():
    with _pools_lock:
        pool = _pools.get(os.getpid())
        if pool is None: return
        pool['references'] -= 1
        if pool['references']: return
        del _pools[os.getpid()]
    manager = pool['manager']
    process = manager._process
    shutdown = threading.Thread(target=manager.shutdown, daemon=True)
    shutdown.start(); shutdown.join(3.0)
    if process.is_alive():
        process.terminate(); process.join(1.0)
    if process.is_alive():
        process.kill(); process.join(1.0)
    if process.is_alive():
        raise RPCShutdownError(f'RPC manager process {process.pid} exit is unconfirmed')
    shutdown.join(.1)


class MPRemote:
    def __init__(self, *, rpc_timeout_sec=300.0):
        if not isinstance(rpc_timeout_sec, (int, float)) or not 0 < rpc_timeout_sec <= 86400:
            raise ValueError('RPC timeout must be a positive finite duration')
        self.rpc_timeout_sec = float(rpc_timeout_sec)
        self.owner_pid = os.getpid()
        self.owner_identity = process_identity(self.owner_pid)
        self.callback_thread_id = threading.get_ident()
        self.closed = multiprocessing.get_context('spawn').Event()
        self._local_closed = False
        self._owner_registered = False
        manager = _pool_acquire()
        self.manager_pid = manager._process.pid
        try:
            self.requests = manager.Queue(maxsize=128)
            io.add_process_messages_callback(self.io_callback)
            self._owner_registered = True
        except BaseException:
            _pool_release()
            raise

    def _execute(self, args, kwargs): raise NotImplementedError

    def _owner_alive(self):
        identity = process_identity(self.owner_pid)
        return identity is not None and identity == self.owner_identity

    def io_callback(self):
        if self._local_closed or self.closed.is_set(): return
        for _ in range(128):
            try: request = self.requests.get_nowait()
            except queue.Empty: return
            except (EOFError, BrokenPipeError, ConnectionError, OSError):
                self.closed.set(); return
            reply = request['reply']
            try:
                if time.monotonic() >= request['deadline']:
                    response = {'id': request['id'], 'ok': False, 'error': 'RPC request expired before owner execution'}
                else:
                    try:
                        result = self._execute(request['args'], request['kwargs'])
                        response = {'id': request['id'], 'ok': True, 'result': result}
                    except Exception as error:
                        response = {'id': request['id'], 'ok': False,
                                    'error': f'{type(error).__name__}: {error}',
                                    'traceback': traceback.format_exc()[-16384:]}
                try: reply.send(response)
                except (EOFError, BrokenPipeError, ConnectionError, OSError): pass
            finally:
                reply.close()

    def __call__(self, *args, **kwargs):
        if self._local_closed or self.closed.is_set(): raise RPCOwnerStopped('RPC is closed')
        if not self._owner_alive(): raise RPCOwnerStopped('RPC owner process stopped or its PID was reused')
        if os.getpid() == self.owner_pid:
            return self._execute(args, kwargs)
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.rpc_timeout_sec
        receive, send = multiprocessing.Pipe(duplex=False)
        try:
            try:
                self.requests.put({'id': request_id, 'args': args, 'kwargs': kwargs,
                                   'deadline': deadline, 'reply': send}, timeout=self.rpc_timeout_sec)
                # Manager.put has synchronously duplicated this handle. Do
                # not keep a local writer that would hide EOF if owner dies.
                send.close()
            except queue.Full as error:
                raise TimeoutError('RPC request queue timed out') from error
            except (EOFError, BrokenPipeError, ConnectionError, OSError) as error:
                raise RPCOwnerStopped('RPC request service stopped') from error
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError(f'RPC response timed out after {self.rpc_timeout_sec:g} seconds')
                if self.closed.is_set() or not self._owner_alive(): raise RPCOwnerStopped('RPC owner stopped while awaiting response')
                if receive.poll(min(remaining, .05)):
                    try: response = receive.recv()
                    except (EOFError, BrokenPipeError, ConnectionError, OSError) as error:
                        raise RPCOwnerStopped('RPC reply channel closed') from error
                    if response.get('id') != request_id: raise RPCRemoteError('RPC reply request identity mismatch')
                    if not response.get('ok'): raise RPCRemoteError(response.get('error', 'RPC callback failed'))
                    return response['result']
        finally:
            receive.close(); send.close()

    def close(self):
        if self._local_closed: return
        self._local_closed = True
        if os.getpid() != self.owner_pid or process_identity(self.owner_pid) != self.owner_identity: return
        self.closed.set()
        if self._owner_registered:
            callbacks = io.process_messages_callbacks.get(self.callback_thread_id, [])
            try: callbacks.remove(self.io_callback)
            except ValueError: pass
            self._owner_registered = False
            _pool_release()

    def __getstate__(self):
        return {'requests': self.requests, 'closed': self.closed,
                'owner_pid': self.owner_pid, 'owner_identity': self.owner_identity,
                'rpc_timeout_sec': self.rpc_timeout_sec, '_local_closed': False,
                '_owner_registered': False, 'callback_thread_id': self.callback_thread_id}

    def __setstate__(self, state): self.__dict__.update(state)
