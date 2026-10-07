import traceback
import multiprocessing
import time
import sys
import os
import queue
from core.interact import interact as io


class Subprocessor(object):

    class SilenceException(Exception):
        pass

    class LifecycleError(RuntimeError):
        pass

    class Cli(object):
        def __init__ ( self, client_dict ):
            # 重要：multiprocessing.spawn 会 exec 新解释器。
            # 在某些 PTY/IDE 环境下，标准流 fd(0/1/2) 可能被标记为 close-on-exec，
            # 会导致子进程 Python 初始化阶段报错：
            #   Fatal Python error: init_sys_streams: can't initialize sys standard streams
            # 因此这里确保标准 fd 可继承；若缺失则绑定到 /dev/null。
            for fd in (0, 1, 2):
                try:
                    os.fstat(fd)
                except Exception:
                    try:
                        dn = os.open(os.devnull, os.O_RDWR)
                        os.dup2(dn, fd)
                        os.close(dn)
                    except Exception:
                        pass
                try:
                    os.set_inheritable(fd, True)
                except Exception:
                    pass

            s2c = multiprocessing.Queue()
            c2s = multiprocessing.Queue()
            self.p = multiprocessing.Process(target=self._subprocess_run, args=(client_dict,s2c,c2s) )
            self.s2c = s2c
            self.c2s = c2s
            self.p.daemon = True
            try:
                self.p.start()
                # Parent never writes results or reads requests. Closing its
                # duplicate endpoints lets a truncated child response report
                # EOF instead of blocking forever after a silent child exit.
                c2s._writer.close()
                s2c._reader.close()
            except BaseException:
                for channel in (s2c, c2s):
                    channel.cancel_join_thread(); channel.close()
                raise
            self.shutdown_timeout_sec = 2.0
            self.pid = self.p.pid
            self.exit_confirmed = False
            self._queues_closed = False

            self.state = None
            self.sent_time = None
            self.sent_data = None
            self.name = None
            self.host_dict = None

        def kill(self):
            if self.p.is_alive(): self.p.terminate()
            self.p.join(timeout=self.shutdown_timeout_sec)
            if self.p.is_alive():
                self.p.kill(); self.p.join(timeout=self.shutdown_timeout_sec)
            if self.p.is_alive():
                raise Subprocessor.LifecycleError(f'Subprocess {self.pid} exit remains unconfirmed')
            self.exit_confirmed = True
            if not self._queues_closed:
                for channel in (self.s2c, self.c2s):
                    channel.cancel_join_thread(); channel.close()
                self._queues_closed = True

        #overridable optional
        def on_initialize(self, client_dict):
            #initialize your subprocess here using client_dict
            pass

        #overridable optional
        def on_finalize(self):
            #finalize your subprocess here
            pass

        #overridable
        def process_data(self, data):
            #process 'data' given from host and return result
            raise NotImplementedError

        #overridable optional
        def get_data_name (self, data):
            #return string identificator of your 'data'
            return "undefined"

        def log_info(self, msg): self.c2s.put ( {'op': 'log_info', 'msg':msg } )
        def log_err(self, msg): self.c2s.put ( {'op': 'log_err' , 'msg':msg } )
        def progress_bar_inc(self, c): self.c2s.put ( {'op': 'progress_bar_inc' , 'c':c } )

        def _subprocess_run(self, client_dict, s2c, c2s):
            self.c2s = c2s
            data = None
            is_error = False
            try:
                self.on_initialize(client_dict)

                c2s.put ( {'op': 'init_ok'} )

                while True:
                    msg = s2c.get()
                    op = msg.get('op','')
                    if op == 'data':
                        data = msg['data']
                        result = self.process_data (data)
                        c2s.put ( {'op': 'success', 'data' : data, 'result' : result} )
                        data = None
                    elif op == 'close':
                        break

                    time.sleep(0.001)

                self.on_finalize()
                c2s.put ( {'op': 'finalized'} )
            except Subprocessor.SilenceException as e:
                is_error = True
                c2s.put ( {'op': 'error', 'data' : data} )
            except Exception as e:
                is_error = True
                err_msg = traceback.format_exc()
                c2s.put ( {'op': 'error', 'data' : data, 'err_msg' : err_msg} )

            # Never wait without a bound for a Queue feeder during teardown.
            c2s.cancel_join_thread(); c2s.close()
            s2c.cancel_join_thread(); s2c.close()
            self.c2s = None
            if is_error: raise SystemExit(1)

        # disable pickling
        def __getstate__(self):
            return dict()
        def __setstate__(self, d):
            self.__dict__.update(d)

    #overridable
    def __init__(self, name, SubprocessorCli_class, no_response_time_sec = 0, io_loop_sleep_time=0.005, initialize_subprocesses_in_serial=False,
                 initialize_timeout_sec=120.0, finalize_timeout_sec=30.0, shutdown_timeout_sec=2.0):
        if not issubclass(SubprocessorCli_class, Subprocessor.Cli):
            raise ValueError("SubprocessorCli_class must be subclass of Subprocessor.Cli")

        self.name = name
        self.SubprocessorCli_class = SubprocessorCli_class
        self.no_response_time_sec = no_response_time_sec
        self.io_loop_sleep_time = io_loop_sleep_time
        self.initialize_subprocesses_in_serial = initialize_subprocesses_in_serial
        for name, value in (('initialize', initialize_timeout_sec), ('finalize', finalize_timeout_sec), ('shutdown', shutdown_timeout_sec)):
            if not isinstance(value, (int, float)) or not 0 < value <= 86400:
                raise ValueError(f'{name} timeout must be a positive finite duration')
        self.initialize_timeout_sec = float(initialize_timeout_sec)
        self.finalize_timeout_sec = float(finalize_timeout_sec)
        self.shutdown_timeout_sec = float(shutdown_timeout_sec)
        self.lifecycle_errors = []

    #overridable
    def process_info_generator(self):
        #yield per process (name, host_dict, client_dict)
        raise NotImplementedError

    #overridable optional
    def on_clients_initialized(self):
        #logic when all subprocesses initialized and ready
        pass

    #overridable optional
    def on_clients_finalized(self):
        #logic when all subprocess finalized
        pass

    #overridable
    def get_data(self, host_dict):
        #return data for processing here
        raise NotImplementedError

    #overridable
    def on_data_return (self, host_dict, data):
        #you have to place returned 'data' back to your queue
        raise NotImplementedError

    #overridable
    def on_result (self, host_dict, data, result):
        #your logic what to do with 'result' of 'data'
        raise NotImplementedError

    #overridable
    def get_result(self):
        #return result that will be returned in func run()
        return None

    #overridable
    def on_tick(self):
        #tick in main loop
        #return True if system can be finalized when no data in get_data, orelse False
        return True

    #overridable
    def on_check_run(self):
        return True

    def run(self):
        if not self.on_check_run():
            return self.get_result()
        self.clis, all_clis = [], []
        host_initialized = False
        host_finalized = False

        def messages(cli):
            while True:
                try:
                    yield cli.c2s.get_nowait()
                except queue.Empty:
                    return
                except (EOFError, OSError, ValueError):
                    return

        def failed(cli, phase, message, return_data=False):
            self.lifecycle_errors.append({'name': cli.name, 'pid': cli.pid, 'phase': phase, 'message': message})
            io.log_err(f'{self.name}/{cli.name}: {message}')
            data = cli.sent_data
            cli.sent_data = None
            try:
                if return_data and data is not None:
                    self.on_data_return(cli.host_dict, data)
            finally:
                cli.kill()
                if cli in self.clis: self.clis.remove(cli)

        def initialize(cli):
            for obj in messages(cli):
                op = obj.get('op', '')
                if op == 'init_ok': cli.state = 0
                elif op == 'log_info': io.log_info(obj['msg'])
                elif op == 'log_err': io.log_err(obj['msg'])
                elif op == 'error':
                    failed(cli, 'initialize', obj.get('err_msg', 'Worker initialization failed'))
                    return
            if cli not in self.clis: return
            if not cli.p.is_alive():
                failed(cli, 'initialize', f'Worker exited during initialization (exitcode={cli.p.exitcode})')
            elif cli.state != 0 and time.monotonic() - cli.init_started >= self.initialize_timeout_sec:
                failed(cli, 'initialize', f'Initialization deadline exceeded ({self.initialize_timeout_sec:g}s)')

        try:
            for name, host_dict, client_dict in self.process_info_generator():
                try:
                    cli = self.SubprocessorCli_class(client_dict)
                except BaseException as error:
                    raise self.LifecycleError(f'Unable to start subprocess {name}: {error}') from error
                cli.state = 1; cli.sent_time = 0; cli.sent_data = None
                cli.name = name; cli.host_dict = host_dict
                cli.shutdown_timeout_sec = self.shutdown_timeout_sec
                cli.init_started = time.monotonic()
                self.clis.append(cli); all_clis.append(cli)
                if self.initialize_subprocesses_in_serial:
                    while cli in self.clis and cli.state != 0:
                        initialize(cli)
                        io.process_messages(.005)
            while self.clis and any(cli.state != 0 for cli in self.clis):
                for cli in self.clis[:]: initialize(cli)
                io.process_messages(.005)
            if not self.clis:
                raise self.LifecycleError(f'Unable to initialize Subprocessor {self.name}; {self.lifecycle_errors}')
            host_initialized = True
            self.on_clients_initialized()
            while True:
                for cli in self.clis[:]:
                    for obj in messages(cli):
                        op = obj.get('op', '')
                        if op == 'success':
                            self.on_result(cli.host_dict, obj['data'], obj['result'])
                            cli.sent_data = None; cli.state = 0; cli.sent_time = 0
                        elif op == 'error':
                            failed(cli, 'process', obj.get('err_msg', 'Worker processing failed'), return_data=True)
                            break
                        elif op == 'log_info': io.log_info(obj['msg'])
                        elif op == 'log_err': io.log_err(obj['msg'])
                        elif op == 'progress_bar_inc': io.progress_bar_inc(obj['c'])
                    if cli not in self.clis: continue
                    if not cli.p.is_alive():
                        failed(cli, 'process', f'Worker exited without a result (exitcode={cli.p.exitcode})', return_data=True)
                    elif (cli.state == 1 and cli.sent_time and self.no_response_time_sec
                          and time.monotonic() - cli.sent_time > self.no_response_time_sec):
                        failed(cli, 'process', f'Response deadline exceeded ({self.no_response_time_sec:g}s)', return_data=True)
                if not self.clis:
                    raise self.LifecycleError(f'All subprocesses stopped before completion: {self.lifecycle_errors}')
                for cli in self.clis[:]:
                    if cli.state == 0:
                        data = self.get_data(cli.host_dict)
                        if data is not None:
                            cli.s2c.put({'op': 'data', 'data': data})
                            cli.sent_time = time.monotonic(); cli.sent_data = data; cli.state = 1
                if self.io_loop_sleep_time:
                    io.process_messages(self.io_loop_sleep_time)
                if self.on_tick() and all(cli.state == 0 for cli in self.clis): break
            for cli in self.clis:
                cli.s2c.put({'op': 'close'}); cli.sent_time = time.monotonic()
            while any(cli.state != 2 for cli in self.clis):
                for cli in self.clis:
                    if cli.state == 2: continue
                    responses = list(messages(cli))
                    if any(obj.get('op') == 'error' for obj in responses):
                        self.lifecycle_errors.append({'name': cli.name, 'pid': cli.pid, 'phase': 'finalize', 'message': 'Worker finalization raised an exception'})
                        raise self.LifecycleError(f'Worker finalization failed: {cli.name}')
                    finalized = any(obj.get('op') == 'finalized' for obj in responses)
                    if not cli.p.is_alive() and cli.p.exitcode not in (0, None):
                        self.lifecycle_errors.append({'name': cli.name, 'pid': cli.pid, 'phase': 'finalize', 'message': f'Worker exitcode {cli.p.exitcode}'})
                        raise self.LifecycleError(f'Worker exited during finalization: {cli.name} ({cli.p.exitcode})')
                    if time.monotonic() - cli.sent_time >= self.finalize_timeout_sec and not finalized and cli.p.is_alive():
                        self.lifecycle_errors.append({'name': cli.name, 'pid': cli.pid, 'phase': 'finalize', 'message': 'Worker finalization deadline exceeded'})
                        raise self.LifecycleError(f'Worker finalization deadline exceeded: {cli.name}')
                    if finalized or not cli.p.is_alive():
                        cli.kill(); cli.state = 2
                io.process_messages(.005)
            host_finalized = True; self.on_clients_finalized()
            return self.get_result()
        finally:
            cleanup_errors = []
            for cli in all_clis:
                try: cli.kill()
                except BaseException as error: cleanup_errors.append(str(error))
            if host_initialized and not host_finalized:
                try: self.on_clients_finalized()
                except BaseException as error: cleanup_errors.append(str(error))
            if cleanup_errors:
                raise self.LifecycleError('Subprocessor cleanup failed: ' + '; '.join(cleanup_errors))
