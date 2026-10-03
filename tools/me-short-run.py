"""Run one fresh, small ME development training within a hard wall-clock cap.

The cap includes trainer startup, checkpoint saving and shutdown. This runner
never resumes or overwrites a model, and accepts only CPU or one GPU. Its JSON
records functional evidence; it does not assert final face-swap visual quality.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
MIN_SECONDS = 60
MAX_SECONDS = 3600
DEFAULT_SECONDS = 900
SAVE_RESERVE_SECONDS = 30
HARD_STOP_RESERVE_SECONDS = 1
RESULT_NAME = "short-run-result.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(target, value):
    target = Path(target)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Aligned SRC directory or packed faceset")
    parser.add_argument("--dst", required=True, help="Aligned DST directory or packed faceset")
    parser.add_argument("--output", required=True, help="New or empty isolated run directory")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_SECONDS,
                        help="Entire run cap, 60..3600 seconds (default: 900)")
    args = parser.parse_args(argv)
    if not MIN_SECONDS <= args.max_seconds <= MAX_SECONDS:
        parser.error("--max-seconds must be between 60 and 3600; long training is prohibited")
    return args


def small_config(device):
    return dict(archi="liae-ud", resolution=64, ae_dims=32, e_dims=16,
                d_dims=16, d_mask_dims=16, batch_size=2, use_fp16=device == "cuda:0",
                use_rg=False, data_workers=0, gan_power=0.0, true_face_power=0.0,
                pretrain=False)


def prepare_output(args, run_id):
    output = Path(args.output).resolve()
    for side in ("src", "dst"):
        dataset = Path(getattr(args, side)).resolve(strict=True)
        if dataset.is_dir() and (output == dataset or dataset in output.parents):
            raise ValueError("Run output cannot be inside an input faceset")
        setattr(args, side, str(dataset))
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file() or not (ROOT / "_internal" / "DeepFaceLab" / "me.py").is_file():
        raise FileNotFoundError("The project-local Python and me.py must exist")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Short run requires a new or empty output; existing model/output is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership also prevents simultaneous invocations claiming the
    # same initially empty directory between the check and the first write.
    with (output / ".short-run-owner.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(runId=run_id, runnerPid=os.getpid()), stream)
    (output / "model").mkdir()
    (output / "control.jsonl").touch(exist_ok=False)
    return output, python


def command_and_environment(args, output, python):
    command = [str(python), "-u", str(ROOT / "_internal" / "DeepFaceLab" / "me.py"),
               "web-train", "--src", args.src, "--dst", args.dst,
               "--model", str(output / "model"), "--name", "short-run",
               "--device", args.device, "--config", str(output / "config.json"),
               "--steps", "0", "--threads", "2", "--seed", "0",
               "--save-every", "1000", "--preview-every", "1000",
               "--backup-every", "5000", "--backup-keep", "2"]
    # Do not inherit another WebUI job's controls, evaluation or preview paths.
    env = {key: value for key, value in os.environ.items() if not key.startswith("DFL_WEB_")}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               DFL_WEB_CONTROL_FILE=str(output / "control.jsonl"),
               DFL_WEB_CONTROL_ACK_FILE=str(output / "control-ack.json"),
               DFL_WEB_HEARTBEAT_FILE=str(output / "trainer-heartbeat.json"),
               DFL_WEB_PREVIEW_FILE=str(output / "trainer-preview.png"))
    return command, env


class WindowsJob:
    """A kill-on-close job; the child is assigned before its first instruction.

    A suspended hidden process closes the spawn/assignment race. Windows kills
    every descendant in this owned job even if a data worker outlives its parent.
    No process is selected by executable name or by a later PID search.
    """

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = c = ctypes
        self.w = w = wintypes
        self.api = api = c.WinDLL("kernel32", use_last_error=True)
        handle = w.HANDLE
        api.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        api.CreateJobObjectW.restype = handle
        api.SetInformationJobObject.argtypes = [handle, c.c_int, c.c_void_p, w.DWORD]
        api.SetInformationJobObject.restype = w.BOOL
        api.AssignProcessToJobObject.argtypes = [handle, handle]
        api.AssignProcessToJobObject.restype = w.BOOL
        api.TerminateJobObject.argtypes = [handle, w.UINT]
        api.TerminateJobObject.restype = w.BOOL
        api.CloseHandle.argtypes = [handle]
        api.CloseHandle.restype = w.BOOL
        api.WaitForSingleObject.argtypes = [handle, w.DWORD]
        api.WaitForSingleObject.restype = w.DWORD
        api.GetExitCodeProcess.argtypes = [handle, c.POINTER(w.DWORD)]
        api.GetExitCodeProcess.restype = w.BOOL
        api.TerminateProcess.argtypes = [handle, w.UINT]
        api.TerminateProcess.restype = w.BOOL
        api.ResumeThread.argtypes = [handle]
        api.ResumeThread.restype = w.DWORD
        api.GetProcessTimes.argtypes = [handle, c.POINTER(w.FILETIME), c.POINTER(w.FILETIME),
                                       c.POINTER(w.FILETIME), c.POINTER(w.FILETIME)]
        api.GetProcessTimes.restype = w.BOOL
        api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        api.OpenProcess.restype = handle
        api.IsProcessInJob.argtypes = [handle, handle, c.POINTER(w.BOOL)]
        api.IsProcessInJob.restype = w.BOOL
        api.InitializeProcThreadAttributeList.argtypes = [c.c_void_p, w.DWORD, w.DWORD,
                                                        c.POINTER(c.c_size_t)]
        api.InitializeProcThreadAttributeList.restype = w.BOOL
        api.UpdateProcThreadAttribute.argtypes = [c.c_void_p, w.DWORD, c.c_size_t,
                                                 c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p]
        api.UpdateProcThreadAttribute.restype = w.BOOL
        api.DeleteProcThreadAttributeList.argtypes = [c.c_void_p]

        class BasicLimit(c.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c.c_longlong), ("PerJobUserTimeLimit", c.c_longlong),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                        ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

        class IOCounters(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
                        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimit(c.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                        ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]

        self.job = api.CreateJobObjectW(None, None)
        if not self.job:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(self.job, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            api.CloseHandle(self.job)
            self.job = None
            raise error
        self.process = None
        self.pid = None
        self.created_at = None

    def start(self, command, env, log, deadline):
        import msvcrt
        c, w, api = self.ctypes, self.w, self.api

        class StartupInfo(c.Structure):
            _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
                        ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
                        ("dwXSize", w.DWORD), ("dwYSize", w.DWORD), ("dwXCountChars", w.DWORD),
                        ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD), ("dwFlags", w.DWORD),
                        ("wShowWindow", w.WORD), ("cbReserved2", w.WORD), ("lpReserved2", c.c_void_p),
                        ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]

        class StartupInfoEx(c.Structure):
            _fields_ = [("StartupInfo", StartupInfo), ("lpAttributeList", c.c_void_p)]

        class ProcessInfo(c.Structure):
            _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE),
                        ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]

        api.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL,
                                       w.DWORD, c.c_void_p, w.LPCWSTR,
                                       c.POINTER(StartupInfoEx), c.POINTER(ProcessInfo)]
        api.CreateProcessW.restype = w.BOOL
        size = c.c_size_t()
        api.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        attributes = c.create_string_buffer(size.value)
        if not api.InitializeProcThreadAttributeList(attributes, 1, 0, c.byref(size)):
            raise c.WinError(c.get_last_error())
        info = StartupInfoEx()
        info.StartupInfo.cb = c.sizeof(info)
        info.StartupInfo.dwFlags = 0x101  # STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
        info.StartupInfo.wShowWindow = 0  # SW_HIDE
        info.lpAttributeList = c.cast(attributes, c.c_void_p)
        process = ProcessInfo()
        environment = c.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in
                                              sorted(env.items(), key=lambda item: item[0].upper())) + "\0\0")
        with open(os.devnull, "rb") as null_input:
            handles = (w.HANDLE * 2)(msvcrt.get_osfhandle(null_input.fileno()), msvcrt.get_osfhandle(log.fileno()))
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            try:
                # Explicit handle list prevents unrelated handles reaching the trainer.
                if not api.UpdateProcThreadAttribute(attributes, 0, 0x20002, handles, c.sizeof(handles), None, None):
                    raise c.WinError(c.get_last_error())
                info.StartupInfo.hStdInput = handles[0]
                info.StartupInfo.hStdOutput = info.StartupInfo.hStdError = handles[1]
                flags = 0x80000 | 0x400 | 0x4 | 0x08000000  # extended info, Unicode env, suspended, no window
                if not api.CreateProcessW(command[0], c.create_unicode_buffer(subprocess.list2cmdline(command)),
                                          None, None, True, flags, environment, str(ROOT), c.byref(info), c.byref(process)):
                    raise c.WinError(c.get_last_error())
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
                api.DeleteProcThreadAttributeList(attributes)
        self.process, self.pid = process.hProcess, process.dwProcessId
        try:
            if not api.AssignProcessToJobObject(self.job, self.process):
                raise c.WinError(c.get_last_error())
            created, exited, kernel, user = (w.FILETIME() for _ in range(4))
            if not api.GetProcessTimes(self.process, c.byref(created), c.byref(exited), c.byref(kernel), c.byref(user)):
                raise c.WinError(c.get_last_error())
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            self.created_at = (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat().replace("+00:00", "Z")
            if time.monotonic() >= deadline:
                raise TimeoutError("The run deadline elapsed during process startup")
            if api.ResumeThread(process.hThread) == 0xFFFFFFFF:
                raise c.WinError(c.get_last_error())
        except BaseException:
            api.TerminateProcess(self.process, 124)  # This still-suspended process belongs to this invocation.
            raise
        finally:
            api.CloseHandle(process.hThread)
        return self

    def poll(self):
        if self.process is None:
            return None
        status = self.api.WaitForSingleObject(self.process, 0)
        if status == 0x102:
            return None
        if status != 0:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        code = self.w.DWORD()
        if not self.api.GetExitCodeProcess(self.process, self.ctypes.byref(code)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return code.value

    def identify_trainer(self, state):
        # Windows venv launchers can start a separate python_base process. Trust
        # its heartbeat only while that PID demonstrably belongs to this job.
        if not isinstance(state, dict) or type(state.get("pid")) is not int or state["pid"] <= 0:
            return None
        c, w, api = self.ctypes, self.w, self.api
        handle = api.OpenProcess(0x1000 | 0x100000, False, state["pid"])
        if not handle:
            return None
        try:
            contained = w.BOOL()
            if (not api.IsProcessInJob(handle, self.job, c.byref(contained)) or not contained.value
                    or api.WaitForSingleObject(handle, 0) != 0x102):
                return None
            created, exited, kernel, user = (w.FILETIME() for _ in range(4))
            if not api.GetProcessTimes(handle, c.byref(created), c.byref(exited), c.byref(kernel), c.byref(user)):
                return None
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            created_at = (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat().replace("+00:00", "Z")
            if (not heartbeat_matches(state, state["pid"], created_at)
                    or datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    < datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))):
                return None
            return dict(pid=state["pid"], createdAt=created_at,
                        heartbeatStartedAt=state["startedAt"], containedInOwnedTree=True, observedAt=utc_now())
        finally:
            api.CloseHandle(handle)

    def kill(self):
        if not self.api.TerminateJobObject(self.job, 124):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self):
        # Kill-on-close also cleans up descendants after a normally exited trainer.
        if self.job:
            self.api.CloseHandle(self.job)
            self.job = None
        if self.process:
            self.api.CloseHandle(self.process)
            self.process = None


class PosixGroup:
    def __init__(self):
        self.process = None
        self.pid = None
        self.created_at = None

    def start(self, command, env, log, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("The run deadline elapsed during process startup")
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
        self.pid, self.created_at = self.process.pid, utc_now()
        return self

    def poll(self):
        return self.process.poll() if self.process else None

    def kill(self):
        if self.process:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def identify_trainer(self, state):
        if heartbeat_matches(state, self.pid, self.created_at) and self.poll() is None:
            return dict(pid=self.pid, createdAt=self.created_at,
                        heartbeatStartedAt=state["startedAt"], containedInOwnedTree=True, observedAt=utc_now())
        return None

    def close(self):
        self.kill()


class DeadlineWatchdog:
    """Independent of the monitor, file reading and graceful checkpoint saving."""

    def __init__(self, owner, deadline):
        self.owner = owner
        self.expired = False
        self.error = None
        self.fired_at = None
        self.timer = threading.Timer(max(0, deadline - time.monotonic()), self.expire)
        self.timer.daemon = True

    def expire(self):
        self.expired = True
        self.fired_at = utc_now()
        try:
            self.owner.kill()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def start(self):
        self.timer.start()

    def cancel(self):
        self.timer.cancel()
        if self.timer.ident is not None:
            self.timer.join(timeout=0.5)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def append_close(output, run_id):
    request = dict(operation="close", requestedAt=utc_now(), runId=run_id)
    with (output / "control.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(request) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return request


def ack_matches(ack, request, checkpoint):
    if not isinstance(ack, dict) or not request:
        return False
    try:
        return (ack.get("operation") == "close" and ack.get("requestedAt") == request["requestedAt"]
                and ack.get("status") == "completed" and type(ack.get("iteration")) is int
                and ack["iteration"] >= 1 and Path(ack.get("checkpoint", "")).resolve() == checkpoint.resolve())
    except (TypeError, OSError):
        return False


def heartbeat_matches(state, pid, created_at):
    if not isinstance(state, dict) or type(state.get("pid")) is not int or state["pid"] != pid:
        return False
    try:
        started = datetime.fromisoformat(state["startedAt"].replace("Z", "+00:00"))
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return started.tzinfo is not None and started >= created
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def run(args):
    # Start before output setup and process spawn: startup consumes the same cap.
    began, started_at = time.monotonic(), utc_now()
    if not MIN_SECONDS <= args.max_seconds <= MAX_SECONDS or args.device not in ("cpu", "cuda:0"):
        raise ValueError("Only cpu/cuda:0 and a 60..3600-second run are supported")
    run_id = str(uuid.uuid4())
    output, python = prepare_output(args, run_id)
    config = small_config(args.device)
    atomic_json(output / "config.json", config)
    command, env = command_and_environment(args, output, python)
    deadline = began + args.max_seconds
    hard_stop = deadline - HARD_STOP_RESERVE_SECONDS
    close_at = deadline - SAVE_RESERVE_SECONDS
    report = dict(schemaVersion=1, runId=run_id, startedAt=started_at, stoppedAt=None,
                  maxSeconds=args.max_seconds, gracefulStopAfterSeconds=args.max_seconds - SAVE_RESERVE_SECONDS,
                  hardStopAfterSeconds=args.max_seconds - HARD_STOP_RESERVE_SECONDS,
                  device=args.device, config=config, src=args.src, dst=args.dst,
                  output=str(output), model=str(output / "model"), command=command,
                  runnerPid=os.getpid(), process=None, trainerProcess=None, exitCode=None, forcedStop=False,
                  close=None, heartbeat=None, completed=False)
    atomic_json(output / RESULT_NAME, report)
    owner = watchdog = None
    request = None
    next_progress_at = began
    interrupted = False
    previous_handlers = {}

    def stop_runner(signum, frame):
        nonlocal interrupted
        interrupted = True

    try:
        owner = WindowsJob() if os.name == "nt" else PosixGroup()
        watchdog = DeadlineWatchdog(owner, hard_stop)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop_runner)
        watchdog.start()
        with (output / "training.log").open("xb", buffering=0) as log:
            owner.start(command, env, log, hard_stop)
            report["process"] = dict(pid=owner.pid, createdAt=owner.created_at)
            atomic_json(output / RESULT_NAME, report)
            while owner.poll() is None:
                state = read_json(output / "trainer-heartbeat.json")
                if report["trainerProcess"] is None:
                    identity = owner.identify_trainer(state)
                    if identity is not None:
                        report["trainerProcess"] = identity
                        atomic_json(output / RESULT_NAME, report)
                if time.monotonic() >= next_progress_at:
                    observed = state if isinstance(state, dict) else {}
                    print(f"[ME short run] elapsed={time.monotonic() - began:.1f}s/{args.max_seconds}s "
                          f"phase={observed.get('phase', 'starting')} iteration={observed.get('iteration', 0)}", flush=True)
                    next_progress_at = time.monotonic() + 30
                if request is None and (interrupted or time.monotonic() >= close_at):
                    request = append_close(output, run_id)
                    report["close"] = {**request, "ackMatched": False}
                    atomic_json(output / RESULT_NAME, report)
                if time.monotonic() >= deadline:
                    owner.kill()
                    report["forcedStop"] = True
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            report["exitCode"] = owner.poll()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if owner is not None:
            try:
                owner.kill()
            except Exception as kill_error:
                report["killError"] = f"{type(kill_error).__name__}: {kill_error}"
            report["forcedStop"] = owner.pid is not None
            report["exitCode"] = owner.poll()
    finally:
        if watchdog is not None:
            watchdog.cancel()
            report["forcedStop"] = report["forcedStop"] or watchdog.expired
        report["watchdog"] = dict(fired=bool(watchdog and watchdog.expired),
                                  firedAt=watchdog.fired_at if watchdog else None,
                                  error=watchdog.error if watchdog else None,
                                  containment="windows-job-object" if os.name == "nt" else "posix-process-group")
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        ack = read_json(output / "control-ack.json")
        state = read_json(output / "trainer-heartbeat.json")
        report["close"] = {**(request or {}), "ackMatched": ack_matches(ack, request, output / "model" / "me.pt"), "ack": ack}
        trainer = report["trainerProcess"] or {}
        matched = (heartbeat_matches(state, trainer.get("pid"), trainer.get("createdAt"))
                   and state.get("startedAt") == trainer.get("heartbeatStartedAt"))
        report["heartbeat"] = dict(identityMatched=bool(matched), state=state)
        if owner is not None:
            owner.close()
        report["stoppedAt"] = utc_now()
        report["elapsedSeconds"] = round(time.monotonic() - began, 6)
        report["wallCapMet"] = report["elapsedSeconds"] <= args.max_seconds
        report["interrupted"] = interrupted
        report["completed"] = (report["exitCode"] == 0 and not report["forcedStop"]
                               and report["wallCapMet"] and report["close"]["ackMatched"]
                               and report["heartbeat"]["identityMatched"]
                               and isinstance(state, dict) and state.get("phase") == "finished"
                               and state.get("iteration") == (ack or {}).get("iteration"))
        atomic_json(output / RESULT_NAME, report)
    return report


def main(argv=None):
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
