"""Short-run guardrails and owned-process deadline contracts; no training."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("me_short_run", ROOT / "tools" / "me-short-run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments(tmp_path, **updates):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(exist_ok=True)
    dst.mkdir(exist_ok=True)
    return SimpleNamespace(src=str(src), dst=str(dst), output=str(tmp_path / "run"),
                           device="cpu", max_seconds=60, **updates)


@pytest.mark.parametrize("seconds", [0, 59, 3601, 43200])
def test_cli_rejects_long_or_invalid_wall_cap(seconds):
    with pytest.raises(SystemExit) as error:
        runner.parse_args(["--src", "src", "--dst", "dst", "--output", "run", "--max-seconds", str(seconds)])
    assert error.value.code == 2


def test_cli_default_small_config_and_single_device():
    args = runner.parse_args(["--src", "src", "--dst", "dst", "--output", "run"])
    assert args.max_seconds == 900 and args.device == "cuda:0"
    assert runner.small_config("cuda:0") == dict(archi="liae-ud", resolution=64, ae_dims=32,
        e_dims=16, d_dims=16, d_mask_dims=16, batch_size=2, use_fp16=True, use_rg=False,
        data_workers=0, gan_power=0.0, true_face_power=0.0, pretrain=False)
    assert not runner.small_config("cpu")["use_fp16"]
    with pytest.raises(SystemExit):
        runner.parse_args(["--src", "src", "--dst", "dst", "--output", "run", "--device", "cuda:0,1"])


@pytest.mark.parametrize("name", ["me.pt", "training.log", ".short-run-owner.json"])
def test_nonempty_output_refuses_without_replacing_existing_files(tmp_path, name):
    args = arguments(tmp_path)
    output = Path(args.output)
    output.mkdir()
    existing = output / name
    existing.write_bytes(b"preserve this existing run")
    with pytest.raises(FileExistsError, match="never overwritten"):
        runner.prepare_output(args, "new-run-id")
    assert existing.read_bytes() == b"preserve this existing run"
    assert list(output.iterdir()) == [existing]


def test_fresh_output_claim_and_environment_are_isolated(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    output, python = runner.prepare_output(args, "new-run-id")
    assert json.loads((output / ".short-run-owner.json").read_text())["runId"] == "new-run-id"
    assert (output / "model").is_dir()
    monkeypatch.setenv("DFL_WEB_CONTROL_FILE", "other-job-control.jsonl")
    monkeypatch.setenv("DFL_WEB_EVAL_MANIFEST", "other-job-manifest.json")
    command, env = runner.command_and_environment(args, output, python)
    assert command[3] == "web-train"
    assert "--resume" not in command and "--initialize-from" not in command
    assert env["DFL_WEB_CONTROL_FILE"] == str(output / "control.jsonl")
    assert env["DFL_WEB_CONTROL_ACK_FILE"] == str(output / "control-ack.json")
    assert env["DFL_WEB_HEARTBEAT_FILE"] == str(output / "trainer-heartbeat.json")
    assert "DFL_WEB_EVAL_MANIFEST" not in env
    with pytest.raises(FileExistsError):
        runner.prepare_output(args, "competing-run-id")


def test_ack_and_heartbeat_must_identify_this_control_and_process(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "control.jsonl").touch()
    request = runner.append_close(output, "run-one")
    checkpoint = output / "model" / "me.pt"
    ack = dict(operation="close", requestedAt=request["requestedAt"], status="completed",
               iteration=10, checkpoint=str(checkpoint))
    assert runner.ack_matches(ack, request, checkpoint)
    assert not runner.ack_matches({**ack, "requestedAt": "previous request"}, request, checkpoint)
    assert not runner.ack_matches({**ack, "checkpoint": str(tmp_path / "unrelated" / "me.pt")}, request, checkpoint)
    assert not runner.ack_matches({**ack, "status": "failed"}, request, checkpoint)
    assert not runner.ack_matches({**ack, "iteration": 0}, request, checkpoint)
    created = datetime.now(timezone.utc)
    state = dict(pid=12345, startedAt=(created + timedelta(seconds=1)).isoformat())
    assert runner.heartbeat_matches(state, 12345, created.isoformat())
    assert not runner.heartbeat_matches(state, 54321, created.isoformat())
    assert not runner.heartbeat_matches(state, 12345, (created + timedelta(seconds=2)).isoformat())


def test_watchdog_kills_only_its_owned_tree_without_waiting_for_ack(monkeypatch):
    timers = []

    class FakeTimer:
        ident = None

        def __init__(self, delay, callback):
            self.delay, self.callback, self.started, self.cancelled = delay, callback, False, False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    class Owner:
        kills = 0

        def kill(self):
            self.kills += 1

    monkeypatch.setattr(runner.threading, "Timer", FakeTimer)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)
    owner = Owner()
    watchdog = runner.DeadlineWatchdog(owner, 159.0)
    watchdog.start()
    assert timers[0].delay == 59 and timers[0].started
    timers[0].callback()
    assert owner.kills == 1 and watchdog.expired and watchdog.fired_at
    watchdog.cancel()
    assert timers[0].cancelled


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_hidden_suspended_spawn_is_owned_and_can_be_killed(tmp_path):
    owner = runner.WindowsJob()
    try:
        with (tmp_path / "child.log").open("xb", buffering=0) as log:
            code = ('import json, os, time; from datetime import datetime, timezone; '
                    'print(json.dumps(dict(pid=os.getpid(), startedAt=datetime.now(timezone.utc).isoformat())), flush=True); '
                    'time.sleep(60)')
            owner.start([sys.executable, "-u", "-c", code],
                        dict(os.environ), log, time.monotonic() + 10)
            assert owner.pid and owner.created_at and owner.poll() is None
            ready_deadline = time.monotonic() + 5
            while not (tmp_path / "child.log").read_bytes().strip():
                assert owner.poll() is None
                assert time.monotonic() < ready_deadline
                time.sleep(0.05)
            state = json.loads((tmp_path / "child.log").read_bytes())
            identity = owner.identify_trainer(state)
            assert identity["pid"] == state["pid"] and identity["containedInOwnedTree"]
            # The venv launcher PID and actual python_base heartbeat PID can differ.
            assert identity["heartbeatStartedAt"] == state["startedAt"]
            assert not owner.identify_trainer({**state, "pid": os.getpid()})
            owner.kill()
            stopped_deadline = time.monotonic() + 5
            while owner.poll() is None:
                assert time.monotonic() < stopped_deadline
                time.sleep(0.05)
            assert owner.poll() == 124
    finally:
        owner.close()


@pytest.mark.parametrize("graceful", [True, False])
def test_monitor_budget_includes_startup_and_watchdog_records_forced_stop(tmp_path, monkeypatch, graceful):
    args = arguments(tmp_path)
    clock = SimpleNamespace(now=0.0)
    state = dict(pid=222, startedAt=runner.utc_now(), phase="training", iteration=10)
    active_watchdogs = []

    class Owner:
        pid = 111
        created_at = state["startedAt"]
        exit_code = None

        def start(self, command, env, log, deadline):
            clock.now += 20  # Startup must consume the original cap.
            return self

        def identify_trainer(self, observed):
            return dict(pid=222, createdAt=self.created_at,
                        heartbeatStartedAt=state["startedAt"], containedInOwnedTree=True)

        def poll(self):
            control = Path(args.output) / "control.jsonl"
            if graceful and self.exit_code is None and control.exists() and control.read_text().strip():
                request = json.loads(control.read_text().strip())
                runner.atomic_json(Path(args.output) / "control-ack.json", dict(operation="close",
                    requestedAt=request["requestedAt"], status="completed", iteration=10,
                    checkpoint=str(Path(args.output) / "model" / "me.pt")))
                state["phase"] = "finished"
                self.exit_code = 0
            return self.exit_code

        def kill(self):
            self.exit_code = 124

        def close(self):
            pass

    class Watchdog:
        expired, fired_at, error = False, None, None

        def __init__(self, owner, deadline):
            self.owner, self.deadline = owner, deadline
            active_watchdogs.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    def sleep(amount):
        clock.now += amount
        for watchdog in active_watchdogs:
            if not watchdog.expired and clock.now >= watchdog.deadline:
                watchdog.expired, watchdog.fired_at = True, runner.utc_now()
                watchdog.owner.kill()

    original_read = runner.read_json
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(runner, "WindowsJob", Owner)
    monkeypatch.setattr(runner, "PosixGroup", Owner)
    monkeypatch.setattr(runner, "DeadlineWatchdog", Watchdog)
    monkeypatch.setattr(runner, "read_json", lambda path: dict(state) if Path(path).name == "trainer-heartbeat.json" else original_read(path))
    report = runner.run(args)
    assert report["process"]["pid"] == 111 and report["trainerProcess"]["pid"] == 222
    assert report["heartbeat"]["identityMatched"] and report["wallCapMet"]
    assert active_watchdogs[0].deadline == 59
    assert report["stoppedAt"] and report["startedAt"] and report["close"]["requestedAt"]
    if graceful:
        assert report["completed"] and report["close"]["ackMatched"]
        assert 30 <= report["elapsedSeconds"] < 31 and not report["forcedStop"]
    else:
        assert not report["completed"] and not report["close"]["ackMatched"]
        assert 59 <= report["elapsedSeconds"] < 60 and report["forcedStop"] and report["exitCode"] == 124
    assert json.loads((Path(args.output) / runner.RESULT_NAME).read_text())["completed"] == graceful
