import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { JobManager } from "../server/job-manager.mjs";
import { createMeDetachedRunner, createMeSupervisorToken, MeDetachedRunner } from "../server/me-detached-runner.mjs";
import { PATHS, jobDirectory } from "../server/paths.mjs";

async function until(check, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Timed out waiting for ME supervisor state");
}

test("ME log replay reports a distinct durable byte offset for each output record", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-log-offset-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const records = ["第一行", "second"].map((data) => `${JSON.stringify({ data })}\n`);
  await writeFile(path.join(directory, "me-output.ndjson"), records.join(""), "utf8");
  await writeFile(path.join(directory, "me-runner-state.json"), JSON.stringify({
    token, trainerPid: 123, supervisorPid: 456, status: "running", updatedAt: new Date().toISOString(),
  }));
  const runner = new MeDetachedRunner({ directory, token });
  const received = [];
  runner.onData((data, offset) => received.push({ data, offset }));
  runner.onExit(() => {});
  await until(() => received.length === 2);
  assert.deepEqual(received, [
    { data: "第一行", offset: Buffer.byteLength(records[0]) },
    { data: "second", offset: Buffer.byteLength(records.join("")) },
  ]);
  runner.dispose();
});

test("ME runner observes a supervisor that advances from starting to running after attach", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-starting-reconnect-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const stateFile = path.join(directory, "me-runner-state.json");
  await writeFile(stateFile, JSON.stringify({
    token, status: "starting", trainerPid: null, supervisorPid: 456,
    updatedAt: new Date().toISOString(),
  }));
  const runner = await createMeDetachedRunner({ directory, token, attach: true });
  let observed = null;
  runner.onState((state) => { observed = state; });
  runner.onData(() => {});
  runner.onExit(() => {});
  await writeFile(stateFile, JSON.stringify({
    token, status: "running", trainerPid: 123, supervisorPid: 456,
    updatedAt: new Date().toISOString(),
  }));
  await until(() => observed?.status === "running");
  assert.equal(runner.pid, 123);
  runner.dispose();
});

test("ME runner reports a stale orphan only after supervisor and trainer both exit", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-sidecar-orphan-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const stale = new Date(Date.now() - 60_000).toISOString();
  const state = {
    token, status: "running", updatedAt: stale,
    supervisorPid: 101, launcherPid: 202, trainerPid: 303,
  };
  await writeFile(path.join(directory, "me-runner-state.json"), JSON.stringify(state));
  const alive = new Set([303]);
  const runner = new MeDetachedRunner({
    directory, token, initialState: state, pollMs: 10,
    processAliveImpl: (pid) => alive.has(pid),
  });
  const exits = [];
  runner.onData(() => {});
  runner.onExit((result) => exits.push(result));
  await runner.poll();
  assert.equal(exits.length, 0, "a live trainer must prevent orphan recovery");
  alive.clear();
  await until(() => exits.length === 1);
  assert.deepEqual(exits, [{ exitCode: 1, signal: null }]);
  await runner.poll();
  assert.equal(exits.length, 1, "orphan exit must be reported exactly once");
});

test("ME supervisor reports a missing fixed executable as a launch failure", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-sidecar-start-fail-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  await assert.rejects(createMeDetachedRunner({
    directory, token,
    executable: path.join(directory, "missing-python.exe"),
    args: [], cwd: directory, env: process.env,
  }), /ENOENT|spawn|not found/i);
  const state = JSON.parse(await readFile(path.join(directory, "me-runner-state.json"), "utf8"));
  assert.equal(state.status, "failed");
  assert.equal(state.token, token, "terminal failure must retain its job identity");
});

test("Web restart recovers only a verified tokenless legacy supervisor failure", {
  skip: !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async (t) => {
  const now = new Date();
  const requestedAt = new Date(now.getTime() - 2000).toISOString();
  const cases = [
    { name: "verified", launchMatches: true, trainerPid: 99999991, expected: "cancelled" },
    { name: "foreign-token", launchMatches: false, trainerPid: 99999992, expected: "orphaned" },
    { name: "live-trainer", launchMatches: true, trainerPid: process.pid, expected: "orphaned" },
  ];
  for (const entry of cases) {
    const id = `me-legacy-failure-${entry.name}-${Date.now()}`;
    const directory = jobDirectory(id);
    await mkdir(directory, { recursive: true });
    t.after(() => rm(directory, { recursive: true, force: true }));
    const token = createMeSupervisorToken();
    await writeFile(path.join(directory, "me-launch.json"), JSON.stringify({
      token: entry.launchMatches ? token : createMeSupervisorToken(),
    }));
    await writeFile(path.join(directory, "me-runner-state.json"), JSON.stringify({
      status: "failed", supervisorPid: 99999993, trainerPid: null,
      exitCode: 1, endedAt: now.toISOString(), updatedAt: now.toISOString(),
      error: "EPERM: state rename failed",
    }));
    await writeFile(path.join(directory, "metadata.json"), JSON.stringify({
      id, commandId: "train.me", label: "ME", shortLabel: "ME", profile: "pytorch",
      category: "training", launchMode: "guided", parameters: {}, controls: ["close"],
      locks: [`lock:${id}`], state: "stopping", pid: entry.trainerPid,
      exitCode: null, signal: null, sequence: 0, createdAt: requestedAt,
      startedAt: requestedAt, endedAt: null, stopReason: "safe-stop-timeout",
      stopRequestedAt: requestedAt, commandLine: "test fixture", error: null,
      latestPrompt: null, latestMetric: null, latestProgress: null,
      previewVersion: null, latestEvaluationSnapshotId: null, evaluation: null,
      health: null, meSupervisorToken: token, runnerOutputOffset: 0,
    }));
    entry.id = id;
  }
  const manager = new JobManager({ trainingEvaluationManager: { initialize: async () => {} } });
  await manager.initialize();
  for (const entry of cases) {
    const job = manager.get(entry.id);
    assert.equal(job.state, entry.expected, entry.name);
    if (entry.expected === "cancelled") {
      assert.equal(job.exitCode, 1);
      assert.equal(job.stopReason, "safe-stop-timeout");
      assert.match(job.error, /监督状态写入失败/);
    }
    assert.equal(manager.locks.has(`lock:${entry.id}`), false);
  }
  await manager.flushAll();
});

test("ME supervisor survives Web runner disposal and reconnects to remaining output and exit", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-sidecar-reconnect-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const first = await createMeDetachedRunner({
    directory, token, executable: process.execPath,
    args: ["-e", "console.log('first'); setTimeout(() => { console.log('after-reconnect'); process.exit(0); }, 1000)"],
    cwd: directory, env: process.env,
  });
  let firstOutput = "";
  first.onData((data) => { firstOutput += data; });
  first.onExit(() => {});
  await until(() => firstOutput.includes("first"));
  const offset = first.outputOffset;
  first.dispose();
  const second = await createMeDetachedRunner({ directory, token, outputOffset: offset, attach: true });
  let resumedOutput = "";
  let finalExit;
  second.onData((data) => { resumedOutput += data; });
  second.onExit((result) => { finalExit = result; });
  await until(() => finalExit);
  assert.match(resumedOutput, /after-reconnect/);
  assert.doesNotMatch(resumedOutput, /first/);
  assert.equal(finalExit.exitCode, 0);
  const state = JSON.parse(await readFile(path.join(directory, "me-runner-state.json"), "utf8"));
  assert.equal(state.token, token);
  assert.equal(state.status, "exited");
});

test("ME supervisor accepts force-kill only with its persisted token", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-sidecar-kill-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const runner = await createMeDetachedRunner({
    directory, token, executable: process.execPath,
    args: ["-e", "console.log('running'); setInterval(() => {}, 1000)"],
    cwd: directory, env: process.env,
  });
  let exit;
  runner.onData(() => {});
  runner.onExit((result) => { exit = result; });
  await runner.kill();
  await until(() => exit);
  assert.notEqual(exit.exitCode, 0);
});

test("force-kill reaches the real trainer after a Windows launcher has exited", {
  skip: process.platform !== "win32",
}, async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-launcher-force-kill-"));
  let actualPid = null;
  t.after(async () => {
    if (actualPid) {
      try { process.kill(actualPid); } catch {}
    }
    await rm(directory, { recursive: true, force: true });
  });
  const heartbeatFile = path.join(directory, "trainer-heartbeat.json");
  const trainerSource = `
    const fs = require('fs');
    const now = new Date().toISOString();
    fs.writeFileSync(process.env.DFL_WEB_HEARTBEAT_FILE, JSON.stringify({
      pid: process.pid, startedAt: now, at: now, phaseAt: now,
      progressAt: now, phase: 'training', iteration: 1,
    }));
    setInterval(() => {}, 1000);
  `;
  const launcherSource = `
    const { spawn } = require('child_process');
    spawn(process.execPath, ['-e', ${JSON.stringify(trainerSource)}], {
      stdio: 'inherit', env: process.env,
    });
    setTimeout(() => process.exit(0), 500);
  `;
  const runner = await createMeDetachedRunner({
    directory, token: createMeSupervisorToken(), executable: process.execPath,
    args: ["-e", launcherSource], cwd: directory,
    env: { ...process.env, DFL_WEB_HEARTBEAT_FILE: heartbeatFile },
  });
  let exit = null;
  runner.onData(() => {});
  runner.onExit((result) => { exit = result; });
  const state = await until(() => {
    const current = runner.state;
    return current?.trainerPidSource === "heartbeat" ? current : null;
  }).catch(async (error) => {
    const output = await readFile(path.join(directory, "me-output.ndjson"), "utf8").catch(() => "");
    const heartbeat = await readFile(heartbeatFile, "utf8").catch(() => "");
    throw new Error(`${error.message}; state=${JSON.stringify(runner.state)}; heartbeat=${heartbeat}; output=${output}`);
  });
  actualPid = state.trainerPid;
  assert.notEqual(state.launcherPid, actualPid);
  await new Promise((resolve) => setTimeout(resolve, 800));
  await runner.kill();
  await until(() => exit, 8000);
  await until(() => {
    try { process.kill(actualPid, 0); return false; }
    catch { return true; }
  }, 8000);
});

test("ME health reports a live heartbeat and stalled iterations without ending the job", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-heartbeat-health-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  let current = Date.now();
  const manager = new JobManager({ now: () => new Date(current), metadataWriter: async () => {} });
  const job = {
    id: "me-health-test", state: "running", directory,
    runner: { state: { updatedAt: new Date(current).toISOString() } },
    startedAt: new Date(current).toISOString(), latestMetric: { iterationTimeMs: 1000 },
    health: null, sequence: 0, events: [], writeChain: Promise.resolve(),
    metadataWriteChain: Promise.resolve(), currentEventBytes: 0, eventSegmentIndex: 0,
    eventsFile: path.join(directory, "events.ndjson"), metadataFile: path.join(directory, "metadata.json"),
  };
  const firstProgressAt = new Date(current).toISOString();
  await writeFile(path.join(directory, "trainer-heartbeat.json"), JSON.stringify({
    at: firstProgressAt, progressAt: firstProgressAt, phase: "training", iteration: 10,
  }));
  manager.startMeHealthWatcher(job);
  await until(() => job.health?.state === "healthy");
  current += 310_000;
  job.runner.state.updatedAt = new Date(current).toISOString();
  await writeFile(path.join(directory, "trainer-heartbeat.json"), JSON.stringify({
    at: new Date(current).toISOString(), progressAt: firstProgressAt, phase: "training", iteration: 10,
  }));
  manager.startMeHealthWatcher(job);
  await until(() => job.health?.state === "hung");
  assert.match(job.health.reason, /迭代/);
  assert.equal(job.state, "running");
  assert.equal(job.events.filter((event) => event.type === "job.warning").length, 1);
  current += 620_000;
  job.runner.state.updatedAt = new Date(current).toISOString();
  await writeFile(path.join(directory, "trainer-heartbeat.json"), JSON.stringify({
    at: new Date(current).toISOString(), progressAt: firstProgressAt,
    phaseAt: new Date(current - 620_000).toISOString(), phase: "backup", iteration: 10,
  }));
  manager.startMeHealthWatcher(job);
  await until(() => job.health?.phase === "backup" && job.health?.state === "hung");
  assert.match(job.health.reason, /backup.*10 分钟/);
  clearInterval(job.healthTimer);
  await job.writeChain;
});

test("ME health reports Trainer stalls even when the supervisor state is stale", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-stale-supervisor-health-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const current = Date.now();
  const manager = new JobManager({ now: () => new Date(current), metadataWriter: async () => {} });
  const job = {
    id: "me-stale-supervisor-health", state: "running", directory, pid: 123,
    runner: { state: { updatedAt: new Date(current - 40_000).toISOString() } },
    startedAt: new Date(current - 600_000).toISOString(),
    latestMetric: { iterationTimeMs: 1000 }, health: null,
    sequence: 0, events: [], writeChain: Promise.resolve(), metadataWriteChain: Promise.resolve(),
    currentEventBytes: 0, eventSegmentIndex: 0,
    eventsFile: path.join(directory, "events.ndjson"), metadataFile: path.join(directory, "metadata.json"),
  };
  const heartbeatFile = path.join(directory, "trainer-heartbeat.json");
  const check = async (value, expectedState, expectedReason) => {
    await writeFile(heartbeatFile, JSON.stringify({ pid: 123, phase: "training", iteration: 10, ...value }));
    manager.startMeHealthWatcher(job);
    await until(() => job.health?.state === expectedState && expectedReason.test(job.health.reason ?? ""));
  };
  t.after(() => clearInterval(job.healthTimer));
  await check({ at: new Date(current).toISOString(), progressAt: new Date(current).toISOString(),
    phaseAt: new Date(current).toISOString() }, "unknown", /监督进程/);
  await check({ at: new Date(current - 31_000).toISOString(),
    progressAt: new Date(current - 31_000).toISOString() }, "hung", /心跳超过 30 秒/);
  await check({ at: new Date(current).toISOString(), progressAt: new Date(current - 310_000).toISOString(),
    phaseAt: new Date(current - 310_000).toISOString() }, "hung", /迭代/);
  await check({ at: new Date(current).toISOString(), progressAt: new Date(current - 610_000).toISOString(),
    phaseAt: new Date(current - 610_000).toISOString(), phase: "backup" }, "hung", /backup.*10 分钟/);
  job.runner.state = {
    updatedAt: new Date(current).toISOString(), trainerPidSource: "heartbeat", trainerPid: 123,
  };
  await writeFile(heartbeatFile, JSON.stringify({ pid: 999, phase: "training", iteration: 10,
    at: new Date(current).toISOString(), progressAt: new Date(current).toISOString(),
    phaseAt: new Date(current).toISOString() }));
  manager.startMeHealthWatcher(job);
  await until(() => job.health?.state === "unknown" && /PID 不一致/.test(job.health.reason));
  assert.equal(job.pid, 123, "a mismatched heartbeat must not replace the supervisor's Trainer PID");
  assert.equal(job.state, "running", "health warnings must not end training");
  await job.writeChain;
});

test("terminal ME jobs expose the last heartbeat as history without hiding stop failure evidence", async () => {
  const observation = {
    state: "hung", heartbeatAt: "2026-10-02T16:33:35.380Z",
    progressAt: "2026-10-02T16:33:34.246Z", phase: "finished",
    phaseSince: "2026-10-02T16:33:35.380Z", iteration: 99214,
    reason: "训练心跳超过 30 秒未更新",
  };
  const written = [];
  const manager = new JobManager({
    metadataWriter: async (_file, metadata) => written.push(metadata),
  });
  const job = {
    id: "me-terminal-health", commandId: "train.me", state: "running",
    createdAt: "2026-10-02T13:52:24.605Z", health: observation,
    stopReason: null, exitCode: null, error: null,
    metadataFile: "fixture-metadata.json", metadataWriteChain: Promise.resolve(),
  };
  manager.jobs.set(job.id, job);
  assert.equal(manager.get(job.id).health.state, "hung", "a live stall must remain visible");

  job.state = "cancelled";
  job.stopReason = "safe-stop-timeout";
  job.exitCode = 1;
  job.error = "ME 监督状态写入失败：EPERM";
  for (const state of ["cancelled", "failed", "succeeded", "orphaned"]) {
    job.state = state;
    const visible = manager.get(job.id);
    assert.equal(visible.health.state, "inactive", state);
    assert.equal(visible.health.reason, null, state);
    assert.equal(visible.health.phase, "finished", state);
    assert.equal(visible.health.heartbeatAt, observation.heartbeatAt, state);
    assert.deepEqual(visible.health.lastObservation, observation, state);
    assert.deepEqual(manager.list()[0], visible, `${state} is also projected in /api/jobs`);
  }
  job.state = "cancelled";
  const visible = manager.get(job.id);
  assert.equal(visible.stopReason, "safe-stop-timeout");
  assert.equal(visible.exitCode, 1);
  assert.match(visible.error, /监督状态写入失败/);
  await manager.persist(job, { force: true });
  assert.deepEqual(written[0].health, observation, "metadata retains the original health evidence");
  assert.equal(written[0].stopReason, "safe-stop-timeout");
  assert.strictEqual(job.health, observation, "API projection does not modify the in-memory record");
});

test("JobManager reattaches ME training and retains GPU/model locks after service restart", {
  skip: !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async (t) => {
  const id = `me-reconnect-${Date.now()}`;
  const directory = jobDirectory(id);
  await mkdir(directory, { recursive: true });
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const first = await createMeDetachedRunner({
    directory, token, executable: process.execPath,
    args: ["-e", "console.log('trainer-running'); setInterval(() => {}, 1000)"],
    cwd: directory, env: process.env,
  });
  first.dispose();
  const now = new Date().toISOString();
  await writeFile(path.join(directory, "metadata.json"), `${JSON.stringify({
    id, commandId: "train.me", label: "ME", shortLabel: "ME", profile: "pytorch",
    category: "training", launchMode: "guided", parameters: {}, controls: ["save", "close"],
    locks: ["workspace:model", "gpu"], state: "starting", pid: first.pid,
    exitCode: null, signal: null, sequence: 0, createdAt: now, startedAt: now, endedAt: null,
    stopReason: null, stopRequestedAt: null, commandLine: "test fixture", error: null,
    latestPrompt: null, latestMetric: null, latestProgress: null,
    previewVersion: null, latestEvaluationSnapshotId: null, evaluation: null,
    health: null, meSupervisorToken: token, runnerOutputOffset: 0,
  })}\n`, "utf8");
  const manager = new JobManager({ trainingEvaluationManager: { initialize: async () => {} } });
  await manager.initialize();
  assert.equal(manager.get(id).state, "running");
  assert.equal(manager.locks.get("gpu"), id);
  assert.equal(manager.locks.get("workspace:model"), id);
  await manager.control(id, "save");
  assert.match(await readFile(path.join(directory, "control.jsonl"), "utf8"), /"operation":"save"/);
  await manager.control(id, "force-kill");
  await until(() => manager.get(id).state === "cancelled");
  assert.equal(manager.get(id).stopReason, "force-kill");
  assert.equal(manager.locks.has("gpu"), false);
  assert.equal(manager.locks.has("workspace:model"), false);
  await manager.flushAll();
  assert.equal(path.resolve(PATHS.jobsRoot), path.resolve(path.dirname(directory)));
});

test("Windows-style Python launcher PID resolves to the actual heartbeat PID in JobManager", {
  skip: !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async (t) => {
  const id = `me-launcher-pid-${Date.now()}`;
  const directory = jobDirectory(id);
  await mkdir(directory, { recursive: true });
  t.after(async () => {
    await until(async () => {
      try {
        const state = JSON.parse(await readFile(path.join(directory, "me-runner-state.json"), "utf8"));
        return ["exited", "failed"].includes(state.status);
      } catch { return false; }
    }, 8000).catch(() => {});
    await rm(directory, { recursive: true, force: true });
  });
  const heartbeatFile = path.join(directory, "trainer-heartbeat.json");
  const trainerSource = `
    const fs = require('fs');
    const now = new Date().toISOString();
    fs.writeFileSync(process.env.DFL_WEB_HEARTBEAT_FILE, JSON.stringify({
      pid: process.pid, startedAt: now, at: now, phaseAt: now,
      progressAt: now, phase: 'training', iteration: 1,
    }));
    console.log('[#000001][10ms][0.1][0.1]');
    setTimeout(() => process.exit(0), 4000);
  `;
  const launcherSource = `
    const { spawn } = require('child_process');
    const child = spawn(process.execPath, ['-e', ${JSON.stringify(trainerSource)}], {
      stdio: 'inherit', env: process.env,
    });
    child.on('close', (code) => process.exit(code ?? 1));
  `;
  const token = createMeSupervisorToken();
  const first = await createMeDetachedRunner({
    directory, token, executable: process.execPath, args: ["-e", launcherSource],
    cwd: directory, env: { ...process.env, DFL_WEB_HEARTBEAT_FILE: heartbeatFile },
  });
  first.dispose();
  const now = new Date().toISOString();
  await writeFile(path.join(directory, "metadata.json"), `${JSON.stringify({
    id, commandId: "train.me", label: "ME", shortLabel: "ME", profile: "pytorch",
    category: "training", launchMode: "guided", parameters: {}, controls: ["close"],
    locks: ["workspace:model", "gpu"], state: "running", pid: first.pid,
    exitCode: null, signal: null, sequence: 0, createdAt: now, startedAt: now, endedAt: null,
    stopReason: null, stopRequestedAt: null, commandLine: "test launcher", error: null,
    latestPrompt: null, latestMetric: null, latestProgress: null,
    previewVersion: null, latestEvaluationSnapshotId: null, evaluation: null,
    health: null, meSupervisorToken: token, runnerOutputOffset: 0,
  })}\n`, "utf8");
  const manager = new JobManager({ trainingEvaluationManager: { initialize: async () => {} } });
  await manager.initialize();
  const heartbeat = await until(async () => {
    try { return JSON.parse(await readFile(heartbeatFile, "utf8")); }
    catch { return null; }
  });
  await until(() => manager.get(id).pid === heartbeat.pid && manager.get(id).health?.state === "healthy");
  const state = await until(() => {
    const current = manager.getInternal(id).runner?.state;
    return current?.trainerPidSource === "heartbeat" ? current : null;
  });
  assert.notEqual(state.launcherPid, heartbeat.pid);
  assert.equal(state.trainerPid, heartbeat.pid);
  assert.equal(manager.get(id).pid, heartbeat.pid);
  await until(() => ["succeeded", "failed"].includes(manager.get(id).state), 8000);
  await manager.flushAll();
});

test("ME safe-stop acknowledgement survives a Web service restart without changing request identity", {
  skip: !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async (t) => {
  const id = `me-safe-reconnect-${Date.now()}`;
  const directory = jobDirectory(id);
  await mkdir(directory, { recursive: true });
  t.after(() => rm(directory, { recursive: true, force: true }));
  const token = createMeSupervisorToken();
  const controlFile = path.join(directory, "control.jsonl");
  const ackFile = path.join(directory, "control-ack.json");
  const childSource = `
    const fs = require('fs');
    const control = ${JSON.stringify(controlFile)};
    const ack = ${JSON.stringify(ackFile)};
    let closing = false;
    setInterval(() => {
      if (closing || !fs.existsSync(control)) return;
      const request = fs.readFileSync(control, 'utf8').split('\\n').filter(Boolean)
        .map(line => JSON.parse(line)).find(value => value.operation === 'close');
      if (!request) return;
      closing = true;
      setTimeout(() => {
        fs.writeFileSync(ack, JSON.stringify({ operation: 'close', status: 'completed',
          requestedAt: request.requestedAt, checkpoint: 'me.pt' }));
        process.exit(0);
      }, 900);
    }, 50);
  `;
  const first = await createMeDetachedRunner({
    directory, token, executable: process.execPath,
    args: ["-e", childSource], cwd: directory, env: process.env,
  });
  first.dispose();
  const now = new Date().toISOString();
  await writeFile(path.join(directory, "metadata.json"), `${JSON.stringify({
    id, commandId: "train.me", label: "ME", shortLabel: "ME", profile: "pytorch",
    category: "training", launchMode: "guided", parameters: {}, controls: ["save", "close"],
    locks: ["workspace:model", "gpu"], state: "running", pid: first.pid,
    exitCode: null, signal: null, sequence: 0, createdAt: now, startedAt: now, endedAt: null,
    stopReason: null, stopRequestedAt: null, commandLine: "test fixture", error: null,
    latestPrompt: null, latestMetric: null, latestProgress: null,
    previewVersion: null, latestEvaluationSnapshotId: null, evaluation: null,
    health: null, meSupervisorToken: token, runnerOutputOffset: 0,
  })}\n`, "utf8");
  const trainingEvaluationManager = { initialize: async () => {} };
  const before = new JobManager({ trainingEvaluationManager });
  await before.initialize();
  const closing = await before.control(id, "close");
  const requestedAt = closing.stopRequestedAt;
  assert.equal(closing.stopReason, "safe-stop");
  before.getInternal(id).runner.dispose();
  clearInterval(before.getInternal(id).previewTimer);
  clearInterval(before.getInternal(id).healthTimer);
  clearTimeout(before.getInternal(id).stopTimer);
  await before.flushAll();

  const after = new JobManager({ trainingEvaluationManager });
  await after.initialize();
  assert.equal(after.get(id).stopRequestedAt, requestedAt);
  if (after.get(id).state === "stopping") {
    assert.equal((await after.control(id, "close")).stopRequestedAt, requestedAt);
  }
  await until(() => after.get(id).state === "cancelled");
  assert.equal(after.get(id).stopReason, "safe-stop");
  assert.equal(after.get(id).exitCode, 0);
  assert.equal(after.locks.has("gpu"), false);
  await after.flushAll();
});

test("Windows state-file rename denial survives Web restart and recovers before an acknowledged ME close", {
  skip: process.platform !== "win32" || !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async (t) => {
  const id = `me-eperm-reconnect-${Date.now()}`;
  const directory = jobDirectory(id);
  assert.equal(path.relative(PATHS.jobsRoot, directory), id, "fixture must stay in the isolated jobs root");
  await mkdir(directory, { recursive: true });
  const stateFile = path.join(directory, "me-runner-state.json");
  const heartbeatFile = path.join(directory, "trainer-heartbeat.json");
  const controlFile = path.join(directory, "control.jsonl");
  const ackFile = path.join(directory, "control-ack.json");
  const startsFile = path.join(directory, "trainer-starts.txt");
  const trainerFile = path.join(directory, "fake-trainer.cjs");
  const token = createMeSupervisorToken();
  const managers = [];
  let runner = null;
  let lock = null;
  let supervisorPid = null;
  let trainerPid = null;
  t.after(async () => {
    const hasExited = (pid) => {
      try { process.kill(pid, 0); return false; }
      catch (error) { return error.code === "ESRCH"; }
    };
    if (lock) {
      lock.stdin.end("\n");
      await until(() => lock.exitCode !== null || lock.signalCode !== null, 5000)
        .catch(async () => {
          lock.kill();
          await until(() => lock.exitCode !== null || lock.signalCode !== null, 5000).catch(() => {});
        });
    }
    for (const manager of managers) {
      const job = manager.jobs.get(id);
      if (!job) continue;
      job.runner?.dispose();
      clearInterval(job.previewTimer);
      clearInterval(job.healthTimer);
      clearTimeout(job.stopTimer);
      await manager.flushAll().catch(() => {});
    }
    runner?.dispose();
    for (const pid of [trainerPid, supervisorPid]) {
      if (!pid || hasExited(pid)) continue;
      try { process.kill(pid); } catch {}
      await until(() => hasExited(pid), 5000).catch(() => {});
    }
    await rm(directory, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
  });

  await writeFile(trainerFile, `
    const fs = require('node:fs');
    const startedAt = new Date().toISOString();
    fs.appendFileSync(${JSON.stringify(startsFile)}, process.pid + '\\n');
    let closing = false;
    setInterval(() => {
      const at = new Date().toISOString();
      fs.writeFileSync(${JSON.stringify(heartbeatFile)}, JSON.stringify({
        pid: process.pid, startedAt, at, progressAt: at, phaseAt: startedAt,
        phase: 'training', iteration: 1,
      }));
      if (closing || !fs.existsSync(${JSON.stringify(controlFile)})) return;
      const request = fs.readFileSync(${JSON.stringify(controlFile)}, 'utf8').split('\\n')
        .filter(Boolean).map((line) => JSON.parse(line))
        .find((entry) => entry.operation === 'close');
      if (!request) return;
      closing = true;
      fs.writeFileSync(${JSON.stringify(ackFile)}, JSON.stringify({
        operation: 'close', status: 'completed', requestedAt: request.requestedAt,
        checkpoint: 'fixture-checkpoint.bin',
      }));
      setTimeout(() => process.exit(0), 100);
    }, 100);
  `, "utf8");
  runner = await createMeDetachedRunner({
    directory, token, executable: process.execPath, args: [trainerFile], cwd: directory,
    env: { ...process.env, DFL_WEB_HEARTBEAT_FILE: heartbeatFile },
  });
  supervisorPid = runner.supervisorPid;
  trainerPid = await until(async () => {
    try { return JSON.parse(await readFile(heartbeatFile, "utf8")).pid; }
    catch { return null; }
  });
  runner.dispose();
  const now = new Date().toISOString();
  await writeFile(path.join(directory, "metadata.json"), `${JSON.stringify({
    id, commandId: "train.me", label: "ME", shortLabel: "ME", profile: "pytorch",
    category: "training", launchMode: "guided", parameters: {}, controls: ["close"],
    locks: ["workspace:model", "gpu"], state: "running", pid: trainerPid,
    exitCode: null, signal: null, sequence: 0, createdAt: now, startedAt: now, endedAt: null,
    stopReason: null, stopRequestedAt: null, commandLine: "fake trainer fixture", error: null,
    latestPrompt: null, latestMetric: null, latestProgress: null,
    previewVersion: null, latestEvaluationSnapshotId: null, evaluation: null,
    health: null, meSupervisorToken: token, runnerOutputOffset: 0,
  })}\n`, "utf8");
  const trainingEvaluationManager = { initialize: async () => {} };
  const before = new JobManager({ trainingEvaluationManager });
  managers.push(before);
  await before.initialize();
  assert.equal(before.get(id).pid, trainerPid);
  assert.equal(before.get(id).runtimeIndependent, true);

  const lockSource = `
    $ErrorActionPreference = 'Stop'
    $handle = [System.IO.File]::Open(
      $env:ME_STATE_LOCK_FILE,
      [System.IO.FileMode]::Open,
      [System.IO.FileAccess]::Read,
      [System.IO.FileShare]::ReadWrite)
    try {
      [Console]::Out.WriteLine('LOCKED')
      [Console]::In.ReadLine() | Out-Null
    } finally { $handle.Dispose() }
  `;
  lock = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand",
    Buffer.from(lockSource, "utf16le").toString("base64")], {
    env: { ...process.env, ME_STATE_LOCK_FILE: stateFile },
    stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
  });
  let lockOutput = "";
  let lockError = "";
  lock.stdin.on("error", () => {});
  lock.stderr.on("data", (chunk) => { lockError += chunk; });
  lock.stdout.on("data", (chunk) => { lockOutput += chunk; });
  await until(() => {
    if (lock.exitCode !== null || lock.signalCode !== null) {
      throw new Error(`state lock exited: ${lockError}`);
    }
    return lockOutput.includes("LOCKED");
  });
  const lockedState = JSON.parse(await readFile(stateFile, "utf8"));
  const probe = `${stateFile}.probe.tmp`;
  await writeFile(probe, "{}\n", "utf8");
  await assert.rejects(rename(probe, stateFile), (error) =>
    ["EPERM", "EACCES", "EBUSY"].includes(error.code), "Windows must deny replacement while locked");

  const beforeJob = before.getInternal(id);
  beforeJob.runner.dispose();
  clearInterval(beforeJob.previewTimer);
  clearInterval(beforeJob.healthTimer);
  await before.flushAll();
  const after = new JobManager({ trainingEvaluationManager });
  managers.push(after);
  await after.initialize();
  assert.equal(after.get(id).state, "running");
  assert.equal(after.get(id).pid, trainerPid);
  assert.equal(after.getInternal(id).runner.supervisorPid, supervisorPid);
  assert.equal(after.locks.get("gpu"), id);
  assert.deepEqual((await readFile(startsFile, "utf8")).trim().split(/\r?\n/), [String(trainerPid)]);

  await until(async () => {
    const output = await readFile(path.join(directory, "me-output.ndjson"), "utf8").catch(() => "");
    return /无法(?:保存 ME 监督状态|读取 ME Trainer 身份).*?(?:EPERM|EACCES|EBUSY)/.test(output);
  }, 20_000);
  assert.equal(JSON.parse(await readFile(stateFile, "utf8")).updatedAt, lockedState.updatedAt,
    "denied state writes must leave the last valid supervisor record available");
  lock.stdin.end("\n");
  await until(() => lock.exitCode !== null || lock.signalCode !== null);
  lock = null;
  await until(async () => {
    const state = JSON.parse(await readFile(stateFile, "utf8"));
    return state.token === token && state.status === "running"
      && Date.parse(state.updatedAt) > Date.parse(lockedState.updatedAt);
  }, 10_000);

  const closing = await after.control(id, "close");
  assert.equal(closing.stopReason, "safe-stop");
  await until(() => after.get(id).state === "cancelled", 10_000);
  const acknowledgement = JSON.parse(await readFile(ackFile, "utf8"));
  assert.equal(acknowledgement.requestedAt, closing.stopRequestedAt);
  assert.equal(acknowledgement.status, "completed");
  const terminal = after.get(id);
  assert.equal(terminal.pid, trainerPid);
  assert.equal(terminal.stopReason, "safe-stop");
  assert.equal(terminal.exitCode, 0);
  assert.equal(after.locks.has("gpu"), false);
  await after.flushAll();
  const supervisorState = JSON.parse(await readFile(stateFile, "utf8"));
  assert.equal(supervisorState.token, token);
  assert.equal(supervisorState.status, "exited");
  assert.equal(supervisorState.exitCode, 0);
  const durableJob = JSON.parse(await readFile(path.join(directory, "metadata.json"), "utf8"));
  assert.equal(durableJob.state, "cancelled");
  assert.equal(durableJob.stopReason, "safe-stop");
  assert.equal(durableJob.stopRequestedAt, closing.stopRequestedAt);
  assert.equal(durableJob.meSupervisorToken, token);
  assert.deepEqual((await readFile(startsFile, "utf8")).trim().split(/\r?\n/), [String(trainerPid)]);
});
