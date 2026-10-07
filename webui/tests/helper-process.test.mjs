import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import test from "node:test";
import { activeHelperProcesses, closeHelperProcesses, runHelperProcess } from "../server/helper-process.mjs";
import { PATHS } from "../server/paths.mjs";

function alive(pid) { try { process.kill(pid, 0); return true; } catch (error) { if (error.code === "ESRCH") return false; throw error; } }
for (const cause of ["cancel", "timeout", "overflow"]) {
  test(`shared helper ${cause} confirms interpreter and launcher exit before settlement`, {
    skip: process.platform !== "win32" || !process.env.DFLSN_ISOLATED_TEST_ROOT, timeout: 20_000,
  }, async t => {
    const controller = new AbortController();
    let pids;
    const script = "import os,json,time\nprint(json.dumps([os.getpid(),os.getppid()]),flush=True)\ntime.sleep(.2)\n"
      + (cause === "overflow" ? "print('x'*4096,flush=True)\n" : "") + "time.sleep(60)";
    await assert.rejects(runHelperProcess(PATHS.python, ["-u", "-c", script], {
      signal: controller.signal, timeoutMs: cause === "timeout" ? 2_000 : 15_000,
      maxBytes: cause === "overflow" ? 512 : 4096,
      onStdout: chunk => { if (!pids) { pids = JSON.parse(chunk.toString()); assert.ok(pids.every(alive)); if (cause === "cancel") controller.abort(); } },
    }), error => error.code === ({ cancel: "OPERATION_CANCELLED", timeout: "HELPER_TIMEOUT", overflow: "HELPER_OUTPUT_TOO_LARGE" })[cause]);
    assert.ok(pids); assert.notEqual(pids[0], pids[1]); assert.equal(pids.some(alive), false);
    assert.equal(activeHelperProcesses().length, 0);
    t.diagnostic(JSON.stringify({ cause, pids, allExited: true }));
  });
}

test("helper does not return stdout before stream close or before stop verification", async () => {
  const controller = new AbortController();
  let entered, release;
  const ready = new Promise(done => { entered = done; });
  const confirmation = new Promise(done => { release = done; });
  let completed = false;
  const promise = runHelperProcess(process.execPath, ["-e", "process.stdout.write('ready'); setInterval(()=>{},1000)"], {
    signal: controller.signal, onStdout: () => entered(), spawnProcess: spawn,
    terminateProcessTree: async child => { child.kill(); await confirmation; }, timeoutMs: 20_000,
  }).catch(error => { completed = true; throw error; });
  await ready; controller.abort(); await new Promise(done => setTimeout(done, 50));
  assert.equal(completed, false); assert.equal(activeHelperProcesses()[0].state, "stopping");
  release(); await assert.rejects(promise, { code: "OPERATION_CANCELLED" });
  assert.equal(activeHelperProcesses().length, 0);
});

test("service helper shutdown waits for active helper cleanup", async () => {
  let ready;
  const entered = new Promise(done => { ready = done; });
  const task = runHelperProcess(process.execPath, ["-e", "console.log('ready');setInterval(()=>{},1000)"], {
    onStdout: () => ready(), terminateProcessTree: async child => { child.kill(); },
  });
  const cancelled = assert.rejects(task, { code: "OPERATION_CANCELLED" });
  await entered; await closeHelperProcesses(); await cancelled;
  assert.equal(activeHelperProcesses().length, 0);
});

test("unconfirmed helper termination remains reserved and rejects later mutations", async () => {
  // A distinct module instance contains this intentionally latched failure.
  const isolated = await import(`../server/helper-process.mjs?stop-failure=${Date.now()}`);
  const fake = new EventEmitter();
  fake.stdout = new EventEmitter(); fake.stderr = new EventEmitter();
  fake.stdin = new EventEmitter(); fake.stdin.end = () => {};
  fake.pid = null;
  const controller = new AbortController();
  const work = isolated.runHelperProcess("controlled-fixture", [], {
    spawnProcess: () => fake, signal: controller.signal,
    terminateProcessTree: async () => { throw new Error("controlled verification unavailable"); },
  });
  controller.abort();
  await assert.rejects(work, { code: "HELPER_STOP_UNCONFIRMED" });
  assert.equal(isolated.activeHelperProcesses()[0].state, "stop-unconfirmed");
  assert.throws(() => isolated.assertHelperStopsConfirmed(), { code: "HELPER_STOP_UNCONFIRMED" });
  await assert.rejects(isolated.closeHelperProcesses(), { code: "HELPER_STOP_UNCONFIRMED" });
});
