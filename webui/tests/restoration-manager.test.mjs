import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { RestorationManager } from "../server/restoration-manager.mjs";
import { PATHS } from "../server/paths.mjs";
import { terminateChildProcessTree } from "../server/process-tree.mjs";

// Real 1x1 RGB PNG fixture. Model inference belongs to the Python/real-run tests.
const IMAGE = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVQIHWP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC", "base64");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
async function fixture(callback, options = {}) {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-restoration-manager-"));
  const workspaceRoot = path.join(temporary, "workspace"), runtimeRoot = path.join(workspaceRoot, ".webui");
  await mkdir(path.join(workspaceRoot, "data_src"), { recursive: true });
  await writeFile(path.join(workspaceRoot, "data_src", "frame.png"), IMAGE);
  const manager = new RestorationManager({ workspaceRoot, runtimeRoot, assetsRoot: path.join(temporary, "assets"), ...options });
  await manager.initialize();
  try { await callback({ manager, workspaceRoot, runtimeRoot }); }
  finally { await manager.close(); await rm(temporary, { recursive: true, force: true }); }
}
async function completion(manager, id) {
  await manager.pending.get(id);
  return manager.getTask(id);
}
async function publish(requestFile) {
  const request = JSON.parse(await readFile(requestFile, "utf8"));
  const root = path.dirname(path.dirname(requestFile));
  const directory = path.join(root, "outputs", request.taskId);
  await mkdir(path.join(directory, "images"), { recursive: true });
  const outputs = request.inputs.map(input => ({ sourceName: input.name, name: `${path.parse(input.name).name}.png`,
    inputSha256: input.sha256, outputSha256: digest(IMAGE), width: 1, height: 1, mode: "RGB", metadataCopied: false }));
  for (const output of outputs) await writeFile(path.join(directory, "images", output.name), IMAGE);
  const report = { taskId: request.taskId, side: request.side, modelId: request.modelId, selectedCount: outputs.length,
    outputs, originalsPreserved: true, atomicBatch: true, metadataCopied: false };
  await writeFile(path.join(directory, "manifest.json"), JSON.stringify(report));
  return report;
}

test("source frame inventory excludes aligned and exposes bounded batch fingerprint", async () => {
  await fixture(async ({ manager, workspaceRoot }) => {
    await mkdir(path.join(workspaceRoot, "data_src", "aligned"));
    await writeFile(path.join(workspaceRoot, "data_src", "aligned", "face.png"), IMAGE);
    const inventory = await manager.listInputs({ side: "src" });
    assert.deepEqual(inventory.inputs.map(input => input.name), ["frame.png"]);
    assert.equal(inventory.inputs[0].sha256, digest(IMAGE));
    assert.deepEqual(inventory.range, { start: 1, end: 1 });
    assert.equal(inventory.limit, 500); assert.equal(inventory.defaultModelId, "swinir-psnr");
    assert.equal(inventory.models.find(model => model.id === "gfpgan-v1.4").sourceFramesSupported, false);
    await assert.rejects(manager.listInputs({ side: "src", limit: 501 }));
  });
});

test("source snapshots changing after review cannot launch inference", async () => {
  await fixture(async ({ manager, workspaceRoot }) => {
    const inventory = await manager.listInputs({ side: "src" });
    await writeFile(path.join(workspaceRoot, "data_src", "frame.png"), Buffer.concat([IMAGE, Buffer.from("changed")]));
    await assert.rejects(manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint }), error => error.code === "RESTORATION_SOURCE_CHANGED");
    assert.deepEqual(manager.activeTasks(), []);
  }, { processRunner: () => assert.fail("Changed sources must not reach inference") });
});

test("completed new copies require full publication verification and survive manager reload", async () => {
  await fixture(async ({ manager, workspaceRoot, runtimeRoot }) => {
    const inventory = await manager.listInputs({ side: "src" });
    const task = await manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint });
    const done = await completion(manager, task.taskId);
    assert.equal(done.status, "completed");
    assert.equal(done.outputs[0].inputSha256, digest(IMAGE));
    assert.equal(path.basename(await manager.resolveOutputDirectory(task.taskId)), "images");
    const image = await manager.resultImage(task.taskId, "frame.png");
    assert.deepEqual(image.buffer, IMAGE);
    assert.deepEqual(await readFile(path.join(workspaceRoot, "data_src", "frame.png")), IMAGE);
    const reloaded = new RestorationManager({ workspaceRoot, runtimeRoot });
    await reloaded.initialize();
    assert.equal(reloaded.getTask(task.taskId).status, "completed");
    await writeFile(image.path, Buffer.concat([IMAGE, Buffer.from("tamper")]));
    await assert.rejects(reloaded.resolveOutputDirectory(task.taskId), error => error.code === "RESTORATION_SOURCE_CHANGED");
    await reloaded.close();
  }, { processRunner: publish });
});

test("partial worker outputs are never exposed as completed copies", async () => {
  await fixture(async ({ manager }) => {
    const inventory = await manager.listInputs({ side: "src" });
    const task = await manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint });
    const done = await completion(manager, task.taskId);
    assert.equal(done.status, "failed"); assert.deepEqual(done.outputs, []);
    await assert.rejects(manager.resolveOutputDirectory(task.taskId), error => error.code === "RESTORATION_NOT_COMPLETE");
  }, { processRunner: async () => { throw new Error("inference failed midway"); } });
});

test("GPU/activity lock blocks concurrent tasks and cancellation waits for worker exit", async () => {
  const events = []; let workerStopped = false;
  await fixture(async ({ manager }) => {
    const inventory = await manager.listInputs({ side: "src" });
    const input = { side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint };
    const task = await manager.createTask(input);
    assert.equal(manager.activeTasks()[0].id, task.taskId);
    await assert.rejects(manager.createTask(input), error => error.code === "RESTORATION_BUSY");
    const cancelled = await manager.cancelTask(task.taskId);
    assert.equal(cancelled.status, "cancelled"); assert.equal(workerStopped, true);
    assert.deepEqual(manager.activeTasks(), []); assert.equal(events.at(-1), null);
  }, { onActiveChange: id => events.push(id), processRunner: (_file, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => { workerStopped = true; reject(new Error("stopped")); }, { once: true });
  }) });
});

test("external busy checks and GFPGAN stage mismatch reject before worker launch", async () => {
  await fixture(async ({ manager }) => {
    const inventory = await manager.listInputs({ side: "src" });
    const input = { side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint };
    await assert.rejects(manager.createTask({ ...input, modelId: "gfpgan-v1.4" }), error => error.code === "RESTORATION_STAGE_UNSUPPORTED");
    await assert.rejects(manager.createTask(input), /GPU already used/);
    assert.deepEqual(manager.activeTasks(), []);
  }, { assertCanStart: async () => { throw new Error("GPU already used"); }, processRunner: () => assert.fail("Busy worker must not start") });
});

test("real venv launcher cancellation holds activity until interpreter and tree termination complete", async () => {
  let release, reportStopped;
  const delayedConfirmation = new Promise(resolve => { release = resolve; });
  const treeStopped = new Promise(resolve => { reportStopped = resolve; });
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "dfl-restoration-child-"));
  await mkdir(path.join(projectRoot, "webui", "python"), { recursive: true });
  await writeFile(path.join(projectRoot, "webui", "python", "restoration_task.py"),
    "import os,sys,time\nfrom pathlib import Path\nr=Path(sys.argv[sys.argv.index('--request')+1])\nPath(str(r)+'.ready').write_text(str(os.getpid()))\ntime.sleep(60)\n");
  try {
    await fixture(async ({ manager }) => {
      try {
      const inventory = await manager.listInputs({ side: "src" });
      const task = await manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint });
      const marker = path.join(manager.root, "requests", `${task.taskId}.json.ready`);
      let pid;
      for (let i = 0; i < 100; i++) {
        pid = await readFile(marker, "utf8").then(Number).catch(error => { if (error.code !== "ENOENT") throw error; });
        if (pid) break;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      assert.ok(pid, "Actual interpreter must start before cancellation");
      const cancelled = manager.cancelTask(task.taskId);
      await treeStopped;
      assert.throws(() => process.kill(pid, 0), "The actual Python interpreter must be gone");
      assert.equal(manager.activeTasks()[0].id, task.taskId, "Activity stays reserved while tree confirmation is pending");
      assert.equal(manager.getTask(task.taskId).status, "running");
      await assert.rejects(manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint }), error => error.code === "RESTORATION_BUSY");
      release();
      assert.equal((await cancelled).status, "cancelled");
      assert.deepEqual(manager.activeTasks(), []);
      } finally { release(); }
    }, { projectRoot, pythonPath: PATHS.python, terminateProcessTree: async child => {
      try { await terminateChildProcessTree(child); } finally { reportStopped(); }
      await delayedConfirmation;
    } });
  } finally { release(); await rm(projectRoot, { recursive: true, force: true }); }
});

test("unconfirmed tree termination is failed, retains activity and refuses a safe-close claim", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "dfl-restoration-stop-failure-"));
  await mkdir(path.join(projectRoot, "webui", "python"), { recursive: true });
  await writeFile(path.join(projectRoot, "webui", "python", "restoration_task.py"),
    "import os,sys,time\nfrom pathlib import Path\nr=Path(sys.argv[sys.argv.index('--request')+1])\nPath(str(r)+'.ready').write_text(str(os.getpid()))\ntime.sleep(60)\n");
  try {
    await fixture(async ({ manager }) => {
      const inventory = await manager.listInputs({ side: "src" });
      const task = await manager.createTask({ side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint });
      let pid;
      for (let i = 0; i < 100; i++) {
        pid = await readFile(path.join(manager.root, "requests", `${task.taskId}.json.ready`), "utf8").then(Number).catch(error => { if (error.code !== "ENOENT") throw error; });
        if (pid) break;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      try {
        assert.ok(pid);
        assert.equal((await manager.cancelTask(task.taskId)).status, "failed");
        assert.match(manager.getTask(task.taskId).error, /simulated unconfirmed tree/);
        assert.equal(manager.activeTasks()[0].id, task.taskId);
        await assert.rejects(manager.close(), error => error.code === "RESTORATION_STOP_UNCONFIRMED");
        assert.throws(() => process.kill(pid, 0), "This test actually terminates its process before injecting the confirmation failure");
      } finally {
        // Only undo the artificial failure after this test's real terminator finished.
        manager.stopUnconfirmed = null; manager.activeId = null;
      }
    }, { projectRoot, pythonPath: PATHS.python, terminateProcessTree: async child => {
      await terminateChildProcessTree(child); throw new Error("simulated unconfirmed tree");
    } });
  } finally { await rm(projectRoot, { recursive: true, force: true }); }
});

for (const cause of ["termination rejects", "stream never closes"]) {
  test(`restoration ${cause} settles as failed without releasing activity`, { timeout: 3000 }, async () => {
    const isolated = await import(`../server/helper-process.mjs?restoration-stop=${encodeURIComponent(cause)}-${Date.now()}`);
    const child = new EventEmitter();
    child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
    child.stdin = new EventEmitter(); child.stdin.end = () => {};
    const events = [];
    await fixture(async ({ manager }) => {
      const inventory = await manager.listInputs({ side: "src" });
      const input = { side: "src", names: ["frame.png"], fingerprint: inventory.fingerprint };
      const task = await manager.createTask(input);
      const retained = path.join(manager.root, "staging", task.taskId);
      await mkdir(retained);
      await writeFile(path.join(retained, "partial.txt"), "worker-owned partial output");
      let cancel;
      try {
        cancel = manager.cancelTask(task.taskId);
        const result = await Promise.race([cancel, new Promise(resolve => setTimeout(() => resolve("not-settled"), 500))]);
        assert.notEqual(result, "not-settled", "A failed stop must settle even without a child close event");
        assert.equal(result.status, "failed");
        assert.equal(manager.stopUnconfirmed.code, "RESTORATION_STOP_UNCONFIRMED");
        assert.equal(manager.activeTasks()[0].id, task.taskId);
        assert.equal(events.at(-1), task.taskId);
        assert.equal(await readFile(path.join(retained, "partial.txt"), "utf8"), "worker-owned partial output");
        await assert.rejects(manager.createTask(input), { code: "RESTORATION_BUSY" });
        await assert.rejects(manager.close(), { code: "RESTORATION_STOP_UNCONFIRMED" });
        assert.throws(() => isolated.assertHelperStopsConfirmed(), { code: "HELPER_STOP_UNCONFIRMED" });
        child.emit("close", 1);
        assert.equal(manager.activeTasks()[0].id, task.taskId, "A late close must not silently clear an unconfirmed stop");
      } finally {
        child.emit("close", 1);
        await cancel;
        // Release only this simulated fixture's latched reservation for cleanup.
        manager.stopUnconfirmed = null; manager.activeId = null;
      }
    }, { onActiveChange: id => events.push(id),
      runProcess: (executable, args, options) => isolated.runHelperProcess(executable, args, { ...options, spawnProcess: () => child, closeWaitMs: 20 }),
      terminateProcessTree: async () => { if (cause === "termination rejects") throw new Error("controlled taskkill access denied"); },
    });
  });
}
