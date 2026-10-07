import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { RuntimeServer } from "../server/app-server.mjs";
import { OperationManager } from "../server/operation-manager.mjs";
import { PATHS } from "../server/paths.mjs";

const isolated = Boolean(process.env.DFLSN_ISOLATED_TEST_ROOT);

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function harness(t, { onUpdate, runner = async () => ({ ok: true }), jobs = [], imageTasks = [], cleanup, onClose } = {}) {
  assert.ok(path.resolve(PATHS.workspaceRoot).startsWith(path.resolve(process.env.DFLSN_ISOLATED_TEST_ROOT) + path.sep));
  await mkdir(PATHS.runtimeRoot, { recursive: true });
  const root = await mkdtemp(path.join(PATHS.runtimeRoot, "quality-lock-test-"));
  const operationManager = new OperationManager({ root, onUpdate });
  await operationManager.initialize();
  const closeEvents = [];
  const server = new RuntimeServer({
    operationManager,
    jobManager: { list: () => jobs, activeJobs: () => jobs, flushAll: async () => {} },
    imageServiceManager: { activeTasks: () => imageTasks, close: async () => { closeEvents.push("image-close"); await onClose?.(server); } },
    restorationManager: { activeTasks: () => [], close: async () => { closeEvents.push("restoration-close"); await onClose?.(server); } },
  });
  const operationSpec = server.operationSpec.bind(server);
  server.operationSpec = (body) => ({ ...operationSpec(body), run: runner });
  t.after(async () => {
    cleanup?.();
    await server.stop();
    await rm(root, { recursive: true, force: true });
  });
  return { server, operationManager, closeEvents };
}

for (const state of ["queued", "running"]) {
  test(`exclusive quality reservation releases when cancelled at ${state} before its runner starts`, { skip: !isolated, timeout: 10_000 }, async (t) => {
    let manager, cancellation, triggered = false, called = false;
    const setup = await harness(t, {
      onUpdate: (operation) => {
        if (!triggered && operation.status === state) {
          triggered = true;
          // The real manager emits this update before invoking its runner.
          cancellation = manager.cancel(operation.id);
        }
      },
      runner: async () => { called = true; return {}; },
    });
    manager = setup.operationManager;
    const operation = await setup.server.startOperation({ kind: "mask-assist", side: "dst" });
    await cancellation;
    const completed = await manager.wait(operation.id);
    assert.equal(triggered, true);
    assert.equal(called, false);
    assert.equal(completed.status, "cancelled");
    assert.equal(setup.server.qualityOperationReservation, null);
    assert.doesNotThrow(() => setup.server.assertQualityOperationIdle());
  });
}

test("normal service stop cancels quality work and awaits its cleanup before closing services", { skip: !isolated, timeout: 10_000 }, async (t) => {
  const entered = deferred(), aborted = deferred(), releaseCleanup = deferred();
  let cleaned = false, stopFinished = false, closeAdmissionChecks = 0;
  const { server, operationManager, closeEvents } = await harness(t, {
    cleanup: () => releaseCleanup.resolve(),
    onClose: async (runtime) => {
      await assert.rejects(runtime.startOperation({ kind: "mask-assist", side: "dst" }), (error) => error.code === "RUNTIME_STOPPING");
      closeAdmissionChecks += 1;
    },
    runner: async ({ signal }) => {
      const cancelled = new Promise((resolve) => signal.addEventListener("abort", () => { aborted.resolve(); resolve(); }, { once: true }));
      entered.resolve();
      await cancelled;
      await releaseCleanup.promise;
      cleaned = true;
      throw new DOMException("cancelled after cleanup", "AbortError");
    },
  });
  const operation = await server.startOperation({ kind: "mask-publish", side: "dst" });
  await entered.promise;
  const stopping = server.stop().then(() => { stopFinished = true; });
  await aborted.promise;
  assert.equal(stopFinished, false);
  assert.equal(cleaned, false);
  assert.deepEqual(closeEvents, []);
  await assert.rejects(server.startOperation({ kind: "mask-assist", side: "dst" }), (error) => error.code === "RUNTIME_STOPPING");
  releaseCleanup.resolve();
  await stopping;
  assert.equal(cleaned, true);
  assert.equal(operationManager.get(operation.id).status, "cancelled");
  assert.equal(server.qualityOperationReservation, null);
  assert.deepEqual(closeEvents, ["image-close", "restoration-close"]);
  assert.equal(closeAdmissionChecks, 2);
});

test("quality work excludes other quality and read-analysis operations while status stays readable", { skip: !isolated, timeout: 10_000 }, async (t) => {
  const entered = deferred(), release = deferred();
  const { server, operationManager } = await harness(t, { cleanup: () => release.resolve(), runner: async () => {
    entered.resolve(); await release.promise; return { ready: true };
  } });
  const operation = await server.startOperation({ kind: "mask-assist", side: "dst" });
  await entered.promise;
  for (const kind of ["mask-assist", "mask-publish", "best-faceset-init", "best-faceset-analyze", "best-faceset-select", "best-faceset-publish", "best-faceset-recover", "pack", "similarity"]) {
    await assert.rejects(server.startOperation({ kind, side: "dst" }), (error) => error.code === "QUALITY_TASK_BUSY");
  }
  let mutationCalled = false;
  await assert.rejects(server.withWorkspaceMutation("test mutation", async () => { mutationCalled = true; }), (error) => error.code === "QUALITY_TASK_BUSY");
  assert.equal(mutationCalled, false);
  assert.equal(operationManager.list().length, 1);
  assert.equal(operationManager.get(operation.id).status, "running");
  await assert.rejects(server.stop({ plannedRestart: true }), (error) => error.code === "RUNTIME_RESTART_OPERATION_BUSY");
  release.resolve();
  assert.equal((await operationManager.wait(operation.id)).status, "succeeded");
  assert.doesNotThrow(() => server.assertQualityOperationIdle());
});

test("quality admission respects active training, image jobs and a pending project switch", { skip: !isolated, timeout: 10_000 }, async (t) => {
  const jobs = [{ id: "busy-train", state: "waiting_input" }], imageTasks = [];
  const { server, operationManager } = await harness(t, { jobs, imageTasks });
  await assert.rejects(server.startOperation({ kind: "mask-assist", side: "dst" }), (error) => error.code === "QUALITY_TASK_BUSY");
  jobs.length = 0;
  imageTasks.push({ id: "busy-image", status: "running" });
  await assert.rejects(server.startOperation({ kind: "mask-assist", side: "dst" }), (error) => error.code === "QUALITY_TASK_BUSY");
  imageTasks.length = 0;
  server.projectRestartPending = true;
  await assert.rejects(server.startOperation({ kind: "mask-assist", side: "dst" }), (error) => error.code === "PROJECT_RESTART_PENDING");
  assert.equal(operationManager.list().length, 0);
  assert.equal(server.qualityOperationReservation, null);
});

test("an unconfirmed mask process stop remains failed and retains its workspace reservation", { skip: !isolated, timeout: 10_000 }, async (t) => {
  const entered = deferred();
  const { server, operationManager } = await harness(t, { runner: async ({ signal }) => {
    entered.resolve();
    await new Promise(resolve => signal.addEventListener("abort", resolve, { once: true }));
    throw Object.assign(new Error("test process-tree termination is unconfirmed"), { code: "MASK_STOP_UNCONFIRMED" });
  } });
  try {
    const operation = await server.startOperation({ kind: "mask-assist", side: "dst" });
    await entered.promise;
    await operationManager.cancel(operation.id);
    const result = await operationManager.wait(operation.id);
    assert.equal(result.status, "failed");
    assert.equal(result.error.code, "MASK_STOP_UNCONFIRMED");
    assert.ok(server.qualityOperationReservation.stopUnconfirmed);
    await assert.rejects(server.startOperation({ kind: "mask-assist", side: "dst" }), error => error.code === "MASK_STOP_UNCONFIRMED");
    await assert.rejects(server.stop(), error => error.code === "MASK_STOP_UNCONFIRMED");
  } finally {
    // This fixture has no actual child process; release only its simulated latch.
    server.qualityOperationReservation = null;
  }
});

test("source restoration can coexist with an existing CPU aligned audit while GPU analysis still blocks admission", { skip: !isolated, timeout: 10_000 }, async () => {
  const operations = [{ id: "read-audit", kind: "asset-audit", status: "running" }];
  const server = new RuntimeServer({ operationManager: { list: () => operations },
    jobManager: { list: () => [], activeJobs: () => [] }, imageServiceManager: { activeTasks: () => [] } });
  await server.restorationManager.assertCanStart();
  assert.equal(server.restorationStartReservation, true);
  assert.throws(() => server.assertNoWorkspaceMutation(), error => error.code === "QUALITY_TASK_BUSY");
  server.restorationManager.onActiveChange(null);
  operations.push({ id: "gpu-role", kind: "roles", status: "running" });
  await assert.rejects(server.restorationManager.assertCanStart(), error => error.code === "QUALITY_TASK_BUSY");
  assert.equal(server.restorationStartReservation, false);
});

test("an unconfirmed Best Faceset helper stop retains the exclusive reservation", { skip: !isolated, timeout: 10_000 }, async t => {
  const { server, operationManager } = await harness(t, { runner: async () => {
    throw Object.assign(new Error("fixture helper close cannot be confirmed"), { code: "HELPER_STOP_UNCONFIRMED" });
  } });
  try {
    const operation = await server.startOperation({ kind: "best-faceset-analyze", side: "src" });
    const result = await operationManager.wait(operation.id);
    assert.equal(result.status, "failed"); assert.equal(result.error.code, "HELPER_STOP_UNCONFIRMED");
    assert.ok(server.qualityOperationReservation.stopUnconfirmed);
    await assert.rejects(server.startOperation({ kind: "best-faceset-publish", side: "src" }), error => error.code === "HELPER_STOP_UNCONFIRMED");
  } finally { server.qualityOperationReservation = null; }
});
