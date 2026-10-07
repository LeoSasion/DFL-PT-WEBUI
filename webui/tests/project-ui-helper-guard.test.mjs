import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import { RuntimeServer } from "../server/app-server.mjs";
import { activeHelperProcesses, runHelperProcess } from "../server/helper-process.mjs";

async function fixture(t) {
  const writes = [];
  const idle = { initialize: async () => {}, close: async () => {}, activeTasks: () => [] };
  const uiStateManager = {
    draft: async () => ({ revision: 0, draft: null }),
    inputs: async () => ({ revision: 0, sides: { src: { kind: "aligned" }, dst: { kind: "aligned" } } }),
    saveDraft: async (side, body) => { writes.push({ kind: "draft", side, body }); return { revision: 1, draft: body.draft }; },
    saveInput: async body => { writes.push({ kind: "input", body }); return { revision: 1, sides: { [body.side]: body.selection } }; },
  };
  const server = new RuntimeServer({ uiStateManager, facesetManager: {},
    operationManager: { initialize: async () => {}, list: () => [] },
    trainingEvaluationManager: idle, imageServiceManager: idle, restorationManager: idle,
    jobManager: { initialize: async () => {}, list: () => [], activeJobs: () => [], flushAll: async () => {} } });
  // The last test intentionally leaves the shared helper registry latched. Close
  // only this stub server's sockets, without asking the production stop guard to
  // dismiss a failure or touching any filesystem-backed manager.
  t.after(async () => {
    server.httpServer?.closeAllConnections();
    await new Promise(resolve => server.httpServer?.listening ? server.httpServer.close(resolve) : resolve());
    server.webSocketServer?.close();
  });
  const address = await server.start({ host: "127.0.0.1", port: 0 });
  const base = `http://127.0.0.1:${address.port}`;
  const health = await fetch(`${base}/api/health`);
  const cookie = health.headers.get("set-cookie").split(";", 1)[0];
  await health.arrayBuffer();
  const request = async (route, body) => {
    const response = await fetch(`${base}${route}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { Cookie: cookie, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { status: response.status, payload: await response.json() };
  };
  const bodies = [
    ["/api/best-facesets/src/draft", { projectKey: "controlled-fixture", expectedRevision: 0, draft: null }],
    ["/api/training-inputs", { projectKey: "controlled-fixture", expectedRevision: 0, side: "src", selection: { kind: "aligned" } }],
  ];
  return { server, request, writes, bodies };
}

test("UI autosave remains available during a confirmed ordinary quality operation", async t => {
  const { server, request, writes, bodies } = await fixture(t);
  server.qualityOperationReservation = { label: "controlled running quality operation" };
  for (const [route, body] of bodies) assert.equal((await request(route, body)).status, 200);
  assert.deepEqual(writes.map(item => item.kind), ["draft", "input"]);
});

test("latched quality stop failure blocks both UI state writes but preserves reads", async t => {
  const { server, request, writes, bodies } = await fixture(t);
  server.qualityOperationReservation = { stopUnconfirmed: "controlled helper exit unknown", stopCode: "HELPER_STOP_UNCONFIRMED" };
  for (const [route, body] of bodies) {
    const result = await request(route, body);
    assert.equal(result.status, 409);
    assert.equal(result.payload.error.code, "HELPER_STOP_UNCONFIRMED");
    assert.equal((await request(route)).status, 200);
  }
  assert.deepEqual(writes, []);
});

test("safe runtime shutdown blocks both UI state writes", async t => {
  const { server, request, writes, bodies } = await fixture(t);
  server.isStopping = true;
  for (const [route, body] of bodies) {
    const result = await request(route, body);
    assert.equal(result.status, 409);
    assert.equal(result.payload.error.code, "RUNTIME_STOPPING");
  }
  assert.deepEqual(writes, []);
});

test("shared HELPER_STOP_UNCONFIRMED latch blocks both HTTP state writes before their managers run", async t => {
  const { request, writes, bodies } = await fixture(t);
  const fake = new EventEmitter();
  fake.stdout = new EventEmitter(); fake.stderr = new EventEmitter();
  fake.stdin = new EventEmitter(); fake.stdin.end = () => {}; fake.pid = null;
  const controller = new AbortController();
  const work = runHelperProcess("controlled-fixture-no-process", [], {
    spawnProcess: () => fake, signal: controller.signal,
    terminateProcessTree: async () => { throw new Error("controlled stop verification unavailable"); },
  });
  controller.abort();
  await assert.rejects(work, { code: "HELPER_STOP_UNCONFIRMED" });
  assert.equal(activeHelperProcesses().at(-1).state, "stop-unconfirmed");
  for (const [route, body] of bodies) {
    const result = await request(route, body);
    assert.equal(result.status, 409);
    assert.equal(result.payload.error.code, "HELPER_STOP_UNCONFIRMED");
    assert.equal((await request(route)).status, 200);
  }
  assert.deepEqual(writes, []);
});
