import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { RuntimeServer } from "../server/app-server.mjs";
import { OperationManager } from "../server/operation-manager.mjs";
import { ProjectUiStateError } from "../server/project-ui-state-manager.mjs";

const id = `plan-${"a".repeat(32)}`;
async function fixture(t) {
  const temp = await realpath(os.tmpdir()), root = await realpath(await mkdtemp(path.join(temp, "dfl-ux-api-")));
  assert.equal(path.dirname(root), temp);
  t.after(() => rm(root, { recursive: true, force: true }));
  const projectKey = path.join(root, "workspace"), calls = [];
  let draftRevision = 0, inputRevision = 0, savedInputs = { src: { kind: "aligned" }, dst: { kind: "aligned" } };
  const state = {
    project: key => { calls.push({ action: "project", key }); if (key !== projectKey) throw new ProjectUiStateError("项目已切换", "PROJECT_UI_STATE_PROJECT_CHANGED", 409); },
    draft: async (side, options) => { state.project(options.projectKey); calls.push({ action: "draft", side, options }); return { revision: draftRevision, draft: null }; },
    saveDraft: async (side, body) => { state.project(body.projectKey); calls.push({ action: "saveDraft", side, body });
      if (body.expectedRevision !== draftRevision) throw new ProjectUiStateError("草稿版本冲突", "PROJECT_UI_STATE_REVISION_CONFLICT", 409);
      return { revision: ++draftRevision, draft: body.draft }; },
    inputs: async options => { state.project(options.projectKey); calls.push({ action: "inputs", options }); return { revision: inputRevision, sides: savedInputs }; },
    saveInput: async body => { state.project(body.projectKey); calls.push({ action: "saveInput", body });
      if (body.expectedRevision !== inputRevision) throw new ProjectUiStateError("训练输入版本冲突", "PROJECT_UI_STATE_REVISION_CONFLICT", 409);
      savedInputs = { ...savedInputs, [body.side]: body.selection }; return { revision: ++inputRevision, sides: savedInputs }; },
  };
  const idle = { initialize: async () => {}, close: async () => {}, activeTasks: () => [] };
  const facesets = {
    review: async (side, planId, action, options) => { calls.push({ action: "review", side, planId, reviewAction: action, options }); return { planId, reviewRevision: action === "create" ? 0 : options.expectedRevision + 1 }; },
    publish: async (side, planId, options) => { calls.push({ action: "publish", side, planId, options }); return { planId, dryRun: options.dryRun }; },
  };
  const operationManager = new OperationManager({ root: path.join(root, "operations") });
  const server = new RuntimeServer({ uiStateManager: state, facesetManager: facesets, operationManager,
    trainingEvaluationManager: idle, imageServiceManager: idle, restorationManager: idle,
    jobManager: { initialize: async () => {}, list: () => [], activeJobs: () => [], flushAll: async () => {} } });
  t.after(() => server.stop());
  const address = await server.start({ host: "127.0.0.1", port: 0 }), base = `http://127.0.0.1:${address.port}`;
  const health = await fetch(`${base}/api/health`), cookie = health.headers.get("set-cookie").split(";", 1)[0];
  const request = async (url, { body, authenticated = true, origin } = {}) => {
    const response = await fetch(`${base}${url}`, { method: body === undefined ? "GET" : "POST",
      body: body === undefined ? undefined : JSON.stringify(body), headers: {
        ...(authenticated ? { Cookie: cookie } : {}), ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        ...(origin ? { Origin: origin } : {}) } });
    return { status: response.status, payload: await response.json() };
  };
  return { projectKey, calls, state, facesets, operationManager, server, request };
}

test("draft HTTP reads and writes preserve project, side, browser location and revision errors", async t => {
  const { projectKey, calls, request } = await fixture(t);
  const query = encodeURIComponent(projectKey);
  const read = await request(`/api/best-facesets/dst/draft?projectKey=${query}`, { authenticated: false });
  assert.equal(read.status, 200); assert.equal(read.payload.data.revision, 0);
  assert.deepEqual(calls.find(call => call.action === "draft"), { action: "draft", side: "dst", options: { projectKey } });
  const body = { projectKey, expectedRevision: 0, draft: { planId: id, identityReferences: ["face.jpg"], confirmed: true,
    offset: 120, category: "review", targetCount: 4000, sourceFingerprint: "b".repeat(64) } };
  const saved = await request("/api/best-facesets/dst/draft", { body });
  assert.equal(saved.status, 200); assert.equal(saved.payload.data.revision, 1);
  assert.deepEqual(calls.find(call => call.action === "saveDraft"), { action: "saveDraft", side: "dst", body });
  const stale = await request("/api/best-facesets/dst/draft", { body });
  assert.equal(stale.status, 409); assert.equal(stale.payload.error.code, "PROJECT_UI_STATE_REVISION_CONFLICT");
  const foreign = await request(`/api/best-facesets/dst/draft?projectKey=${encodeURIComponent(projectKey + "-other")}`);
  assert.equal(foreign.status, 409); assert.equal(foreign.payload.error.code, "PROJECT_UI_STATE_PROJECT_CHANGED");
});

test("training input HTTP updates one side and forwards version conflicts without losing the other side", async t => {
  const { projectKey, calls, request } = await fixture(t);
  const read = await request(`/api/training-inputs?projectKey=${encodeURIComponent(projectKey)}`);
  assert.equal(read.status, 200); assert.deepEqual(read.payload.data.sides.src, { kind: "aligned" });
  const src = { projectKey, expectedRevision: 0, side: "src", selection: { kind: "selected", planId: id } };
  const first = await request("/api/training-inputs", { body: src });
  assert.equal(first.status, 200); assert.equal(first.payload.data.revision, 1);
  assert.deepEqual(first.payload.data.sides.dst, { kind: "aligned" });
  const dst = { ...src, side: "dst" };
  const stale = await request("/api/training-inputs", { body: dst });
  assert.equal(stale.status, 409); assert.equal(stale.payload.error.code, "PROJECT_UI_STATE_REVISION_CONFLICT");
  const both = await request("/api/training-inputs", { body: { ...dst, expectedRevision: 1 } });
  assert.equal(both.status, 200); assert.equal(both.payload.data.revision, 2);
  assert.deepEqual(both.payload.data.sides.src, src.selection); assert.deepEqual(both.payload.data.sides.dst, dst.selection);
  assert.deepEqual(calls.find(call => call.action === "saveInput").body, src);
});

test("new state routes preserve loopback origin checks and require session cookies for every write", async t => {
  const { projectKey, calls, request } = await fixture(t);
  const writes = [["/api/best-facesets/src/draft", { projectKey, expectedRevision: 0, draft: null }],
    ["/api/training-inputs", { projectKey, expectedRevision: 0, side: "src", selection: { kind: "aligned" } }],
    ...["create", "decide", "undo"].map(action => ["/api/operations", { kind: `best-faceset-review-${action}`, side: "src",
      parameters: { projectKey, planId: id, expectedRevision: 0, requestId: randomUUID(), identityReferences: ["face.jpg"], members: ["face.jpg"], decision: "keep" } }])];
  for (const [url, body] of writes) {
    const response = await request(url, { body, authenticated: false });
    assert.equal(response.status, 403); assert.equal(response.payload.error.code, "SESSION_REQUIRED");
  }
  assert.equal(calls.length, 0, "unauthenticated requests never reach state or operation managers");
  for (const url of [`/api/best-facesets/src/draft?projectKey=${encodeURIComponent(projectKey)}`, `/api/training-inputs?projectKey=${encodeURIComponent(projectKey)}`]) {
    const response = await request(url, { origin: "https://outside.example" });
    assert.equal(response.status, 403); assert.equal(response.payload.error.code, "ORIGIN_NOT_ALLOWED");
  }
  assert.equal(calls.length, 0);
});

test("all three review operations remain exclusive and carry immutable version, request and reference parameters", async t => {
  const { projectKey, calls, request, server, operationManager } = await fixture(t);
  for (const action of ["create", "decide", "undo"]) {
    const parameters = { projectKey, planId: id, expectedRevision: 7, requestId: randomUUID(), identityReferences: ["face.jpg"],
      members: ["face.jpg", "second.jpg"], decision: "exclude" };
    const spec = server.operationSpec({ kind: `best-faceset-review-${action}`, side: "dst", parameters });
    assert.equal(spec.exclusive, true);
    const started = await request("/api/operations", { body: { kind: spec.kind, side: "dst", parameters } });
    assert.equal(started.status, 202);
    assert.equal((await operationManager.wait(started.payload.data.id)).status, "succeeded");
    const call = calls.filter(item => item.action === "review").at(-1);
    assert.equal(call.side, "dst"); assert.equal(call.planId, id); assert.equal(call.reviewAction, action);
    for (const key of ["expectedRevision", "requestId", "identityReferences", "members", "decision"]) assert.deepEqual(call.options[key], parameters[key]);
    assert.ok(call.options.signal instanceof AbortSignal); assert.equal(typeof call.options.onProgress, "function");
    assert.equal(server.qualityOperationReservation, null);
  }
  let release, entered;
  const waiting = new Promise(resolve => { entered = resolve; });
  server.facesetManager.review = async () => { entered(); return new Promise(resolve => { release = resolve; }); };
  const active = await request("/api/operations", { body: { kind: "best-faceset-review-decide", side: "src",
    parameters: { projectKey, planId: id, expectedRevision: 0, requestId: randomUUID(), identityReferences: [], members: ["face.jpg"], decision: "defer" } } });
  assert.equal(active.status, 202); await waiting;
  for (const action of ["create", "decide", "undo"]) {
    const busy = await request("/api/operations", { body: { kind: `best-faceset-review-${action}`, side: "src",
      parameters: { projectKey, planId: id, expectedRevision: 0, requestId: randomUUID() } } });
    assert.equal(busy.status, 409); assert.equal(busy.payload.error.code, "QUALITY_TASK_BUSY");
  }
  release({ planId: id }); assert.equal((await operationManager.wait(active.payload.data.id)).status, "succeeded");
  assert.equal(server.qualityOperationReservation, null);
});

test("publication HTTP forwards the reviewed revision and current references to the fixed manager", async t => {
  const { projectKey, calls, request, operationManager } = await fixture(t);
  const parameters = { projectKey, planId: id, dryRun: true, expectedRevision: 8, expectedReferences: ["face.jpg"] };
  const started = await request("/api/operations", { body: { kind: "best-faceset-publish", side: "src", parameters } });
  assert.equal(started.status, 202); assert.equal((await operationManager.wait(started.payload.data.id)).status, "succeeded");
  const call = calls.find(item => item.action === "publish");
  assert.equal(call.options.expectedRevision, 8); assert.deepEqual(call.options.expectedReferences, ["face.jpg"]); assert.equal(call.options.dryRun, true);
});
