import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { RuntimeServer } from "../server/app-server.mjs";
import { OperationManager } from "../server/operation-manager.mjs";

test("Best Faceset HTTP API retains session protection, global category positions, and recoverable operation results", async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfl-best-api-")); t.after(() => rm(root, { recursive: true, force: true }));
  const id = `plan-${"a".repeat(32)}`, calls = [], bytes = Buffer.from("exact image bytes");
  const idle = { initialize: async () => {}, close: async () => {}, activeTasks: () => [] };
  const facesetManager = {
    status: async () => ({ qualityDefault: "foreground_tenengrad", analysisBatchLimit: 500 }),
    list: async side => [{ planId: id, side }],
    inspect: async (side, planId, options) => { calls.push({ action: "inspect", side, planId, options }); return { planId, items: [{ position: 782, status: options.status }] }; },
    file: async (side, planId, position, kind) => { calls.push({ action: "file", side, planId, position, kind }); return { bytes, mimeType: "image/jpeg", name: "face.jpg" }; },
    create: async (side, options) => { calls.push({ action: "create", side, options }); return { planId: id, completedCount: 0 }; },
    analyzeAll: async (side, planId, options) => { calls.push({ action: "analyzeAll", side, planId, options }); return { planId, globalReady: true }; },
    publish: async (side, planId, options) => { calls.push({ action: "publish", side, planId, options }); return { dryRun: options.dryRun, counts: { selected: 4, review: 7, rejected: 1 } }; },
  };
  const operationManager = new OperationManager({ root: path.join(root, "operations") });
  const server = new RuntimeServer({ facesetManager, operationManager, trainingEvaluationManager: idle,
    imageServiceManager: idle, restorationManager: idle, jobManager: { initialize: async () => {}, list: () => [], activeJobs: () => [], flushAll: async () => {} } });
  t.after(() => server.stop());
  const address = await server.start({ host: "127.0.0.1", port: 0 }), base = `http://127.0.0.1:${address.port}`;
  const health = await fetch(`${base}/api/health`), cookie = health.headers.get("set-cookie").split(";", 1)[0];
  const request = async (url, body, authenticated = true) => {
    const response = await fetch(`${base}${url}`, { ...(body ? { method: "POST", body: JSON.stringify(body) } : {}),
      headers: { ...(authenticated ? { Cookie: cookie } : {}), ...(body ? { "Content-Type": "application/json" } : {}) } });
    return { status: response.status, payload: await response.json() };
  };
  const unauthorized = await request("/api/operations", { kind: "best-faceset-init", side: "src" }, false);
  assert.equal(unauthorized.status, 403); assert.equal(calls.length, 0);
  const state = await request(`/api/best-facesets/src/plans/${id}?offset=60&limit=60&status=review`);
  assert.equal(state.status, 200); assert.equal(state.payload.data.items[0].position, 782);
  assert.deepEqual(calls[0].options, { offset: 60, limit: 60, status: "review", verifySource: true });
  const image = await fetch(`${base}/api/best-facesets/src/plans/${id}/files/782/original`);
  assert.equal(image.headers.get("content-type"), "image/jpeg"); assert.deepEqual(Buffer.from(await image.arrayBuffer()), bytes);
  const stages = [["best-faceset-init", { targetCount: 4000, identityReferences: ["face.jpg"] }],
    ["best-faceset-analyze", { planId: id, all: true, limit: 500 }], ["best-faceset-publish", { planId: id, dryRun: true, expectedReferences: ["face.jpg"] }]];
  for (const [kind, parameters] of stages) {
    const started = await request("/api/operations", { kind, side: "src", parameters }); assert.equal(started.status, 202);
    const finished = await operationManager.wait(started.payload.data.id); assert.equal(finished.status, "succeeded");
  }
  assert.deepEqual(calls.filter(item => ["create", "analyzeAll", "publish"].includes(item.action)).map(item => item.action), ["create", "analyzeAll", "publish"]);
  assert.deepEqual(calls.find(item => item.action === "create").options.identityReferences, ["face.jpg"]);
  assert.equal(calls.find(item => item.action === "publish").options.dryRun, true);
  assert.deepEqual(calls.find(item => item.action === "publish").options.expectedReferences, ["face.jpg"]);
  assert.equal(server.qualityOperationReservation, null);
});
