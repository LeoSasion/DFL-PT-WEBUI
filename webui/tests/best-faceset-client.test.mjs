import assert from "node:assert/strict";
import test from "node:test";
import { runtimeApi } from "../src/runtime/api.js";

test("Best Faceset client submits recoverable operation stages and never starts training", async t => {
  const original = globalThis.fetch; t.after(() => { globalThis.fetch = original; });
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, body: options.body ? JSON.parse(options.body) : null });
    const body = options.body ? JSON.parse(options.body) : null;
    return new Response(JSON.stringify({ ok: true, data: body?.kind ? { id: "op-fixture", status: "succeeded", result: { ok: true } } : [] }),
      { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const id = "plan-" + "a".repeat(32), identityReferences = ["profile.jpg", "front.jpg"];
  await runtimeApi.createBestFaceset("src", { targetCount: 4000, identityReferences, qualityModel: "foreground_tenengrad" });
  await runtimeApi.analyzeBestFaceset("src", id, { offset: 500, limit: 500 });
  await runtimeApi.selectBestFaceset("src", id, identityReferences);
  await runtimeApi.publishBestFaceset("src", id, true, { expectedReferences: identityReferences });
  await runtimeApi.publishBestFaceset("src", id, false, { expectedReferences: identityReferences, expectedRevision: 7 });
  await runtimeApi.recoverBestFaceset("src", id, true);
  assert.deepEqual(calls.map(item => item.body.kind), ["best-faceset-init", "best-faceset-analyze", "best-faceset-select", "best-faceset-publish", "best-faceset-publish", "best-faceset-recover"]);
  assert.deepEqual(calls[0].body.parameters.identityReferences, identityReferences);
  assert.equal(calls[1].body.parameters.offset, 500); assert.equal(calls[1].body.parameters.limit, 500);
  assert.equal(calls[3].body.parameters.dryRun, true); assert.equal(calls[4].body.parameters.dryRun, false);
  assert.deepEqual(calls[3].body.parameters.expectedReferences, identityReferences);
  assert.deepEqual(calls[4].body.parameters.expectedReferences, identityReferences);
  assert.equal(calls[4].body.parameters.expectedRevision, 7);
  assert.ok(calls.every(item => item.url === "/api/operations"));
});

test("Best drafts carry the current project key and a CAS revision for each side", async t => {
  const original = globalThis.fetch; t.after(() => { globalThis.fetch = original; });
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, method: options.method ?? "GET", body: options.body ? JSON.parse(options.body) : null });
    return new Response(JSON.stringify({ ok: true, data: { revision: 4, draft: null } }), { headers: { "Content-Type": "application/json" } });
  };
  const projectKey = "E:\\fixture\\workspaces\\project-a";
  await runtimeApi.bestFacesetDraft("dst", { projectKey });
  await runtimeApi.saveBestFacesetDraft("dst", { projectKey, expectedRevision: 3, draft: { planId: null, confirmed: false, identityReferences: ["profile.jpg"] } });
  assert.equal(calls[0].url, `/api/best-facesets/dst/draft?projectKey=${encodeURIComponent(projectKey)}`);
  assert.equal(calls[1].url, "/api/best-facesets/dst/draft");
  assert.equal(calls[1].body.projectKey, projectKey); assert.equal(calls[1].body.expectedRevision, 3);
});

test("manual review operations retain side, parent version, CAS and idempotency request IDs", async t => {
  const original = globalThis.fetch; t.after(() => { globalThis.fetch = original; }); const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    const body = JSON.parse(options.body); calls.push(body);
    return new Response(JSON.stringify({ ok: true, data: { id: "op-review", status: "succeeded", result: { planId: "child", reviewRevision: 8 } } }), { headers: { "Content-Type": "application/json" } });
  };
  const planId = "plan-" + "b".repeat(32), requestId = "fixture-request";
  await runtimeApi.runOperation("best-faceset-review-create", "src", { planId, expectedRevision: 0, requestId });
  await runtimeApi.runOperation("best-faceset-review-decide", "src", { planId, expectedRevision: 6, requestId, decision: "keep", members: ["profile.jpg"], identityReferences: ["front.jpg"] });
  await runtimeApi.runOperation("best-faceset-review-undo", "src", { planId, expectedRevision: 7, requestId, identityReferences: ["front.jpg"] });
  assert.deepEqual(calls.map(item => item.parameters.expectedRevision), [0, 6, 7]);
  assert.ok(calls.every(item => item.side === "src" && item.parameters.planId === planId && item.parameters.requestId === requestId));
  assert.deepEqual(calls[1].parameters.members, ["profile.jpg"]);
});
