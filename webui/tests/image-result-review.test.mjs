import assert from "node:assert/strict";
import test from "node:test";
import { imageResultVersions } from "../src/domain/image-result-review.js";
import { runtimeApi } from "../src/runtime/api.js";

test("versions follow the original input across tasks and preserve their parameter and cost records", () => {
  const old = { id: "old", createdAt: "2026-10-01", inputs: [{ inputId: "shared", name: "face.png" }], results: [{ index: 0, name: "result.png" }], prompt: "old prompt", usage: { amount: 1, currency: "CNY" } };
  const recent = { ...old, id: "recent", createdAt: "2026-10-05", prompt: "new prompt", usage: null };
  const unrelated = { ...recent, id: "other", inputs: [{ inputId: "different", name: "face.png" }] };
  const versions = imageResultVersions([old, unrelated, recent], recent, "shared");
  assert.deepEqual(versions.map(item => item.key), ["recent:0", "old:0"]);
  assert.equal(versions[0].task, recent);
  assert.equal(versions[1].task.usage.amount, 1);
  assert.equal(versions[0].task.usage, null);
});

test("text generation and missing selection never imply unrelated versions", () => {
  const task = { id: "text", results: [{ index: 0 }, { index: 1 }] };
  assert.deepEqual(imageResultVersions([task, { id: "other", results: [{ index: 0 }] }], task).map(item => item.key), ["text:0", "text:1"]);
  assert.deepEqual(imageResultVersions([], null, "original"), []);
  assert.deepEqual(imageResultVersions([{ id: "pending", inputs: [{ inputId: "original" }], results: [] }], task, "original"), []);
});

test("save-as client sends the chosen result index and name to the local endpoint without provider contact", async t => {
  const calls = [];
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return new Response(JSON.stringify({ ok: true, data: { exportId: "copy" } }), { headers: { "Content-Type": "application/json" } }); };
  assert.deepEqual(await runtimeApi.saveImageResultCopy("task-1", 2, { name: "pick.png" }), { exportId: "copy" });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/image-service/tasks/task-1/results/2/save-as");
  assert.equal(calls[0].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].options.body), { name: "pick.png" });
  assert.equal(runtimeApi.imageExportUrl("exp-copy"), "/api/image-service/exports/exp-copy");
});
