import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { BestFacesetManager } from "../server/best-faceset-manager.mjs";
import { PATHS, pathExists } from "../server/paths.mjs";

const id = "plan-" + "a".repeat(32);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");

test("expected review conflicts show the recovery reason without a Python traceback", async () => {
  const manager = new BestFacesetManager({ runProcess: async () => ({ code: 1, stdout: "", stderr:
    'Traceback (most recent call last):\n  File "private/source/path.py", line 10\nPlanConflict: 人工保留超过数量上限；请先排除其他图\n' }) });
  await assert.rejects(manager.helper("review-decide"), error => {
    assert.equal(error.status, 409);
    assert.equal(error.code, "BEST_FACESET_CONFLICT");
    assert.equal(error.message, "人工保留超过数量上限；请先排除其他图");
    return true;
  });
});

async function fixture(t, runner) {
  const root = await realpath(await mkdtemp(path.join(os.tmpdir(), "dfl-best-faceset-")));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = { workspaceRoot: path.join(root, "workspace"), runtimeRoot: path.join(root, "workspace", ".webui"),
    repositoryRoot: root, webuiRoot: path.join(root, "webui"), python: "fixed-python" };
  const manager = new BestFacesetManager({ paths, runProcess: runner });
  await mkdir(manager.roots("src").aligned, { recursive: true });
  await mkdir(manager.planPath("src", id), { recursive: true });
  const metadata = { schemaVersion: 1, planId: id, dataset: manager.roots("src").aligned,
    datasetFingerprint: "b".repeat(64), total: 3, targetCount: 2, qualityModel: "foreground_tenengrad", confirmedReferenceMembers: ["face.jpg"],
    state: "finalized", globalSelection: true, selection: { counts: { selected: 1, review: 1, rejected: 1 } } };
  await writeFile(path.join(manager.planPath("src", id), "plan.json"), JSON.stringify(metadata));
  return { manager, metadata, root };
}

test("Best Faceset validates fixed ranges, model allowlist, reference paths and workspace binding before inference", async t => {
  let called = false;
  const { manager, metadata } = await fixture(t, async () => { called = true; throw Error("must not run"); });
  await assert.rejects(manager.create("src", { identityReferences: ["../escape.jpg"] }), error => error.code === "BEST_FACESET_INVALID");
  await assert.rejects(manager.create("src", { qualityModel: "CR-FIQA" }), error => error.code === "BEST_FACESET_QUALITY_INVALID");
  await assert.rejects(manager.analyze("src", id, { limit: 501 }), error => error.code === "BEST_FACESET_RANGE_INVALID");
  await assert.rejects(manager.inspect("src", id, { offset: -1 }), error => error.code === "BEST_FACESET_RANGE_INVALID");
  await writeFile(path.join(manager.planPath("src", id), "plan.json"), JSON.stringify({ ...metadata, dataset: path.join(manager.paths.workspaceRoot, "data_dst", "aligned") }));
  await assert.rejects(manager.inspect("src", id), error => error.code === "BEST_FACESET_PLAN_INVALID");
  assert.equal(called, false);
});

test("global selection refuses a partial inventory, even when its current batch is complete", async t => {
  let calls = [];
  const { manager, metadata } = await fixture(t, async (_exe, args) => {
    calls.push(args[1]);
    return { code: 0, stdout: JSON.stringify({ ...metadata, total: 3, completedCount: 2, globalReady: false, nextOffset: 2,
      selectedRange: { offset: 0, count: 2 }, items: [{ member: "face.jpg", sha256: "c".repeat(64) }] }), stderr: "" };
  });
  await assert.rejects(manager.select("src", id, { identityReferences: ["face.jpg"] }), error => error.code === "BEST_FACESET_INCOMPLETE");
  assert.deepEqual(calls, ["inspect"], "neither confirm nor finalize may execute before full coverage");
});

test("publication rejects stale or revoked identity references before invoking Python publish, including dry run", async t => {
  const actions = [];
  const { manager, metadata } = await fixture(t, async (_exe, args) => {
    actions.push(args[1]);
    return { code: 0, stdout: JSON.stringify({ ...metadata, completedCount: 3, globalReady: true,
      nextOffset: null, selectedRange: { offset: 0, total: 3 }, items: [] }), stderr: "" };
  });
  for (const dryRun of [true, false]) {
    for (const expectedReferences of [["other-person.jpg"], []])
      await assert.rejects(manager.publish("src", id, { dryRun, expectedReferences }), error => error.code === "BEST_FACESET_REFERENCES_CHANGED");
    await assert.rejects(manager.publish("src", id, { dryRun }), error => error.code === "BEST_FACESET_REFERENCES_REQUIRED");
  }
  assert.ok(actions.length > 0); assert.ok(actions.every(action => action === "inspect"), "stale identity cannot enter the publication helper");
});

test("three-class publication verifies every copied byte and sidecar, and only selected is accepted for training", async t => {
  let sourceValid = true;
  const { manager, metadata } = await fixture(t, async (_exe, args) => {
    assert.equal(args[1], "verify-source", "training admission must verify the original source as well as copies");
    return { code: 0, stdout: JSON.stringify({ schemaVersion: 1, planId: id, sourceValid }), stderr: "" };
  });
  const directory = manager.outputPath("src", id);
  const records = [["selected", "face.jpg", Buffer.from("original face")], ["review", "boundary.jpg", Buffer.from("boundary")], ["rejected", "duplicate.jpg", Buffer.from("duplicate")]];
  const entries = [];
  for (const [status, name, bytes] of records) {
    await mkdir(path.join(directory, status), { recursive: true });
    await writeFile(path.join(directory, status, name), bytes);
    await writeFile(path.join(manager.roots("src").aligned, name), bytes);
    entries.push({ member: name, destination: `${status}/${name}`, sha256: sha(bytes), status,
      classification: status === "selected" ? "training-subset" : status === "review" ? "needs-review" : "low-value-duplicate" });
  }
  const sidecar = Buffer.from('{"native":98}'); entries[0].sidecarSha256 = sha(sidecar);
  await writeFile(path.join(directory, "selected", "face.jpg.landmarks.json"), sidecar);
  const receipt = { schemaVersion: 1, kind: "best-training-faceset", batchId: path.basename(directory), planId: id,
    state: "committed", source: { dataset: metadata.dataset, fingerprint: metadata.datasetFingerprint }, counts: { selected: 1, review: 1, rejected: 1 }, entries };
  const receiptFile = path.join(directory, "receipt.json"); await writeFile(receiptFile, JSON.stringify(receipt));
  assert.equal((await manager.requireSelected(path.join(directory, "selected"))).counts.selected, 1);
  sourceValid = false;
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_SOURCE_CHANGED");
  sourceValid = true;
  await writeFile(receiptFile, JSON.stringify({ ...receipt, source: { ...receipt.source, fingerprint: "c".repeat(64) } }));
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_RECEIPT_INVALID");
  await writeFile(receiptFile, JSON.stringify({ ...receipt, entries: entries.slice(0, 2), counts: { selected: 1, review: 1, rejected: 0 } }));
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_RECEIPT_INVALID");
  await writeFile(receiptFile, JSON.stringify(receipt));
  for (const target of [directory, path.dirname(directory), path.join(directory, "review"), path.join(directory, "rejected"), path.join(directory, "selected", "nested")])
    await assert.rejects(manager.requireSelected(target), error => error.code === "BEST_FACESET_TRAINING_PATH_INVALID");
  await writeFile(path.join(directory, "selected", "new.jpg"), "unexpected");
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_COPY_CHANGED");
  await rm(path.join(directory, "selected", "new.jpg"));
  await writeFile(path.join(directory, "review", "boundary.jpg"), "changed");
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_COPY_CHANGED");
  assert.deepEqual(await readFile(path.join(manager.roots("src").aligned, "boundary.jpg")), records[1][2]);
  await writeFile(path.join(directory, "review", "boundary.jpg"), records[1][2]);
  await writeFile(receiptFile, JSON.stringify({ ...receipt, state: "withdrawn" }));
  await assert.rejects(manager.requireSelected(path.join(directory, "selected")), error => error.code === "BEST_FACESET_TRAINING_EMPTY");
});

test("analysis of all faces keeps each helper batch at 500 and stops at complete global coverage", async t => {
  const { manager } = await fixture(t, async () => { throw Error("replaced below"); });
  const batches = []; const progress = [];
  manager.inspect = async () => ({ total: 1201, completedCount: 0, nextOffset: 0, globalReady: false });
  manager.analyze = async (_side, _id, options) => {
    batches.push([options.offset, options.limit]); options.onProgress({ stage: "features", current: Math.min(500, 1201 - options.offset), total: 500 });
    const completedCount = Math.min(options.offset + options.limit, 1201);
    return { total: 1201, completedCount, nextOffset: completedCount === 1201 ? null : completedCount, globalReady: completedCount === 1201 };
  };
  const result = await manager.analyzeAll("src", id, { onProgress: update => progress.push(update) });
  assert.deepEqual(batches, [[0, 500], [500, 500], [1000, 500]]);
  assert.equal(result.globalReady, true); assert.equal(progress.at(-1).current, 1201); assert.equal(progress.at(-1).total, 1201);
});

test("global category paging retains inventory positions and serves a hash-verified preview without a helper per thumbnail", async t => {
  let calls = 0;
  const { manager, metadata } = await fixture(t, async (_exe, args) => {
    calls++; assert.ok(args.includes("--status")); assert.ok(args.includes("review"));
    return { code: 0, stdout: JSON.stringify({ ...metadata, completedCount: 3, globalReady: true,
      selectedRange: { offset: 0, total: 1, count: 1 }, items: [{ position: 2, member: "boundary.jpg", sha256: sha(Buffer.from("boundary")), status: "review" }] }), stderr: "" };
  });
  await writeFile(path.join(manager.roots("src").aligned, "boundary.jpg"), "boundary");
  const state = await manager.inspect("src", id, { offset: 0, limit: 60, status: "review" });
  assert.match(state.items[0].imageUrl, /\/files\/2\/original$/);
  assert.deepEqual((await manager.file("src", id, 2)).bytes, Buffer.from("boundary"));
  assert.equal(calls, 1);
  await writeFile(path.join(manager.roots("src").aligned, "boundary.jpg"), "changed");
  await assert.rejects(manager.file("src", id, 2), error => error.code === "BEST_FACESET_SOURCE_CHANGED");
});

test("actual fixed Python CLI creates and reloads a persistent complete-inventory plan without touching source bytes", { timeout: 30000 }, async t => {
  if (!(await pathExists(PATHS.python))) { t.skip("project Python unavailable"); return; }
  const { manager, metadata } = await fixture(t);
  manager.paths = { ...manager.paths, python: PATHS.python, repositoryRoot: PATHS.repositoryRoot, webuiRoot: PATHS.webuiRoot };
  await writeFile(path.join(manager.roots("src").aligned, "face.jpg"), "JPEG inventory member, no inference in this test");
  const before = await readFile(path.join(manager.roots("src").aligned, "face.jpg"));
  const state = await manager.create("src", { targetCount: 4000, identityReferences: ["face.jpg"] });
  assert.equal(state.total, 1); assert.equal(state.completedCount, 0); assert.equal(state.globalReady, false);
  assert.equal(state.targetCount, 4000); assert.deepEqual(state.identityReferences, ["face.jpg"]);
  const reloaded = await manager.inspect("src", state.planId);
  assert.equal(reloaded.items[0].member, "face.jpg"); assert.equal(reloaded.items[0].sha256, sha(before));
  assert.deepEqual(await readFile(path.join(manager.roots("src").aligned, "face.jpg")), before);
  assert.equal(metadata.total, 3, "the independent fixture plan was not reused or overwritten");
});
