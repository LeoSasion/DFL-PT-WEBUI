import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { BestFacesetManager } from "../server/best-faceset-manager.mjs";
import { ProjectUiStateManager } from "../server/project-ui-state-manager.mjs";

const id = "plan-" + "a".repeat(32);
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
async function fixture(t) {
  const temp = await realpath(os.tmpdir());
  const root = await realpath(await mkdtemp(path.join(temp, "dfl-ux-state-")));
  assert.equal(path.dirname(root), temp, "all recursive cleanup stays in the created temporary directory");
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = { workspaceRoot: path.join(root, "workspace"), runtimeRoot: path.join(root, "workspace/.webui"),
    repositoryRoot: root, webuiRoot: path.join(root, "webui"), python: "fixed-test-python" };
  const expected = new Map();
  const helperCalls = [];
  const facesets = new BestFacesetManager({ paths, runProcess: async (_exe, args) => {
    assert.equal(args[1], "verify-source"); helperCalls.push(args);
    const plan = args[args.indexOf("--plan") + 1];
    const side = plan.includes(`${path.sep}src${path.sep}`) ? "src" : "dst";
    const bytes = await readFile(path.join(facesets.roots(side).aligned, "face.jpg"));
    return hash(bytes) === expected.get(side)
      ? { code: 0, stdout: JSON.stringify({ schemaVersion: 1, planId: id, sourceValid: true }), stderr: "" }
      : { code: 1, stdout: "", stderr: "PlanConflict: 原始素材SHA已变化" };
  } });
  for (const side of ["src", "dst"]) {
    await mkdir(facesets.roots(side).aligned, { recursive: true });
    await mkdir(facesets.planPath(side, id), { recursive: true });
    const bytes = Buffer.from(`isolated-${side}-source`); expected.set(side, hash(bytes));
    await writeFile(path.join(facesets.roots(side).aligned, "face.jpg"), bytes);
    const metadata = { schemaVersion: 1, planId: id, dataset: facesets.roots(side).aligned, datasetFingerprint: hash(bytes),
      total: 1, targetCount: 12, qualityModel: "foreground_tenengrad", confirmedReferenceMembers: ["face.jpg"],
      state: "published", globalSelection: true, createdAt: "2026-10-07T00:00:00Z", reviewVersion: 1, reviewRevision: 2,
      selection: { counts: { selected: 1, review: 0, rejected: 0 } } };
    await writeFile(path.join(facesets.planPath(side, id), "plan.json"), JSON.stringify(metadata));
    const directory = facesets.outputPath(side, id);
    for (const category of ["selected", "review", "rejected"]) await mkdir(path.join(directory, category), { recursive: true });
    await writeFile(path.join(directory, "selected/face.jpg"), bytes);
    const receipt = { schemaVersion: 1, kind: "best-training-faceset", planId: id, batchId: path.basename(directory), state: "committed",
      source: { dataset: metadata.dataset, fingerprint: metadata.datasetFingerprint }, counts: metadata.selection.counts,
      entries: [{ member: "face.jpg", destination: "selected/face.jpg", sha256: hash(bytes), status: "selected" }] };
    await writeFile(path.join(directory, "receipt.json"), JSON.stringify(receipt));
  }
  const manager = new ProjectUiStateManager({ paths, facesetManager: facesets });
  return { root, paths, facesets, manager, helperCalls };
}

test("per-side draft persists references and browser position through manager restart with source confirmation", async t => {
  const { paths, facesets, manager } = await fixture(t);
  const empty = await manager.draft("src");
  const result = await manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0,
    draft: { planId: id, identityReferences: ["face.jpg"], confirmed: true, offset: 120, category: "review", targetCount: 4000,
      sourceFingerprint: empty.sourceFingerprint } });
  assert.equal(result.revision, 1); assert.equal(result.identityConfirmationValid, true);
  const restarted = new ProjectUiStateManager({ paths, facesetManager: facesets });
  const restored = await restarted.draft("src");
  assert.equal(restored.draft.offset, 120); assert.equal(restored.draft.category, "review");
  assert.deepEqual(restored.draft.identityReferences, ["face.jpg"]); assert.equal(restored.draft.targetCount, 4000);
  assert.equal((await restarted.draft("dst")).draft, null, "side drafts cannot bleed into one another");
});

test("concurrent pages use draft CAS and a lost update never overwrites the winning draft", async t => {
  const { paths, manager } = await fixture(t);
  const replies = await Promise.allSettled([1000, 2000].map(targetCount => manager.saveDraft("src", {
    projectKey: paths.workspaceRoot, expectedRevision: 0, draft: { targetCount, identityReferences: ["face.jpg"] } } )));
  assert.equal(replies.filter(result => result.status === "fulfilled").length, 1);
  const failure = replies.find(result => result.status === "rejected");
  assert.equal(failure.reason.code, "PROJECT_UI_STATE_REVISION_CONFLICT");
  const current = await manager.draft("src");
  assert.equal(current.revision, 1);
  assert.equal(current.draft.targetCount, replies.find(result => result.status === "fulfilled").value.draft.targetCount);
});

test("paired explicit SRC/DST inputs survive CAS retry and restart without replacing the other side", async t => {
  const { paths, facesets, manager } = await fixture(t);
  const replies = await Promise.allSettled(["src", "dst"].map(side => manager.saveInput({
    projectKey: paths.workspaceRoot, expectedRevision: 0, side, selection: { kind: "selected", planId: id } })));
  const winner = replies.findIndex(result => result.status === "fulfilled"), loser = 1 - winner;
  assert.equal(replies[loser].reason.code, "PROJECT_UI_STATE_REVISION_CONFLICT");
  const after = await manager.saveInput({ projectKey: paths.workspaceRoot, expectedRevision: 1,
    side: ["src", "dst"][loser], selection: { kind: "selected", planId: id } });
  assert.equal(after.revision, 2);
  assert.equal(after.sides.src.kind, "selected"); assert.equal(after.sides.dst.kind, "selected");
  assert.equal(after.sides.src.valid, true); assert.equal(after.sides.dst.valid, true);
  assert.match(after.sides.src.path, /^data_src\//); assert.match(after.sides.dst.path, /^data_dst\//);
  const restored = await new ProjectUiStateManager({ paths, facesetManager: facesets }).inputs();
  assert.deepEqual(restored.sides, after.sides);
  const defaults = await manager.saveInput({ projectKey: paths.workspaceRoot, expectedRevision: 2, side: "src", selection: { kind: "aligned" } });
  assert.equal(defaults.sides.src.kind, "aligned"); assert.equal(defaults.sides.dst.kind, "selected");
});

test("cross-project and traversal requests are rejected before any state writes", async t => {
  const { paths, manager } = await fixture(t);
  for (const projectKey of [undefined, path.join(paths.workspaceRoot, "other"), path.dirname(paths.workspaceRoot)]) {
    await assert.rejects(manager.saveDraft("src", { projectKey, expectedRevision: 0, draft: { identityReferences: [] } }),
      error => error.code === "PROJECT_UI_STATE_PROJECT_CHANGED");
    await assert.rejects(manager.saveInput({ projectKey, expectedRevision: 0, side: "src", selection: { kind: "aligned" } }),
      error => error.code === "PROJECT_UI_STATE_PROJECT_CHANGED");
  }
  await assert.rejects(manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0,
    draft: { identityReferences: ["../face.jpg"] } }));
  assert.equal((await manager.draft("src")).revision, 0);
  assert.equal((await manager.inputs()).revision, 0);
});

test("source SHA changes invalidate confirmation even after a refreshed draft fingerprint", async t => {
  const { paths, facesets, manager } = await fixture(t);
  const initial = await manager.draft("src");
  await manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0,
    draft: { planId: id, identityReferences: ["face.jpg"], confirmed: true, sourceFingerprint: initial.sourceFingerprint } });
  await writeFile(path.join(facesets.roots("src").aligned, "face.jpg"), "changed isolated source");
  const invalid = await manager.draft("src");
  assert.equal(invalid.sourceValid, false); assert.equal(invalid.draft.confirmed, false);
  await assert.rejects(manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 1,
    draft: { planId: id, identityReferences: ["face.jpg"], confirmed: true, sourceFingerprint: initial.sourceFingerprint } }),
    error => error.code === "PROJECT_UI_STATE_SOURCE_CHANGED");
  const refreshed = await manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 1,
    draft: { planId: id, identityReferences: ["face.jpg"], confirmed: true, sourceFingerprint: invalid.sourceFingerprint } });
  assert.equal(refreshed.sourceValid, false, "fresh page fingerprint cannot resurrect a stale plan snapshot");
  assert.equal(refreshed.identityConfirmationValid, false);
});

test("sidecar changes invalidate draft confirmation without editing the image", async t => {
  const { paths, facesets, manager } = await fixture(t);
  const initial = await manager.draft("src");
  await manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0,
    draft: { identityReferences: ["face.jpg"], confirmed: true, sourceFingerprint: initial.sourceFingerprint } });
  await writeFile(path.join(facesets.roots("src").aligned, "face.jpg.landmarks.json"), '{"changed":true}');
  const result = await manager.draft("src");
  assert.equal(result.sourceValid, false); assert.equal(result.identityConfirmationValid, false);
});

test("withdrawn, modified copies or original-source invalidation refuse selection and preserve saved preference", async t => {
  const { paths, facesets, manager } = await fixture(t);
  await manager.saveInput({ projectKey: paths.workspaceRoot, expectedRevision: 0, side: "src", selection: { kind: "selected", planId: id } });
  const stateFile = path.join(paths.runtimeRoot, "ui-state/training-inputs.json"), before = await readFile(stateFile);
  const receiptFile = path.join(facesets.outputPath("src", id), "receipt.json");
  const receipt = JSON.parse(await readFile(receiptFile, "utf8"));
  await writeFile(receiptFile, JSON.stringify({ ...receipt, state: "withdrawn" }));
  assert.equal((await manager.inputs()).sides.src.valid, false);
  await assert.rejects(manager.saveInput({ projectKey: paths.workspaceRoot, expectedRevision: 1, side: "src", selection: { kind: "selected", planId: id } }));
  await writeFile(receiptFile, JSON.stringify(receipt));
  const copy = path.join(facesets.outputPath("src", id), "selected/face.jpg");
  const originalCopy = await readFile(copy); await writeFile(copy, "tampered independent copy");
  assert.equal((await manager.inputs()).sides.src.valid, false);
  await writeFile(copy, originalCopy);
  await writeFile(path.join(facesets.roots("src").aligned, "face.jpg"), "source updated but published copy remains");
  assert.equal((await manager.inputs()).sides.src.valid, false);
  await assert.rejects(manager.saveInput({ projectKey: paths.workspaceRoot, expectedRevision: 1, side: "src", selection: { kind: "selected", planId: id } }));
  assert.deepEqual(await readFile(stateFile), before, "invalid restore is a visible state, never an implicit switch to aligned");
});

test("damaged state remains an error rather than being overwritten with empty defaults", async t => {
  const { paths, manager } = await fixture(t);
  await manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0, draft: { identityReferences: [] } });
  const file = path.join(paths.runtimeRoot, "ui-state/best-src.json");
  await writeFile(file, "malformed state to preserve");
  await assert.rejects(manager.draft("src"));
  await assert.rejects(manager.saveDraft("src", { projectKey: paths.workspaceRoot, expectedRevision: 0, draft: { identityReferences: [] } }));
  assert.equal(await readFile(file, "utf8"), "malformed state to preserve");
});
