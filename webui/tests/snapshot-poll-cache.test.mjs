import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { TrainingEvaluationManager } from "../server/training-evaluation-manager.mjs";

const stable = value => Array.isArray(value) ? `[${value.map(stable).join(",")}]`
  : value && typeof value === "object" ? `{${Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + stable(value[key])).join(",")}}`
    : JSON.stringify(value);
async function fixture(t, count = 3) {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-snapshot-cache-"));
  t.after(async () => {
    assert.equal(path.dirname(path.resolve(temporary)), path.resolve(os.tmpdir()));
    await rm(temporary, { recursive: true, force: true });
  });
  const manager = new TrainingEvaluationManager({ root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive") });
  const modelKey = "fixture";
  const content = { schemaVersion: 1, modelKey, modelClass: "ME", samples: [] };
  const manifestId = createHash("sha256").update(stable(content)).digest("hex").slice(0, 24);
  const manifestPath = manager.manifestPath(modelKey, manifestId);
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, JSON.stringify({ ...content, manifestId }));
  const ids = [];
  for (let iteration = 1; iteration <= count; iteration++) {
    const snapshotId = `iter-${String(iteration).padStart(8, "0")}-abcdef00`; ids.push(snapshotId);
    const folder = manager.snapshotDirectory(modelKey, snapshotId); await mkdir(folder, { recursive: true });
    await writeFile(path.join(folder, "summary.json"), JSON.stringify({ schemaVersion: 1, snapshotId,
      modelKey, manifestId, iteration, metricSchemaVersion: 1, modelSignature: { family: "ME" },
      createdAt: "2026-10-25T00:00:00Z", samples: [] }));
  }
  await manager.writeIndex(modelKey, { schemaVersion: 1, modelKey, activeManifestId: manifestId, manifestIds: [manifestId], snapshotIds: ids });
  let reads = 0; const original = manager.readJsonLimited.bind(manager);
  manager.readJsonLimited = (...args) => { reads++; return original(...args); };
  return { manager, modelKey, manifestPath, ids, reads: () => reads };
}

test("unchanged polling reads only the index and newly published snapshots are discovered", async t => {
  const { manager, modelKey, ids, reads } = await fixture(t, 300);
  const first = await manager.listSnapshots(modelKey);
  assert.equal(first.snapshots.length, 300);
  assert.equal(reads(), 302, "read each summary and the shared manifest once");
  first.snapshots[0].iteration = -1;
  const before = reads(); const second = await manager.listSnapshots(modelKey);
  assert.equal(second.snapshots[0].iteration, 300, "API callers cannot mutate cached rows");
  assert.equal(reads() - before, 1, "unchanged summaries and manifests need no JSON reads");
  const nextId = "iter-00000301-abcdef00";
  const previous = JSON.parse(await readFile(path.join(manager.snapshotDirectory(modelKey, ids.at(-1)), "summary.json")));
  const folder = manager.snapshotDirectory(modelKey, nextId); await mkdir(folder);
  await writeFile(path.join(folder, "summary.json"), JSON.stringify({ ...previous, iteration: 301, snapshotId: nextId }));
  const beforeNew = reads(); const added = await manager.listSnapshots(modelKey);
  assert.equal(added.snapshots.length, 301); assert.equal(added.snapshots[0].snapshotId, nextId);
  assert.equal(reads() - beforeNew, 3);
});

test("summary edits and manifest corruption invalidate cached rows even when mtime is restored", async t => {
  const { manager, modelKey, manifestPath, ids } = await fixture(t);
  await manager.listSnapshots(modelKey);
  const summaryPath = path.join(manager.snapshotDirectory(modelKey, ids[0]), "summary.json");
  const original = JSON.parse(await readFile(summaryPath));
  const { stat } = await import("node:fs/promises"); const before = await stat(summaryPath);
  await writeFile(summaryPath, JSON.stringify({ ...original, modelKey: "invalid" }));
  await utimes(summaryPath, before.atime, before.mtime);
  const broken = await manager.listSnapshots(modelKey);
  assert.deepEqual(broken.invalidSnapshots, [{ snapshotId: ids[0], code: "SNAPSHOT_INVALID" }]);
  await writeFile(summaryPath, JSON.stringify(original));
  const manifest = await readFile(manifestPath);
  await writeFile(manifestPath, manifest.toString().replace('"ME"', '"XX"'));
  assert.equal((await manager.listSnapshots(modelKey)).invalidSnapshots.length, 3);
  await writeFile(manifestPath, manifest);
  assert.equal((await manager.listSnapshots(modelKey)).snapshots.length, 3);
});

test("archived rows disappear and restored rows are revalidated", async t => {
  const { manager, modelKey, ids } = await fixture(t);
  await manager.listSnapshots(modelKey);
  await manager.archiveSnapshots(modelKey, [ids[0]]);
  assert.equal((await manager.listSnapshots(modelKey)).snapshots.length, 2);
  await manager.restoreSnapshots(modelKey, [ids[0]]);
  assert.equal((await manager.listSnapshots(modelKey)).snapshots.length, 3);
  assert.equal(manager.snapshotRows.size, 3);
});
