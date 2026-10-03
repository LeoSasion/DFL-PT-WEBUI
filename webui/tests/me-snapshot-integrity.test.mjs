import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildTrainingPoseRegression } from "../src/domain/training-pose-regression.js";
import { RuntimeServer } from "../server/app-server.mjs";
import { PATHS } from "../server/paths.mjs";
import {
  createTrainingModelKey, TrainingEvaluationManager,
} from "../server/training-evaluation-manager.mjs";

function probe(side) {
  return {
    schemaVersion: 1, side, datasetKind: "directory",
    datasetFingerprint: "a".repeat(64), sampleCount: 1,
    yawTicks: [0], pitchTicks: [0],
    samples: [{
      id: `${side}-p0-y0-01`, side, name: "人脸😀.jpg", member: "人物/人脸😀.jpg",
      sha256Prefix: "b".repeat(16), sourceFilename: "片段😀.mp4",
      cellId: "p0-y0", yaw: 1e-7, pitch: 0, yawTick: 0, pitchTick: 0,
      sharpness: 1e-7, brightness: 1e-6, hasAppliedMask: false,
    }],
  };
}

test("Python-published snapshots keep cross-language content IDs and reject metric/image tampering", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-snapshot-integrity-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"),
    archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async side => probe(side),
  });
  await manager.initialize();
  const modelName = "快照 😀";
  const modelKey = createTrainingModelKey(modelName);
  const manifest = await manager.createOrReuseManifest(modelKey, { modelName });
  const manifestPath = manager.manifestPath(modelKey, manifest.manifestId);
  const explicitManifest = { ...manifest };
  delete explicitManifest.createdAt;
  await writeFile(manifestPath, JSON.stringify(explicitManifest));
  assert.deepEqual(await manager.getManifest(modelKey, manifest.manifestId), explicitManifest);
  assert.deepEqual((await manager.listManifests(modelKey)).manifests, [explicitManifest]);
  const modelRoot = manager.modelDirectory(modelKey);
  const python = spawnSync(PATHS.python, ["-c", `
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from training_evaluation import AtomicEvaluationSnapshot
root, manifest_path, model_key = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
image = np.full((8, 8, 3), .5, dtype=np.float32)
ids = []
for iteration, mse in ((1, .02), (2, .01)):
    snapshot = AtomicEvaluationSnapshot(root, model_key, manifest['manifestId'], iteration,
        {'modelClass': 'ME', 'archi': 'liae-ud', 'resolution': 64,
         'faceType': 'f', 'dataFormat': 'NCHW', 'label': '人物😀'},
        model_weights_sha256=('c' if iteration == 1 else 'd') * 64)
    for sample in manifest['samples']:
        snapshot.add_sample(sample, {'input': image, 'reconstruction': image},
            {'reconstruction': {'maskedMse': mse, 'probe': 1e-7,
                                'large': 1e20, 'note': '低光😀'}})
    ids.append(snapshot.publish()['snapshotId'])
print(json.dumps(ids))
`, path.join(PATHS.webuiRoot, "python"), modelRoot,
  manifestPath, modelKey], {
    encoding: "utf8", timeout: 30_000,
  });
  assert.equal(python.status, 0, python.stderr);
  const [baselineId, currentId] = JSON.parse(python.stdout.trim());
  assert.match(baselineId, /^iter-00000001-[a-f0-9]{24}$/);
  assert.match(currentId, /^iter-00000002-[a-f0-9]{24}$/);

  const baseline = await manager.getSnapshot(modelKey, baselineId);
  const current = await manager.getSnapshot(modelKey, currentId);
  assert.equal(current.integrityStatus, "summary-verified");
  assert.equal(current.samples[0].metrics.reconstruction.note, "低光😀");
  const selected = () => buildTrainingPoseRegression({
    baseline, current, manifest, side: "dst", metricKey: "maskedMse",
  }).cells.find((cell) => cell.id === "p0-y0");
  assert.equal(selected().status, "improved");
  assert.deepEqual((await manager.listSnapshots(modelKey)).snapshots
    .map((snapshot) => snapshot.integrityStatus), ["summary-verified", "summary-verified"]);

  const summaryPath = path.join(manager.snapshotDirectory(modelKey, currentId), "summary.json");
  const originalSummary = await readFile(summaryPath);
  const changedSummary = JSON.parse(originalSummary.toString("utf8"));
  changedSummary.samples.find((sample) => sample.side === "dst")
    .metrics.reconstruction.maskedMse = .5;
  await writeFile(summaryPath, JSON.stringify(changedSummary));
  await assert.rejects(manager.getSnapshot(modelKey, currentId), {
    code: "SNAPSHOT_INTEGRITY_INVALID",
  });
  const catalogWithCorruption = await manager.listSnapshots(modelKey);
  assert.deepEqual(catalogWithCorruption.snapshots.map(({ snapshotId }) => snapshotId), [baselineId]);
  assert.deepEqual(catalogWithCorruption.invalidSnapshots, [
    { snapshotId: currentId, code: "SNAPSHOT_INTEGRITY_INVALID" },
  ]);
  await writeFile(summaryPath, originalSummary);
  assert.equal((await manager.getSnapshot(modelKey, currentId)).integrityStatus, "summary-verified");

  const imagePath = path.join(manager.snapshotDirectory(modelKey, currentId),
    "samples", "dst-p0-y0-01", "input.webp");
  const originalImage = await readFile(imagePath);
  assert.deepEqual((await manager.resolveSnapshotImage(modelKey, currentId,
    "dst-p0-y0-01", "input")).bytes, originalImage);
  const expectedHash = current.samples.find((sample) => sample.side === "dst")
    .variantSha256.input;
  assert.equal(createHash("sha256").update(originalImage).digest("hex"), expectedHash);
  await writeFile(imagePath, Buffer.from("tampered webp"));
  await assert.rejects(manager.resolveSnapshotImage(modelKey, currentId, "dst-p0-y0-01", "input"), {
    code: "SNAPSHOT_IMAGE_CORRUPT",
  });
  await writeFile(imagePath, originalImage);

  const verifiedResolve = manager.resolveSnapshotImage.bind(manager);
  manager.resolveSnapshotImage = async (...args) => {
    const verified = await verifiedResolve(...args);
    await writeFile(imagePath, Buffer.from("changed after verification"));
    return verified;
  };
  const server = new RuntimeServer({
    trainingEvaluationManager: manager,
    jobManager: { initialize: async () => {}, list: () => [] },
  });
  const address = await server.start({ port: 0 });
  t.after(() => server.stop());
  const response = await fetch(`http://127.0.0.1:${address.port}`
    + `/api/training-evaluations/${modelKey}/snapshots/${currentId}`
    + "/samples/dst-p0-y0-01/input");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "image/webp");
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), originalImage);
  manager.resolveSnapshotImage = verifiedResolve;
  await writeFile(imagePath, originalImage);

  assert.deepEqual((await manager.archiveSnapshots(modelKey, [currentId])).archivedSnapshotIds,
    [currentId]);
  const archivedSummaryPath = path.join(manager.archivedSnapshotDirectory(modelKey, currentId),
    "summary.json");
  await writeFile(archivedSummaryPath, JSON.stringify(changedSummary));
  await assert.rejects(manager.restoreSnapshots(modelKey, [currentId]), {
    code: "SNAPSHOT_INTEGRITY_INVALID",
  });
  assert.deepEqual((await manager.listSnapshots(modelKey)).snapshots.map(({ snapshotId }) => snapshotId),
    [baselineId]);
  assert.deepEqual(JSON.parse(await readFile(archivedSummaryPath, "utf8")), changedSummary);
  await assert.rejects(readFile(summaryPath), { code: "ENOENT" });
  await writeFile(archivedSummaryPath, originalSummary);
  assert.deepEqual((await manager.restoreSnapshots(modelKey, [currentId])).restoredSnapshotIds,
    [currentId]);
  assert.equal((await manager.getSnapshot(modelKey, currentId)).integrityStatus, "summary-verified");
});

test("legacy 8-hex snapshots remain readable with an explicit unanchored status", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-snapshot-legacy-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async side => probe(side),
  });
  await manager.initialize();
  const modelName = "legacy-fixture";
  const modelKey = createTrainingModelKey(modelName);
  const manifest = await manager.createOrReuseManifest(modelKey, { modelName });
  const snapshotId = "iter-00000003-abcdef12";
  const snapshotRoot = manager.snapshotDirectory(modelKey, snapshotId);
  await mkdir(snapshotRoot, { recursive: true });
  await writeFile(path.join(snapshotRoot, "summary.json"), JSON.stringify({
    schemaVersion: 1, snapshotId, modelKey, manifestId: manifest.manifestId,
    iteration: 3, metricSchemaVersion: 1,
    modelSignature: { modelClass: "ME", resolution: 64 },
    createdAt: "2026-10-04T00:00:00Z", samples: [],
  }));
  assert.equal((await manager.getSnapshot(modelKey, snapshotId)).integrityStatus,
    "legacy-unanchored");
  assert.equal((await manager.listSnapshots(modelKey)).snapshots[0].integrityStatus,
    "legacy-unanchored");
  assert.throws(() => manager.snapshotDirectory(modelKey, "iter-00000003-abcdef123456"), {
    code: "SNAPSHOT_ID_INVALID",
  });
});
