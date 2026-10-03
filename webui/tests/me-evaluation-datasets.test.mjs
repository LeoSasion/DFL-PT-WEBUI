import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { prepareCommand } from "../server/command-registry.mjs";
import { PATHS } from "../server/paths.mjs";
import {
  createTrainingModelKey,
  TrainingEvaluationManager,
} from "../server/training-evaluation-manager.mjs";

const srcFaceset = path.join(PATHS.workspaceRoot, "custom", "src.pak");
const dstFaceset = path.join(PATHS.workspaceRoot, "custom", "dst.zip");

function probe(side, kind, fingerprint, member = "alice/000.jpg") {
  return {
    schemaVersion: 1,
    side,
    datasetKind: kind,
    datasetFingerprint: fingerprint,
    sampleCount: 1,
    yawTicks: [0],
    pitchTicks: [0],
    samples: [{
      id: `${side}-p0-y0-01`, side, name: "000.jpg", member,
      sha256Prefix: "a".repeat(16), sourceFilename: "frame.png",
      cellId: "p0-y0", yaw: 0, pitch: 0, yawTick: 0, pitchTick: 0,
      sharpness: 0.5, brightness: 0.5, hasAppliedMask: false,
    }],
  };
}

test("ME evaluation binds exact preflight facesets and preserves packed member identity", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-evaluation-datasets-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const calls = [];
  let srcFingerprint = "a".repeat(64);
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"),
    archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async (side, { datasetPath }) => {
      calls.push([side, datasetPath]);
      return side === "src"
        ? probe(side, "pak", srcFingerprint)
        : probe(side, "zip", "b".repeat(64));
    },
  });
  await manager.initialize();
  const modelName = "packed members";
  const modelKey = createTrainingModelKey(modelName);
  const options = { modelName, srcFaceset, dstFaceset };
  const first = await manager.createOrReuseManifest(modelKey, options);
  assert.deepEqual(calls, [["src", srcFaceset], ["dst", dstFaceset]]);
  assert.equal(first.datasets.src.kind, "pak");
  assert.equal(first.datasets.dst.kind, "zip");
  assert.equal(first.datasets.src.atlasLinkable, false);
  assert.equal(first.datasets.dst.atlasLinkable, false);
  assert.deepEqual(first.samples.map((sample) => sample.member), ["alice/000.jpg", "alice/000.jpg"]);
  assert.equal((await manager.createOrReuseManifest(modelKey, options)).manifestId, first.manifestId);
  srcFingerprint = "c".repeat(64);
  const changed = await manager.createOrReuseManifest(modelKey, options);
  assert.notEqual(changed.manifestId, first.manifestId);
  for (const manifest of [first, changed]) {
    const target = manager.manifestPath(modelKey, manifest.manifestId);
    const explicit = { ...manifest };
    delete explicit.createdAt;
    await writeFile(target, JSON.stringify(explicit));
  }
  assert.deepEqual((await manager.listManifests(modelKey)).manifests.map(({ manifestId }) => manifestId),
    [first.manifestId, changed.manifestId].sort((left, right) => left.localeCompare(right)));
  const invalidTimestamp = manager.manifestPath(modelKey, first.manifestId);
  await writeFile(invalidTimestamp, JSON.stringify({ ...first, createdAt: null }));
  await assert.rejects(manager.getManifest(modelKey, first.manifestId), {
    code: "MANIFEST_INVALID",
  });
  await assert.rejects(manager.createOrReuseManifest(modelKey, { modelName, srcFaceset }), {
    code: "DATASET_PATH_INVALID",
  });
  await assert.rejects(manager.createOrReuseManifest(modelKey, {
    modelName, srcFaceset: "relative.pak", dstFaceset,
  }), { code: "DATASET_PATH_INVALID" });
});

test("ME manifest ID verifies content in Node and Python, including JS number formats", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-manifest-integrity-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const modelName = "表情 😀";
  const modelKey = createTrainingModelKey(modelName);
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async side => {
      const result = probe(side, "directory", "a".repeat(64));
      Object.assign(result.samples[0], {
        sourceFilename: "片段 😀.png", yaw: 1e-7, pitch: 1e20,
        sharpness: 1e-7, brightness: 1e-6,
      });
      return result;
    },
  });
  await manager.initialize();
  const manifest = await manager.createOrReuseManifest(modelKey, { modelName });
  const file = manager.manifestPath(modelKey, manifest.manifestId);
  const python = spawnSync(PATHS.python, ["-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
from training_evaluation import manifest_content_id
with open(sys.argv[2], encoding='utf-8') as stream:
    print(manifest_content_id(json.load(stream)))
`, path.join(PATHS.webuiRoot, "python"), file], { encoding: "utf8", timeout: 30_000 });
  assert.equal(python.status, 0, python.stderr);
  assert.equal(python.stdout.trim(), manifest.manifestId);

  const tampered = JSON.parse(await readFile(file, "utf8"));
  tampered.samples[0].yaw = 77;
  await writeFile(file, JSON.stringify(tampered));
  await assert.rejects(manager.getManifest(modelKey, manifest.manifestId), {
    code: "MANIFEST_INVALID",
  });

  const invalidIdentities = [
    { ...manifest, schemaVersion: 2 },
    { ...manifest, modelClass: "XSeg" },
  ];
  const validContentIds = spawnSync(PATHS.python, ["-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
from training_evaluation import manifest_content_id
print(json.dumps([manifest_content_id(item) for item in json.load(sys.stdin)]))
`, path.join(PATHS.webuiRoot, "python")], {
    encoding: "utf8", input: JSON.stringify(invalidIdentities), timeout: 30_000,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  });
  assert.equal(validContentIds.status, 0, validContentIds.stderr);
  const ids = JSON.parse(validContentIds.stdout.trim());
  for (const [index, identity] of invalidIdentities.entries()) {
    const manifestId = ids[index];
    await writeFile(manager.manifestPath(modelKey, manifestId),
      JSON.stringify({ ...identity, manifestId }));
    await assert.rejects(manager.getManifest(modelKey, manifestId), {
      code: "MANIFEST_INVALID",
    });
  }
});

test("atlas links are limited to each side's default loose directory", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-atlas-links-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async side => probe(side, "directory", "a".repeat(64)),
  });
  await manager.initialize();
  const modelName = "atlas links";
  const modelKey = createTrainingModelKey(modelName);
  const src = path.join(PATHS.workspaceRoot, "data_src", "aligned");
  const dst = path.join(PATHS.workspaceRoot, "data_dst", "aligned");
  const defaultManifest = await manager.createOrReuseManifest(modelKey, {
    modelName, srcFaceset: src, dstFaceset: dst,
  });
  assert.equal(defaultManifest.datasets.src.atlasLinkable, true);
  assert.equal(defaultManifest.datasets.dst.atlasLinkable, true);
  const pretrainingManifest = await manager.createOrReuseManifest(modelKey, {
    modelName, srcFaceset: dst, dstFaceset: dst,
  });
  assert.equal(pretrainingManifest.datasets.src.atlasLinkable, false);
  assert.equal(pretrainingManifest.datasets.dst.atlasLinkable, false);
});

test("ME evaluation rejects unsafe packed member names", async (t) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-evaluation-members-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
    probeBuilder: async side => probe(side, "zip", "a".repeat(64), "../000.jpg"),
  });
  await manager.initialize();
  const modelName = "invalid member";
  await assert.rejects(manager.createOrReuseManifest(createTrainingModelKey(modelName), {
    modelName, srcFaceset, dstFaceset,
  }), { code: "PROBE_SAMPLE_INVALID" });
});

test("guided ME training enables reproducible evaluation for custom PAK, ZIP and pretraining", {
  skip: process.env.DFLSN_ISOLATED_TEST_ROOT ? false : "requires isolated fixture",
}, async (t) => {
  const srcDirectory = path.join(PATHS.workspaceRoot, "evaluation-src");
  const dstDirectory = path.join(PATHS.workspaceRoot, "evaluation-dst");
  t.after(async () => {
    await rm(srcDirectory, { recursive: true, force: true });
    await rm(dstDirectory, { recursive: true, force: true });
  });
  const script = `
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture
src, dst = Path(sys.argv[2]), Path(sys.argv[3])
make_aligned(src / "alice", count=1)
make_aligned(dst / "bob", count=1, offset=25)
pack_fixture(src, "pak")
pack_fixture(dst, "zip")
`;
  const fixture = spawnSync(PATHS.python, ["-c", script, PATHS.currentDflRoot,
    srcDirectory, dstDirectory], { encoding: "utf8", timeout: 30_000 });
  assert.equal(fixture.status, 0, fixture.stderr);
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-evaluation-preflight-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
  });
  await manager.initialize();
  const ready = await prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { forceModelName: "evaluation-packed-smoke", cpuOnly: true,
      srcFaceset: "evaluation-src/faceset.pak", dstFaceset: "evaluation-dst/faceset.zip" },
  });
  assert.equal(ready.preflight.evaluation.enabled, true, ready.preflight.evaluation.reason);
  const manifest = await manager.getManifest(ready.preflight.evaluation.modelKey,
    ready.preflight.evaluation.manifestId);
  assert.equal(manifest.datasets.src.kind, "pak");
  assert.equal(manifest.datasets.dst.kind, "zip");
  assert.equal(manifest.samples.length, 2);
  assert.deepEqual(manifest.samples.map((item) => item.member), ["alice/000.jpg", "bob/000.jpg"]);

  const pretrain = await prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { forceModelName: "evaluation-pretrain-smoke", cpuOnly: true,
      pretrain: true, pretrainingDataDir: "evaluation-dst/faceset.zip" },
  });
  assert.equal(pretrain.preflight.evaluation.enabled, true, pretrain.preflight.evaluation.reason);
  const pretrainManifest = await manager.getManifest(pretrain.preflight.evaluation.modelKey,
    pretrain.preflight.evaluation.manifestId);
  assert.equal(pretrainManifest.datasets.src.fingerprint, pretrainManifest.datasets.dst.fingerprint);
  assert.deepEqual(pretrainManifest.samples.map((item) => item.member), ["bob/000.jpg", "bob/000.jpg"]);
});

test("guided ME training can evaluate separate project-local holdout facesets", {
  skip: process.env.DFLSN_ISOLATED_TEST_ROOT ? false : "requires isolated fixture",
}, async (t) => {
  const paths = Object.fromEntries(["train-src", "train-dst", "holdout-src", "holdout-dst"]
    .map(name => [name, path.join(PATHS.workspaceRoot, `evaluation-${name}`)]));
  t.after(async () => {
    for (const directory of Object.values(paths)) await rm(directory, { recursive: true, force: true });
  });
  const script = `
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.me_fixtures import make_aligned
from tests.test_me_data_features import pack_fixture
train_src, train_dst, holdout_src, holdout_dst = map(Path, sys.argv[2:])
make_aligned(train_src / "training-source", count=1)
make_aligned(train_dst / "training-target", count=1, offset=20)
make_aligned(holdout_src / "heldout-source", count=1, offset=40)
make_aligned(holdout_dst / "heldout-target", count=1, offset=60)
pack_fixture(holdout_src, "pak")
pack_fixture(holdout_dst, "zip")
`;
  const fixture = spawnSync(PATHS.python, ["-c", script, PATHS.currentDflRoot,
    ...Object.values(paths)], { encoding: "utf8", timeout: 30_000 });
  assert.equal(fixture.status, 0, fixture.stderr);
  const temporary = await mkdtemp(path.join(os.tmpdir(), "dfl-me-holdout-preflight-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const manager = new TrainingEvaluationManager({
    root: path.join(temporary, "active"), archiveRoot: path.join(temporary, "archive"),
  });
  await manager.initialize();
  const trainingParameters = {
    forceModelName: "evaluation-holdout-smoke", cpuOnly: true,
    srcFaceset: "evaluation-train-src", dstFaceset: "evaluation-train-dst",
  };
  const holdoutParameters = {
    ...trainingParameters,
    evaluationSrcFaceset: "evaluation-holdout-src/faceset.pak",
    evaluationDstFaceset: "evaluation-holdout-dst/faceset.zip",
  };
  const ready = await prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager, parameters: holdoutParameters });
  const launch = ready.definition.build({ parameters: ready.parameters, preflight: ready.preflight });
  const flagValue = flag => launch.args[launch.args.indexOf(flag) + 1];
  assert.equal(flagValue("--src"), paths["train-src"]);
  assert.equal(flagValue("--dst"), paths["train-dst"]);
  assert.equal(launch.env.DFL_WEB_EVAL_SRC, path.join(paths["holdout-src"], "faceset.pak"));
  assert.equal(launch.env.DFL_WEB_EVAL_DST, path.join(paths["holdout-dst"], "faceset.zip"));
  const manifest = await manager.getManifest(ready.preflight.evaluation.modelKey,
    ready.preflight.evaluation.manifestId);
  assert.equal(manifest.datasets.src.kind, "pak");
  assert.equal(manifest.datasets.dst.kind, "zip");
  assert.deepEqual(manifest.samples.map(sample => sample.member),
    ["heldout-source/000.jpg", "heldout-target/000.jpg"]);

  const defaultReady = await prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager, parameters: trainingParameters });
  const defaultLaunch = defaultReady.definition.build({ parameters: defaultReady.parameters,
    preflight: defaultReady.preflight });
  assert.equal(defaultLaunch.env.DFL_WEB_EVAL_SRC, paths["train-src"]);
  assert.equal(defaultLaunch.env.DFL_WEB_EVAL_DST, paths["train-dst"]);
  const defaultManifest = await manager.getManifest(defaultReady.preflight.evaluation.modelKey,
    defaultReady.preflight.evaluation.manifestId);
  assert.equal(defaultManifest.datasets.src.kind, "directory");
  assert.deepEqual(defaultManifest.samples.map(sample => sample.member),
    ["training-source/000.jpg", "training-target/000.jpg"]);
  assert.notEqual(defaultManifest.manifestId, manifest.manifestId);

  await assert.rejects(prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { ...trainingParameters, evaluationSrcFaceset: holdoutParameters.evaluationSrcFaceset },
  }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { ...holdoutParameters, evaluationSrcFaceset: trainingParameters.srcFaceset },
  }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { ...holdoutParameters, evaluationSrcFaceset: "evaluation-train-src/training-source" },
  }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("train.me", { launchMode: "guided",
    trainingEvaluationManager: manager,
    parameters: { ...holdoutParameters, evaluationSrcFaceset: "../outside" },
  }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("train.me", { launchMode: "guided",
    parameters: holdoutParameters,
  }), { code: "EVALUATION_UNAVAILABLE" });
});
