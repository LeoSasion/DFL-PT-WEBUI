import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { TAE_RESOLUTIONS, TAE_CONFIG_PARAMETERS, TAE_DEFAULT_CONFIG, configFromTaeParameters,
  parametersFromTaeConfig, validateTaeConfig } from "../shared/tae-training-options.mjs";
import { buildCommand, getCommandDefinition, prepareCommand, validateCommandParameters } from "../server/command-registry.mjs";
import { discoverModels } from "../server/workspace-manager.mjs";
import { JobManager } from "../server/job-manager.mjs";
import { PATHS } from "../server/paths.mjs";
import { verifyTaeDependencies } from "../server/tae-dependencies.mjs";

const isolated = { skip: process.env.DFLSN_ISOLATED_TEST_ROOT ? false : "requires isolated fixture" };
const sha = "6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e";
const value = (launch, flag) => launch.args[launch.args.indexOf(flag) + 1];
async function temporary(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "tae-backend-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
async function model(directory, name, config = TAE_DEFAULT_CONFIG, weights = "") {
  const destination = path.join(directory, "model", name);
  await mkdir(destination, { recursive: true });
  await writeFile(path.join(destination, "tae.pt"), "synthetic checkpoint metadata fixture");
  await writeFile(path.join(destination, "metadata.json"), JSON.stringify({ format: "tae-liae-pytorch", version: 1,
    model_class: "TAE", modelClass: "TAE", checkpoint: "tae.pt", name, iteration: 2, config,
    vae_weights: weights, vae_sha256: sha }));
  return destination;
}

test("TAE shared config exactly matches Python and excludes old ME architecture settings", () => {
  const source = "import json,sys;sys.path.insert(0,sys.argv[1]);from tae.config import TrainConfig;print(json.dumps(TrainConfig().to_dict()))";
  const result = spawnSync(PATHS.python, ["-c", source, PATHS.currentDflRoot], { encoding: "utf8", timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(TAE_DEFAULT_CONFIG, JSON.parse(result.stdout));
  assert.equal(new Set(TAE_CONFIG_PARAMETERS.map(item => item.id)).size, TAE_CONFIG_PARAMETERS.length);
  assert.deepEqual(configFromTaeParameters(parametersFromTaeConfig(TAE_DEFAULT_CONFIG)), TAE_DEFAULT_CONFIG);
  assert.throws(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, archi: "liae-ud" }), /不支持/);
  assert.throws(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, resolution: 80 }), /无效/);
  for (const key of ["inter_dims", "hidden_dims"]) {
    assert.throws(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, [key]: 3 }), /无效/);
    assert.doesNotThrow(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, [key]: 4 }));
  }
  assert.throws(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, rgb_weight: 0, edge_weight: 0, dssim_weight: 0 }), /监督/);
});

test("TAE exposes the supported resolution presets and rejects arbitrary API values", () => {
  const source = "import json,sys;sys.path.insert(0,sys.argv[1]);from tae.config import TAE_RESOLUTIONS;print(json.dumps(TAE_RESOLUTIONS))";
  const result = spawnSync(PATHS.python, ["-c", source, PATHS.currentDflRoot], { encoding: "utf8", timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(TAE_RESOLUTIONS, JSON.parse(result.stdout));
  const schema = getCommandDefinition("train.tae").parameters.find(item => item.id === "resolution");
  assert.equal(schema.type, "select");
  assert.deepEqual(schema.options.map(option => option.value), [256, 448, 512, 640, 832]);
  assert.equal(getCommandDefinition("train.tae").label, "训练 TAEQI2.1");
  assert.equal(getCommandDefinition("merge.tae").label, "合成 TAEQI2.1 人脸");
  for (const resolution of TAE_RESOLUTIONS) {
    assert.equal(validateCommandParameters("train.tae", { resolution }, "guided").resolution, resolution);
    assert.doesNotThrow(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, resolution }));
  }
  for (const resolution of [64, 128, 320, 648, 672, 800, 1024]) {
    assert.throws(() => validateCommandParameters("train.tae", { resolution }, "guided"), { code: "PARAMETER_INVALID" });
    assert.throws(() => validateTaeConfig({ ...TAE_DEFAULT_CONFIG, resolution }));
  }
});

test("TAE dependency preflight detects missing modules with an actionable repair path", async () => {
  assert.deepEqual(await verifyTaeDependencies(), { missing: [] });
  await assert.rejects(verifyTaeDependencies({ run: async () => ({ code: 0,
    stdout: JSON.stringify({ missing: ["safetensors"] }) }) }), error =>
    error.code === "TAE_DEPENDENCY_MISSING" && /requirements-tae\.txt/.test(error.message));
  await assert.rejects(verifyTaeDependencies({ run: async () => ({ code: 1, stdout: "" }) }),
    { code: "TAE_RUNTIME_INVALID" });
});

test("TAE registry isolates training, merger and controls from ME and DFM", () => {
  const parameters = validateCommandParameters("train.tae", { forceModelName: "h3-new", batchSize: 2,
    learning_rate: 2e-4, vae_weights: path.join(os.tmpdir(), "TAE asset.safetensors"), maxMinutes: 3,
    gpuIndexes: "0", resetDataState: true, stopAtTarget: true, targetIterations: 4,
    ct_mode: "lct", training_mask_mode: "expanded_feather", random_warp: false,
    stopAtUtc: "2099-01-01T00:00:00Z" }, "guided");
  const launch = buildCommand(getCommandDefinition("train.tae"), { parameters }).launch;
  assert.equal(launch.args[0], PATHS.taeMain);
  assert.equal(launch.args[1], "web-train");
  assert.equal(value(launch, "--max-minutes"), "3");
  assert.equal(value(launch, "--target-iterations"), "4");
  assert.equal(value(launch, "--vae-weights"), parameters.vae_weights);
  const config = JSON.parse(value(launch, "--config-json"));
  assert.equal(config.batch_size, 2);
  assert.equal(config.learning_rate, 2e-4);
  assert.equal(config.ct_mode, "lct");
  assert.equal(config.training_mask_mode, "expanded_feather");
  assert.equal(config.random_warp, false);
  assert.equal(value(launch, "--stop-at-utc"), "2099-01-01T00:00:00Z");
  assert.equal(Object.hasOwn(config, "vae_weights"), false);
  assert.deepEqual(getCommandDefinition("train.tae").controls, ["save", "preview", "close"]);
  assert.equal(getCommandDefinition("export.dfm_tae").id, "export.dfm_tae");
  assert.equal(getCommandDefinition("train.me").parameters.find(item => item.configKey === "archi").default, "liae-ud");
  const mergeParameters = validateCommandParameters("merge.tae", { forceModelName: "h3-new", vae_weights: parameters.vae_weights }, "guided");
  const merge = buildCommand(getCommandDefinition("merge.tae"), { parameters: mergeParameters }).launch;
  assert.equal(value(merge, "--model"), "TAE");
  assert.equal(merge.env.TAE_VAE_WEIGHTS, parameters.vae_weights);
  assert.throws(() => validateCommandParameters("train.tae", { archi: "liae-ud" }, "guided"), { code: "PARAMETER_NOT_ALLOWED" });
});

test("TAE model discovery preserves family, VAE metadata and separate checkpoints", async t => {
  const root = await temporary(t);
  await model(root, "h3-ready", TAE_DEFAULT_CONFIG, "C:/local/h3.safetensors");
  await model(root, "bad-config", { ...TAE_DEFAULT_CONFIG, archi: "liae" });
  const wrong = await model(root, "wrong-checkpoint");
  await rm(path.join(wrong, "tae.pt"));
  await writeFile(path.join(wrong, "me.pt"), "old ME file");
  const inventory = await discoverModels(root);
  assert.equal(inventory.models.length, 1);
  assert.equal(inventory.models[0].type, "TAE");
  assert.equal(inventory.models[0].format, "tae-liae-pytorch");
  assert.equal(inventory.models[0].vae_sha256, sha);
  assert.equal(inventory.models[0].vae_weights, "C:/local/h3.safetensors");
  assert.equal(inventory.taeStats.count, 1);
  assert.equal(inventory.meStats.count, 0);
  const metadataFile = path.join(root, "model", "h3-ready", "metadata.json");
  const renamed = JSON.parse(await readFile(metadataFile, "utf8"));
  await writeFile(metadataFile, JSON.stringify({ ...renamed, model_class: "TAE", modelClass: "TAE" }));
  assert.equal((await discoverModels(root)).models[0].type, "TAE");
});

test("TAE pretraining routes both domains to one project pool without changing model config", () => {
  const parameters = validateCommandParameters("train.tae", { forceModelName: "rtm2-base",
    pretraining: true, pretrainingDataDir: "data_dst/facesetR2.pak", resolution: 512,
    inter_dims: 384, batchSize: 4, srcFaceset: "unused-src", dstFaceset: "unused-dst" }, "guided");
  const launch = buildCommand(getCommandDefinition("train.tae"), { parameters }).launch;
  assert.ok(launch.args.includes("--pretraining"));
  assert.equal(value(launch, "--src"), path.join(PATHS.workspaceRoot, "data_dst", "facesetR2.pak"));
  assert.equal(value(launch, "--src"), value(launch, "--dst"));
  const config = JSON.parse(value(launch, "--config-json"));
  assert.equal(config.resolution, 512);
  assert.equal(config.inter_dims, 384);
  assert.equal(config.batch_size, 4);
  assert.equal(config.pretraining, undefined);
  assert.throws(() => buildCommand(getCommandDefinition("train.tae"), {
    parameters: { ...parameters, pretrainingDataDir: "../outside.pak" },
  }), { code: "PARAMETER_INVALID" });
});

test("TAE resume rejects an accidental change between identity and pretraining", isolated, async t => {
  const weightsRoot = await temporary(t);
  const weights = path.join(weightsRoot, "fixture.safetensors");
  await writeFile(weights, "path-only fixture");
  const directory = await model(PATHS.workspaceRoot, "tae-pretrain-fixture", TAE_DEFAULT_CONFIG, weights);
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, "metadata.json");
  const metadata = JSON.parse(await readFile(file, "utf8"));
  await writeFile(file, JSON.stringify({ ...metadata, training_mode: "pretraining", pretraining_data: "data_src/aligned" }));
  const parameters = { forceModelName: "tae-pretrain-fixture", silentStart: true, cpuOnly: true };
  const ready = await prepareCommand("train.tae", { launchMode: "guided", parameters });
  assert.equal(ready.preflight.training.pretraining, true);
  assert.equal(ready.preflight.training.srcFaceset, ready.preflight.training.dstFaceset);
  assert.ok(buildCommand(ready.definition, ready).launch.args.includes("--pretraining"));
  await assert.rejects(prepareCommand("train.tae", { launchMode: "guided", parameters: {
    ...parameters, pretraining: false,
  } }), { code: "MODEL_CONFIG_INCOMPATIBLE" });
});

test("TAE resume preserves config, rejects cross-family and validates local weights", isolated, async t => {
  const weightsRoot = await temporary(t);
  const weights = path.join(weightsRoot, "fixture.safetensors");
  await writeFile(weights, "only path validation; Python verifies SHA before TAE load");
  const directory = await model(PATHS.workspaceRoot, "h3-resume-fixture", { ...TAE_DEFAULT_CONFIG, resolution: 448 }, weights);
  t.after(() => rm(directory, { recursive: true, force: true }));
  const parameters = { forceModelName: "h3-resume-fixture", silentStart: true, cpuOnly: true };
  const ready = await prepareCommand("train.tae", { launchMode: "guided", parameters });
  for (const decoder_mode of ['tail', 'full', 'frozen']) {
    const changed = await prepareCommand('train.tae', { launchMode: 'guided', parameters: { ...parameters, decoder_mode, decoder_lr_multiplier: .05 } });
    assert.equal(changed.preflight.training.newOptimizationStage, true);
    assert.equal(changed.preflight.training.config.decoder_mode, decoder_mode);
    assert.equal(changed.preflight.training.config.decoder_lr_multiplier, .05);
  }
  assert.equal(ready.preflight.training.config.resolution, 448);
  assert.equal(ready.preflight.training.newOptimizationStage, false);
  for (const training_mask_mode of ['expanded_feather', 'full']) {
    const changed = await prepareCommand('train.tae', { launchMode: 'guided', parameters: { ...parameters, training_mask_mode } });
    assert.equal(changed.preflight.training.newOptimizationStage, true);
    assert.equal(changed.preflight.training.config.training_mask_mode, training_mask_mode);
    assert.equal(JSON.parse(value(buildCommand(changed.definition, changed).launch, '--config-json')).training_mask_mode, training_mask_mode);
  }
  // Checkpoints written before the setting existed retain face-only training.
  const legacyFile = path.join(directory, 'metadata.json');
  const legacy = JSON.parse(await readFile(legacyFile, 'utf8'));
  delete legacy.config.training_mask_mode;
  await writeFile(legacyFile, JSON.stringify(legacy));
  const legacyReady = await prepareCommand('train.tae', { launchMode: 'guided', parameters });
  assert.equal(legacyReady.preflight.training.config.training_mask_mode, 'face');
  assert.equal(legacyReady.preflight.training.newOptimizationStage, false);
  assert.equal(ready.preflight.training.vae_weights, weights);
  assert.equal(ready.preflight.evaluation.enabled, false);
  const mergeReady = await prepareCommand("merge.tae", { launchMode: "guided", parameters: {
    forceModelName: "h3-resume-fixture", cpuOnly: true, maskMode: 4 } });
  assert.equal(mergeReady.preflight.vae_weights, weights);
  assert.equal(buildCommand(mergeReady.definition, mergeReady).launch.env.TAE_VAE_WEIGHTS, weights);
  await assert.rejects(prepareCommand("train.tae", { launchMode: "guided", parameters: { ...parameters, resolution: 256 } }), { code: "MODEL_CONFIG_INCOMPATIBLE" });
  await assert.rejects(prepareCommand("train.tae", { launchMode: "guided", parameters: { ...parameters, gpuIndexes: "0,1" } }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("train.tae", { launchMode: "guided", parameters: { ...parameters, vae_weights: "relative.safetensors" } }), { code: "TAE_VAE_MISSING" });
  await assert.rejects(prepareCommand("train.me", { launchMode: "guided", parameters }), { code: "MODEL_MISSING" });
  await assert.rejects(prepareCommand("train.tae", { launchMode: "guided", parameters: { forceModelName: "H3-RESUME-FIXTURE", cpuOnly: true, vae_weights: weights } }), { code: "MODEL_EXISTS" });
});

test("TAE jobs use detached heartbeat runner and require matching safe-stop acknowledgement", isolated, async t => {
  const weightsRoot = await temporary(t);
  const weights = path.join(weightsRoot, "fixture.safetensors");
  await writeFile(weights, "path-only fixture; no TAE execution");
  let launch, onExit;
  const manager = new JobManager({ runnerFactory() { throw new Error("TAE must use detached runner"); },
    meRunnerFactory(options) { launch = options; return { pid: 12345, onData() {}, onState() {},
      onExit(callback) { onExit = callback; }, dispose() {}, async kill() {} }; } });
  let exitHandled;
  const handleExit = manager.handleExit.bind(manager);
  manager.handleExit = (...args) => { exitHandled = handleExit(...args); return exitHandled; };
  t.after(async () => { await manager.flushAll(); for (const job of manager.jobs.values()) {
    if (job.previewTimer) clearInterval(job.previewTimer); if (job.healthTimer) clearInterval(job.healthTimer);
    if (job.stopTimer) clearTimeout(job.stopTimer); await rm(job.directory, { recursive: true, force: true });
  } });
  const running = await manager.start("train.tae", { launchMode: "guided", parameters: {
    forceModelName: "h3-job-fixture", cpuOnly: true, vae_weights: weights, maxMinutes: 1 } });
  assert.equal(running.runtimeIndependent, true);
  assert.ok(launch.env.DFL_WEB_HEARTBEAT_FILE.endsWith("trainer-heartbeat.json"));
  assert.equal(launch.args[0], PATHS.taeMain);
  await assert.rejects(manager.control(running.id, "evaluate"), { code: "CONTROL_NOT_SUPPORTED" });
  await manager.control(running.id, "close");
  const job = manager.getInternal(running.id);
  const request = JSON.parse((await readFile(job.controlFile, "utf8")).trim());
  await writeFile(job.controlAckFile, JSON.stringify({ operation: "close", status: "completed",
    requestedAt: request.requestedAt, checkpoint: path.join(PATHS.workspaceRoot, "model", "h3-job-fixture", "tae.pt") }));
  onExit({ exitCode: 0, signal: null });
  await exitHandled;
  assert.equal(manager.get(running.id).stopReason, "safe-stop");
  assert.equal(manager.get(running.id).state, "cancelled");
  assert.equal(manager.locks.has("workspace:model"), false);
  const unconfirmed = await manager.start("train.tae", { launchMode: "guided", parameters: {
    forceModelName: "h3-job-unconfirmed-fixture", cpuOnly: true, vae_weights: weights, maxMinutes: 1 } });
  await manager.control(unconfirmed.id, "close");
  const secondJob = manager.getInternal(unconfirmed.id);
  await writeFile(secondJob.controlAckFile, JSON.stringify({ operation: "close", status: "completed",
    requestedAt: "a different request", checkpoint: "tae.pt" }));
  onExit({ exitCode: 0, signal: null });
  await exitHandled;
  assert.equal(manager.get(unconfirmed.id).stopReason, "safe-stop-unconfirmed");
});
