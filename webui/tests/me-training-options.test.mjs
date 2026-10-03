import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { ME_CONFIG_PARAMETERS, ME_DEFAULT_CONFIG, configFromParameters, parametersFromConfig,
  getMePreset, validateMeConfig } from "../shared/me-training-options.mjs";
import { buildCommand, getCommandDefinition, prepareCommand, validateCommandParameters } from "../server/command-registry.mjs";
import { PATHS } from "../server/paths.mjs";
import { JobManager } from "../server/job-manager.mjs";

const isolated = { skip: process.env.DFLSN_ISOLATED_TEST_ROOT ? false : "requires isolated fixture" };
const value = (launch, flag) => launch.args[launch.args.indexOf(flag) + 1];
const prepare = parameters => prepareCommand("train.me", { launchMode: "guided", parameters });
async function model(t, name, config = {}) {
  const directory = path.join(PATHS.workspaceRoot, "model", name);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "me.pt"), "checkpoint fixture");
  await writeFile(path.join(directory, "metadata.json"), JSON.stringify({ format: "me-pytorch", version: 1,
    model_class: "ME", name, checkpoint: "me.pt", iteration: 10, config }));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
async function faceset(t, name, file = "faceset.pak") {
  const directory = path.join(PATHS.workspaceRoot, name);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, file), "faceset fixture");
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("shared ME schema defaults exactly cover the Python configuration", () => {
  const source = "import json,sys; sys.path.insert(0,sys.argv[1]); from me_backend.config import MEConfig; print(json.dumps(MEConfig().to_dict()))";
  const result = spawnSync(PATHS.python, ["-c", source, PATHS.currentDflRoot], { encoding: "utf8", timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(ME_DEFAULT_CONFIG, JSON.parse(result.stdout));
  assert.equal(new Set(ME_CONFIG_PARAMETERS.map(item => item.id)).size, ME_CONFIG_PARAMETERS.length);
  assert.deepEqual(configFromParameters(parametersFromConfig(ME_DEFAULT_CONFIG)), ME_DEFAULT_CONFIG);
  assert.equal(ME_CONFIG_PARAMETERS.filter(item => item.structural).length, 7);
});

test("ME config validates cross-field limits and keeps presets partial", () => {
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, true_face_power: 0.1 }), /TrueFace/);
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, gan_patch_size: 129 }), /Patch/);
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, retraining_samples: true, retraining_capacity: 1 }), /回放容量/);
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, data_workers: true }), /进程数/);
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, legacy: true }), /不支持/);
  assert.equal(getMePreset("rg-fp16").use_rg, true);
  assert.equal(getMePreset("rg-fp16").optimizer_on_cpu, false);
  assert.equal(getMePreset("pretrain").optimizer_on_cpu, false);
  assert.equal(getMePreset("finetune").optimizer_on_cpu, false);
  assert.equal(Object.hasOwn(getMePreset("finetune"), "archi"), false);
  assert.equal(getMePreset("default").batchSize, 4);
  assert.throws(() => getMePreset("missing"), /未知/);
});

test("guided ME command transmits full JSON and explicit operations without shell interpolation", () => {
  const parameters = validateCommandParameters("train.me", {
    forceModelName: "advanced-config", archi: "df-tc", batchSize: 2, gpuIndexes: "0,1", use_rg: true,
    use_fp16: true, gan_power: 0.1, true_face_power: 0.2, data_workers: 2, ct_mode: "mix",
    allowConfigChange: true, resetOptimizer: true, resetDataState: true, debugSamples: true,
    initializeFrom: "initial-model", pretrainingDataDir: "pretrain/faceset.zip", stopAtTarget: true, targetIterations: 50,
  }, "guided");
  const launch = buildCommand(getCommandDefinition("train.me"), { parameters }).launch;
  assert.equal(value(launch, "--device"), "cuda:0,1");
  assert.equal(value(launch, "--target-iterations"), "50");
  const config = JSON.parse(value(launch, "--config-json"));
  assert.equal(config.archi, "df-tc");
  assert.equal(config.batch_size, 2);
  assert.equal(config.ct_mode, "mix");
  assert.equal(config.use_fp16, true);
  assert.equal(Object.hasOwn(config, "allowConfigChange"), false);
  for (const flag of ["--allow-config-change", "--reset-optimizer", "--reset-data-state", "--debug-samples"]) assert.ok(launch.args.includes(flag));
  assert.equal(value(launch, "--initialize-from"), path.join(PATHS.workspaceRoot, "model", "initial-model"));
  const estimate = buildCommand(getCommandDefinition("train.me"), { parameters: { ...parameters, stopAtTarget: false } }).launch;
  assert.ok(!estimate.args.includes("--target-iterations"));
  assert.throws(() => validateCommandParameters("train.me", { batchSize: true }, "guided"), { code: "PARAMETER_INVALID" });
  assert.throws(() => validateCommandParameters("train.me", { srcFaceset: "bad\npath" }, "guided"), { code: "PARAMETER_INVALID" });
});

test("resume inherits omitted legacy fields and reports explicitly approved changes", isolated, async t => {
  await model(t, "advanced-resume", { resolution: 96, batch_size: 2, ae_dims: 64, use_rg: true });
  const base = { forceModelName: "advanced-resume", silentStart: true, cpuOnly: true };
  const strict = await prepare(base);
  assert.equal(strict.preflight.training.config.resolution, 96);
  assert.equal(strict.preflight.training.config.ae_dims, 64);
  assert.equal(strict.preflight.training.config.use_rg, true);
  assert.deepEqual(strict.preflight.training.changes, []);
  const launch = buildCommand(strict.definition, strict).launch;
  assert.equal(JSON.parse(value(launch, "--config-json")).resolution, 96);
  await assert.rejects(prepare({ ...base, lr: 1e-5 }), { code: "MODEL_CONFIG_CHANGE_REQUIRED" });
  const changed = await prepare({ ...base, lr: 1e-5, allowConfigChange: true });
  assert.deepEqual(changed.preflight.training.changes, [{ field: "lr", previous: 5e-5, next: 1e-5 }]);
  await assert.rejects(prepare({ ...base, resolution: 128, allowConfigChange: true }), { code: "MODEL_CONFIG_INCOMPATIBLE" });
  await assert.rejects(prepare({ ...base, adabelief: false, allowConfigChange: true }), { code: "OPTIMIZER_RESET_REQUIRED" });
  const reset = await prepare({ ...base, adabelief: false, allowConfigChange: true, resetOptimizer: true });
  assert.equal(reset.preflight.training.config.adabelief, false);
});

test("ME preflight allows unique GPU lists and rejects CPU FP16 or insufficient batch", isolated, async () => {
  const base = { forceModelName: "advanced-multigpu", gpuIndexes: "0,1", batchSize: 2 };
  const ready = await prepare(base);
  assert.equal(ready.preflight.training.config.batch_size, 2);
  await assert.rejects(prepare({ ...base, gpuIndexes: "0,0" }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepare({ ...base, batchSize: 1 }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepare({ ...base, cpuOnly: true, use_fp16: true }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepare({ ...base, true_face_power: 0.1 }), { code: "PARAMETER_INVALID" });
});

test("pretraining uses its dedicated packed faceset and phase changes require data reset", isolated, async t => {
  const packed = await faceset(t, "advanced-pretraining", "faceset.zip");
  const base = { forceModelName: "advanced-pretrain", cpuOnly: true, pretrain: true,
    pretrainingDataDir: "advanced-pretraining/faceset.zip", srcFaceset: "missing-src", dstFaceset: "missing-dst" };
  const ready = await prepare(base);
  assert.equal(ready.preflight.training.srcFaceset, path.join(packed, "faceset.zip"));
  assert.equal(ready.preflight.training.dstFaceset, ready.preflight.training.srcFaceset);
  assert.equal(ready.preflight.evaluation.enabled, false);
  await assert.rejects(prepare({ ...base, pretrainingDataDir: "" }), { code: "PARAMETER_INVALID" });
  await model(t, "advanced-phase", { pretrain: true });
  await assert.rejects(prepare({ forceModelName: "advanced-phase", silentStart: true, pretrain: false, allowConfigChange: true }), { code: "DATA_RESET_REQUIRED" });
  const transitioned = await prepare({ forceModelName: "advanced-phase", silentStart: true, pretrain: false, allowConfigChange: true, resetDataState: true });
  assert.equal(transitioned.preflight.training.config.pretrain, false);
});

test("custom PAK and ZIP stay within the project and disable unavailable pose evaluation", isolated, async t => {
  const src = await faceset(t, "advanced-source", "source.pak");
  const dst = await faceset(t, "advanced-destination", "destination.zip");
  const ready = await prepare({ forceModelName: "advanced-packed", cpuOnly: true,
    srcFaceset: "advanced-source/source.pak", dstFaceset: path.join(dst, "destination.zip") });
  assert.equal(ready.preflight.training.srcFaceset, path.join(src, "source.pak"));
  assert.equal(ready.preflight.evaluation.enabled, false);
  await assert.rejects(prepare({ forceModelName: "advanced-escape", srcFaceset: "../outside" }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepare({ forceModelName: "advanced-empty-folder", srcFaceset: "advanced-source" }), { code: "INPUT_EMPTY" });
});

test("network initialization requires an existing compatible current-project checkpoint", isolated, async t => {
  const source = await model(t, "advanced-initial", { resolution: 96 });
  const ready = await prepare({ forceModelName: "advanced-initial-copy", initializeFrom: "advanced-initial", resolution: 96, cpuOnly: true });
  assert.equal(ready.preflight.training.initializeFrom, source);
  const inherited = await prepare({ forceModelName: "advanced-initial-copy", initializeFrom: "advanced-initial", cpuOnly: true });
  assert.equal(inherited.preflight.training.config.resolution, 96);
  await assert.rejects(prepare({ forceModelName: "advanced-initial-copy", initializeFrom: "advanced-initial", resolution: 128 }), { code: "MODEL_CONFIG_INCOMPATIBLE" });
  await assert.rejects(prepare({ forceModelName: "advanced-initial", silentStart: true, initializeFrom: "advanced-initial" }), { code: "PARAMETER_INVALID" });
});

test("sample-only output passes postflight and can become a newly trained model", isolated, async t => {
  const directory = path.join(PATHS.workspaceRoot, "model", "advanced-debug");
  await mkdir(directory, { recursive: true });
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(path.join(directory, "debug-src.png"), "png fixture");
  await writeFile(path.join(directory, "debug-dst.png"), "png fixture");
  await getCommandDefinition("train.me").postflight({ parameters: { forceModelName: "advanced-debug", debugSamples: true } });
  const ready = await prepare({ forceModelName: "advanced-debug", cpuOnly: true });
  assert.equal(ready.preflight.training.resumed, false);
  await writeFile(path.join(directory, "unrelated.dat"), "do not overwrite");
  await assert.rejects(prepare({ forceModelName: "advanced-debug", cpuOnly: true }), { code: "MODEL_EXISTS" });
});

test("TF ME import is a CPU model tool with strict components and an absent destination", isolated, async t => {
  const sourceDirectory = await mkdtemp(path.join(os.tmpdir(), "me-web-tf-source-"));
  t.after(() => rm(sourceDirectory, { recursive: true, force: true }));
  for (const component of ["encoder", "inter_AB", "inter_B", "decoder"]) {
    await writeFile(path.join(sourceDirectory, `legacy_ME_${component}.npy`), "numpy network fixture");
  }
  const parameters = { forceModelName: "advanced-import", sourceDirectory, legacyModelName: "legacy_ME", resolution: 96 };
  const ready = await prepareCommand("model.import_me_tf", { launchMode: "guided", parameters });
  assert.equal(ready.definition.category, "model");
  assert.deepEqual(ready.definition.locks, ["workspace:model"]);
  const launch = buildCommand(ready.definition, ready).launch;
  assert.equal(launch.args[1], "import-tf");
  assert.equal(value(launch, "--source"), sourceDirectory);
  assert.equal(value(launch, "--name"), "legacy_ME");
  assert.equal(JSON.parse(value(launch, "--config-json")).resolution, 96);
  assert.ok(!launch.args.includes("--src") && !launch.args.includes("--device"));
  await assert.rejects(prepareCommand("model.import_me_tf", { launchMode: "cli" }), { code: "PARAMETER_INVALID" });
  await assert.rejects(prepareCommand("model.import_me_tf", { launchMode: "guided", parameters: { ...parameters, archi: "df-ud" } }), { code: "MODEL_MISSING" });
  await model(t, "advanced-import");
  await assert.rejects(prepareCommand("model.import_me_tf", { launchMode: "guided", parameters }), { code: "MODEL_EXISTS" });
});

test("ME import starts through the real job contract and releases its model lock on success and failure", isolated, async t => {
  const sourceDirectory = await mkdtemp(path.join(os.tmpdir(), "me-web-import-launch-"));
  t.after(() => rm(sourceDirectory, { recursive: true, force: true }));
  for (const component of ["encoder", "inter_AB", "inter_B", "decoder"]) {
    await writeFile(path.join(sourceDirectory, `legacy_ME_${component}.npy`), "network component fixture");
  }
  let onExit;
  let runnerLaunch;
  let disposed = 0;
  const manager = new JobManager({ runnerFactory: launch => {
    runnerLaunch = launch;
    return { pid: 12345, onData() {}, onExit(callback) { onExit = callback; }, dispose() { disposed += 1; } };
  } });
  let exitHandled;
  const handleExit = manager.handleExit.bind(manager);
  manager.handleExit = (...args) => { exitHandled = handleExit(...args); return exitHandled; };
  t.after(async () => {
    await manager.flushAll();
    for (const job of manager.jobs.values()) await rm(job.directory, { recursive: true, force: true });
  });
  const parameters = { forceModelName: "advanced-import-launch", sourceDirectory, legacyModelName: "legacy_ME", resolution: 96 };
  const running = await manager.start("model.import_me_tf", { launchMode: "guided", parameters });
  assert.equal(running.state, "running");
  assert.deepEqual(running.controls, []);
  assert.equal(runnerLaunch.executable, PATHS.python);
  assert.equal(runnerLaunch.args[1], "import-tf");
  assert.equal(manager.locks.get("workspace:model"), running.id);
  assert.equal(manager.locks.has("gpu"), false);
  await assert.rejects(manager.start("model.import_me_tf", { launchMode: "guided",
    parameters: { ...parameters, forceModelName: "advanced-import-second" } }), { code: "RESOURCE_LOCKED" });
  await model(t, "advanced-import-launch", { resolution: 96 });
  onExit({ exitCode: 0, signal: null });
  await exitHandled;
  assert.equal(manager.get(running.id).state, "succeeded");
  assert.equal(manager.get(running.id).error, null);
  assert.equal(manager.locks.has("workspace:model"), false);
  const second = await manager.start("model.import_me_tf", { launchMode: "guided",
    parameters: { ...parameters, forceModelName: "advanced-import-second" } });
  assert.equal(second.state, "running");
  onExit({ exitCode: 1, signal: null });
  await exitHandled;
  assert.equal(manager.get(second.id).state, "failed");
  assert.equal(manager.locks.has("workspace:model"), false);
  assert.equal(disposed, 2);
});
