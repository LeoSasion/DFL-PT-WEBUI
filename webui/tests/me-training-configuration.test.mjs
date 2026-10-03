import assert from "node:assert/strict";
import test from "node:test";
import { ME_CONFIG_PARAMETERS, getMePreset } from "../shared/me-training-options.mjs";
import { applyMEPreset, initializeMEParameters, meConfigurationChanges, modelMEParameters } from "../src/domain/me-training-configuration.js";

test("complete model defaults include false and zero values and map batch IDs", () => {
  const result = modelMEParameters(ME_CONFIG_PARAMETERS, { config: { batch_size: 9, use_rg: false, gan_power: 0, archi: "df" } });
  assert.equal(Object.keys(result).length, ME_CONFIG_PARAMETERS.length);
  assert.equal(result.batchSize, 9);
  assert.equal(result.archi, "df");
  assert.equal(result.use_rg, false);
  assert.equal(result.gan_power, 0);
  assert.equal(result.optimizer_on_cpu, false);
});

test("initialization restores source configuration and clears resumed training operations", () => {
  const source = { name: "pretrained", config: { archi: "df-dtc", resolution: 224, pretrain: true, random_warp: false } };
  const result = initializeMEParameters(ME_CONFIG_PARAMETERS,
    { allowConfigChange: true, resetOptimizer: true, resetDataState: true, pretrainingDataDir: "old", forceModelName: "new" }, source);
  assert.equal(result.initializeFrom, "pretrained");
  assert.equal(result.archi, "df-dtc");
  assert.equal(result.pretrain, false);
  assert.equal(result.random_warp, false);
  assert.equal(result.forceModelName, "new");
  assert.equal(result.pretrainingDataDir, "");
  assert.equal(result.resetOptimizer, false);
});

test("change comparison treats typed numeric strings as equivalent but detects empty input", () => {
  const model = { config: { resolution: 128, batch_size: 4, lr: 0.00005 } };
  const parameters = { ...modelMEParameters(ME_CONFIG_PARAMETERS, model), batchSize: "4", lr: "0.00005" };
  assert.deepEqual(meConfigurationChanges(ME_CONFIG_PARAMETERS, parameters, model), []);
  assert.deepEqual(meConfigurationChanges(ME_CONFIG_PARAMETERS, { ...parameters, batchSize: "" }, model).map(change => change.schema.id), ["batchSize"]);
  assert.deepEqual(meConfigurationChanges(ME_CONFIG_PARAMETERS, { ...parameters, use_rg: true }, model).map(change => change.schema.id), ["use_rg"]);
});

test("presets cannot replace checkpoint architecture or carry unknown task parameters", () => {
  const current = { ...modelMEParameters(ME_CONFIG_PARAMETERS, { config: { archi: "df-dtc", resolution: 192 } }), forceModelName: "saved" };
  const result = applyMEPreset(ME_CONFIG_PARAMETERS, current, { ...getMePreset("default"), executable: "evil" }, { structuralLocked: true });
  assert.equal(result.archi, "df-dtc");
  assert.equal(result.resolution, 192);
  assert.equal(result.forceModelName, "saved");
  assert.equal(Object.hasOwn(result, "executable"), false);
  const micro = applyMEPreset(ME_CONFIG_PARAMETERS, current, getMePreset("finetune"), { structuralLocked: true });
  assert.equal(micro.archi, "df-dtc");
  assert.equal(micro.lr, 0.00001);
  assert.equal(micro.random_warp, false);
  assert.equal(micro.retraining_samples, true);
});
