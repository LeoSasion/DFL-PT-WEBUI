import assert from "node:assert/strict";
import test from "node:test";
import { TAE_CONFIG_PARAMETERS } from "../../shared/tae-training-options.mjs";
import { commandModelFamily, createTaskConfiguration, selectTaskModel, taskModels, taskOutputLocations } from "./task-configuration.js";
import { getSavedTrainingModelState, resolveSavedTrainingModel, selectTrainingJob } from "./training-context.js";
import { getNextWorkflowStep, getTrainingSaveStatus } from "./workflow-readiness.js";
import { trainingModelFamily } from "./tae-training-configuration.js";

const command = { id: "train.tae", category: "training", parameters: [
  { id: "forceModelName", default: "" }, { id: "silentStart", default: false },
  { id: "vae_weights", default: "" }, { id: "maxMinutes", default: 60 },
  { id: "resetDataState", default: false }, ...TAE_CONFIG_PARAMETERS,
  { id: "pretraining", default: false }, { id: "pretrainingDataDir", default: "" },
] };
const h3 = { name: "shared-name", type: "TAE", format: "tae-liae-pytorch", files: ["tae.pt"],
  config: { resolution: 448, batch_size: 2, random_flip: false, half_resolution_probability: 0 },
  vae_weights: "E:\\models\\TAE-int8.safetensors" };
const me = { name: "shared-name", type: "ME", format: "me-pytorch", files: ["me.pt"], config: { resolution: 128 } };

test("TAE model family and checkpoint selection never cross into ME", () => {
  assert.equal(commandModelFamily(command), "TAE");
  assert.equal(commandModelFamily({ id: "merge.tae" }), "TAE");
  assert.deepEqual(taskModels(command, [me, h3]), [h3]);
  assert.deepEqual(taskModels({ id: "train.me" }, [me, h3]), [me]);
  assert.deepEqual(taskModels(command, [{ ...h3, files: ["me.pt"] }]), []);
  assert.deepEqual(taskModels(command, [{ ...h3, type: "TAE" }]), [{ ...h3, type: "TAE" }]);
  assert.equal(trainingModelFamily(null, { ...h3, type: "TAE" }), "TAE");
});

test("TAE resume locks persisted configuration and permits base asset relocation", () => {
  const resumed = createTaskConfiguration(command, { resolution: 832, maxMinutes: 90 }, [me, h3]);
  assert.equal(resumed.newModel, false);
  assert.equal(resumed.parameters.resolution, 448);
  assert.equal(resumed.parameters.batchSize, 2);
  assert.equal(resumed.parameters.random_flip, false);
  assert.equal(resumed.parameters.half_resolution_probability, 0);
  assert.equal(resumed.parameters.training_mask_mode, 'face');
  assert.equal(resumed.parameters.vae_weights, h3.vae_weights);
  assert.equal(resumed.parameters.maxMinutes, 90);
  assert.equal(resumed.parameters.silentStart, true);
  const relocated = createTaskConfiguration(command, { vae_weights: "E:\\moved\\same-h3.safetensors" }, [h3]);
  assert.equal(relocated.parameters.vae_weights, "E:\\moved\\same-h3.safetensors");
  assert.equal(relocated.parameters.resolution, 448);
});

test('TAE restores the recorded mask mode and exposes it as resume-mutable', () => {
  const schema = TAE_CONFIG_PARAMETERS.find(item => item.id === 'training_mask_mode');
  assert.equal(schema.resumeMutable, true);
  assert.deepEqual(schema.options.map(item => item.value), ['face', 'expanded_feather', 'full']);
  const saved = { ...h3, config: { ...h3.config, training_mask_mode: 'expanded_feather' } };
  const restored = selectTaskModel(command, {}, saved.name, [saved]);
  assert.equal(restored.parameters.training_mask_mode, 'expanded_feather');
});

test("creating a fresh TAE resets previous checkpoint settings", () => {
  const next = selectTaskModel(command, { resolution: 832, resetDataState: true, vae_weights: "wrong" }, "__new__", [h3]);
  assert.equal(next.newModel, true);
  assert.equal(next.parameters.forceModelName, "");
  assert.equal(next.parameters.resolution, 256);
  assert.equal(next.parameters.resetDataState, false);
  assert.equal(next.parameters.vae_weights, "");
  assert.deepEqual(taskOutputLocations(command, {}, { forceModelName: "new-h3" }).paths, ["model/new-h3"]);
});

test("active TAE owns controls even when an old ME terminal is selected", () => {
  const old = { id: "old", commandId: "train.me", state: "cancelled" };
  const live = { id: "live", commandId: "train.tae", state: "running", parameters: { forceModelName: "shared-name" } };
  assert.equal(selectTrainingJob([old, live], old), live);
  assert.equal(resolveSavedTrainingModel([me, h3], live), h3);
  assert.equal(resolveSavedTrainingModel([me], live), null);
  assert.equal(getSavedTrainingModelState([me], live), "missing");
  assert.equal(trainingModelFamily(live, h3), "TAE");
});

test("TAE-only inventory is resumable and routes to the TAE merger", () => {
  assert.equal(resolveSavedTrainingModel([h3]), h3);
  assert.equal(resolveSavedTrainingModel([h3, me]), null);
  assert.equal(getSavedTrainingModelState([h3, me]), "ambiguous");
  assert.equal(trainingModelFamily(null, h3), "TAE");
  assert.equal(getNextWorkflowStep({ models: [h3] }).commandId, "merge.tae");
});

test("TAE safe stop confirmation still requires the acknowledged terminal state", () => {
  const job = { commandId: "train.tae", state: "cancelled", stopReason: "safe-stop" };
  assert.equal(getTrainingSaveStatus(job, h3).saveState, "confirmed");
  assert.equal(getTrainingSaveStatus({ ...job, stopReason: "safe-stop-unconfirmed" }, h3).saveState, "unconfirmed");
  assert.equal(getTrainingSaveStatus({ ...job, state: "stopping" }, h3).saveState, "pending");
});

test("TAE resumes the saved pretraining mode and pool and resets both for a fresh model", () => {
  const base = { ...h3, training_mode: "pretraining", pretraining_data: "data_dst/facesetR2.pak" };
  const resumed = createTaskConfiguration(command, {}, [base]);
  assert.equal(resumed.parameters.pretraining, true);
  assert.equal(resumed.parameters.pretrainingDataDir, base.pretraining_data);
  const fresh = selectTaskModel(command, resumed.parameters, "__new__", [base]);
  assert.equal(fresh.parameters.pretraining, false);
  assert.equal(fresh.parameters.pretrainingDataDir, "");
});
