import { activeJobStates } from "./job-presentation.js";
import { faceModelFamily } from "./tae-training-configuration.js";

// A terminal selection must never redirect controls away from a live trainer.
export function selectTrainingJob(jobs = [], selectedJob = null) {
  let first = null, queued = null, selected = null;
  for (const job of jobs) {
    if (job.commandId !== "train.me" && job.commandId !== "train.tae") continue;
    if (job.state !== "queued" && activeJobStates.has(job.state)) return job;
    first ??= job;
    if (job.state === "queued") queued ??= job;
    if (job.id === selectedJob?.id) selected ??= job;
  }
  return queued ?? selected ?? first;
}

// An explicitly named model that has disappeared is not interchangeable with
// another checkpoint. Leave the new-task model picker in control when ambiguous.
export function getSavedTrainingContext(models = [], job = null) {
  const family = job ? (job.commandId === "train.tae" ? "TAE" : "ME") : null;
  const requestedName = job?.parameters?.forceModelName;
  let candidate = null;
  for (const model of models) {
    const modelFamily = faceModelFamily(model);
    if (family ? modelFamily !== family : modelFamily !== "ME" && modelFamily !== "TAE") continue;
    if (requestedName) {
      if (model.name === requestedName) return { model, state: "available" };
    } else {
      if (candidate) return { model: null, state: "ambiguous" };
      candidate = model;
    }
  }
  return { model: candidate, state: requestedName ? "missing" : candidate ? "available" : "empty" };
}

export function resolveSavedTrainingModel(models = [], job = null) {
  return getSavedTrainingContext(models, job).model;
}

export function getSavedTrainingModelState(models = [], job = null) {
  return getSavedTrainingContext(models, job).state;
}
