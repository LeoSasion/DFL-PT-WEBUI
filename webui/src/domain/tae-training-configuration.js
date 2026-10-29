// TAE has an independent checkpoint contract. Resume never borrows ME settings.
export function modelTAEParameters(schemas = [], model = null) {
  const values = Object.fromEntries(schemas.filter(schema => schema.configKey).map(schema => [
    schema.id, model?.config?.[schema.configKey] ?? schema.default,
  ]));
  const vaeSchema = schemas.find(schema => schema.id === "vae_weights");
  if (vaeSchema) values.vae_weights = model?.vae_weights ?? model?.metadata?.vae_weights ?? vaeSchema.default;
  if (schemas.some(schema => schema.id === "pretraining")) values.pretraining = model?.training_mode === "pretraining";
  if (schemas.some(schema => schema.id === "pretrainingDataDir")) values.pretrainingDataDir = model?.pretraining_data ?? "";
  return values;
}

export function restoreTAEModelConfiguration(command, current, model) {
  const next = { ...current, ...modelTAEParameters(command?.parameters, model),
    forceModelName: model?.name ?? "", silentStart: Boolean(model) };
  if (Object.hasOwn(next, "resetDataState")) next.resetDataState = false;
  return next;
}

export function faceModelFamily(model) {
  const family = model?.type?.toUpperCase();
  return family;
}

export function trainingModelFamily(job, savedModel) {
  return job?.commandId === "train.tae" || (!job && faceModelFamily(savedModel) === "TAE") ? "TAE" : "ME";
}
