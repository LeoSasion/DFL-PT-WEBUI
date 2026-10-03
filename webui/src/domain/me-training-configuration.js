// The command schema is the shared source of defaults and configuration keys.
// Fall back to the two original fields for metadata from older servers.
export function meConfigSchemas(schemas = []) {
  return schemas.filter(schema => schema.configKey || ["resolution", "batchSize"].includes(schema.id));
}

export function meConfigKey(schema) {
  return schema.configKey ?? (schema.id === "batchSize" ? "batch_size" : schema.id);
}

export function modelMEParameters(schemas, model) {
  return Object.fromEntries(meConfigSchemas(schemas).map(schema => [
    schema.id, model?.config?.[meConfigKey(schema)] ?? schema.default,
  ]));
}

export function initializeMEParameters(schemas, current, model) {
  const next = { ...current, ...modelMEParameters(schemas, model), initializeFrom: model?.name ?? "" };
  if (Object.hasOwn(next, "pretrain")) next.pretrain = false;
  for (const id of ["allowConfigChange", "resetOptimizer", "resetDataState"]) {
    if (Object.hasOwn(next, id)) next[id] = false;
  }
  if (Object.hasOwn(next, "pretrainingDataDir")) next.pretrainingDataDir = "";
  return next;
}

export function meConfigurationChanges(schemas, parameters, model) {
  if (!model) return [];
  const saved = modelMEParameters(schemas, model);
  return meConfigSchemas(schemas).filter(schema => {
    const value = parameters[schema.id];
    const original = saved[schema.id];
    if (schema.type === "number" || typeof schema.default === "number") {
      return value === "" || value === null || value === undefined || Number(value) !== Number(original);
    }
    return value !== original;
  }).map(schema => ({ schema, from: saved[schema.id], to: parameters[schema.id] }));
}

export function applyMEPreset(schemas, current, preset, { structuralLocked = false } = {}) {
  const editable = new Map(schemas.map(schema => [schema.id, schema]));
  const next = { ...current };
  for (const [id, value] of Object.entries(preset ?? {})) {
    const schema = editable.get(id);
    if (schema && (!structuralLocked || !schema.structural)) next[id] = value;
  }
  return next;
}
