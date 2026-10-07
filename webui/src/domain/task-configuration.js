import { initializeMEParameters, modelMEParameters } from "./me-training-configuration.js";

export function commandModelFamily(command) {
  if (command?.id === "xseg.train") return "XSEG";
  if (["model.import_me_tf", "merge.preview_me"].includes(command?.id)) return "ME";
  return /^(?:train|merge|export\.dfm)[._](me)$/i.exec(command?.id ?? "")?.[1].toUpperCase() ?? null;
}

export function taskModels(command, models = []) {
  const family = commandModelFamily(command);
  return models.filter(model => {
    if (!family || model.type?.toUpperCase() !== family) return false;
    if (model.ready === false) return false;
    if (family === "ME") return model.format === "me-pytorch" && model.files?.includes("me.pt");
    if (Array.isArray(model.files)) return model.files.some(file => /_data\.dat$/i.test(file));
    // Older runtimes represented the whole XSeg directory as a model name.
    return !(family === "XSEG" && model.name === "xseg_model");
  });
}

export function createTaskConfiguration(command, supplied = {}, models = []) {
  const schemas = command?.parameters ?? [];
  const parameters = Object.fromEntries(schemas.map(schema => [schema.id,
    Object.hasOwn(supplied ?? {}, schema.id) ? supplied[schema.id] : schema.default,
  ]));
  const hasModel = schemas.some(schema => schema.id === "forceModelName");
  const saved = taskModels(command, models);
  const importing = command?.id === "model.import_me_tf";
  let newModel = importing;
  if (hasModel && !importing && !Object.hasOwn(supplied ?? {}, "forceModelName") && saved.length === 1) {
    parameters.forceModelName = saved[0].name;
    if (Object.hasOwn(parameters, "silentStart")) parameters.silentStart = true;
  }
  if (hasModel && command.category === "training") {
    newModel = parameters.forceModelName
      ? !saved.some(model => model.name === parameters.forceModelName)
      : saved.length === 0 && !Object.hasOwn(supplied ?? {}, "forceModelName");
    if (newModel && Object.hasOwn(parameters, "silentStart")) parameters.silentStart = false;
  }
  if (command?.id === "train.me" && !newModel) {
    const selected = saved.find(model => model.name === parameters.forceModelName);
    if (selected && !Object.hasOwn(supplied ?? {}, "silentStart")) parameters.silentStart = true;
    if (selected) {
      // Hydrate every field before applying deliberate task settings. A saved
      // model must never inherit the defaults of a different architecture.
      Object.assign(parameters, modelMEParameters(schemas, selected));
      for (const schema of schemas) {
        if (Object.hasOwn(supplied ?? {}, schema.id)) parameters[schema.id] = supplied[schema.id];
      }
    }
  }
  if (command?.id === "train.me" && newModel && parameters.initializeFrom) {
    const source = saved.find(model => model.name === parameters.initializeFrom);
    if (source) {
      Object.assign(parameters, initializeMEParameters(schemas, parameters, source));
      for (const schema of schemas) {
        if (Object.hasOwn(supplied ?? {}, schema.id)) parameters[schema.id] = supplied[schema.id];
      }
    }
  }
  if (parameters.cpuOnly && Object.hasOwn(parameters, "gpuIndexes")) parameters.gpuIndexes = "";
  return { parameters, newModel };
}

export function restoreMEModelConfiguration(command, current, model, { preserveSupplied = false } = {}) {
  const schemas = command?.parameters ?? [];
  const restored = modelMEParameters(schemas, model);
  const parameters = { ...current, ...restored, forceModelName: model?.name ?? "" };
  if (preserveSupplied) {
    for (const id of Object.keys(restored)) {
      if (Object.hasOwn(current ?? {}, id)) parameters[id] = current[id];
    }
  }
  for (const id of ["allowConfigChange", "resetOptimizer", "resetDataState"]) {
    if (Object.hasOwn(parameters, id)) parameters[id] = false;
  }
  for (const id of ["initializeFrom", "pretrainingDataDir"]) {
    if (Object.hasOwn(parameters, id)) parameters[id] = "";
  }
  if (Object.hasOwn(parameters, "silentStart")) parameters.silentStart = Boolean(model);
  return parameters;
}

export function selectTaskModel(command, current, name, models = []) {
  if (command?.id === "model.import_me_tf") return { parameters: { ...current }, newModel: true };
  const creating = name === "__new__";
  const saved = taskModels(command, models);
  const model = saved.find(candidate => candidate.name === name);
  let parameters = { ...current, forceModelName: creating ? "" : name };
  if (command?.id === "train.me") {
    parameters = restoreMEModelConfiguration(command, parameters, creating ? null : model);
    parameters.forceModelName = creating ? "" : name;
  }
  if (Object.hasOwn(parameters, "silentStart")) parameters.silentStart = !creating && Boolean(name);
  return { parameters, newModel: creating };
}

export function modelNameIssue(name, creating, models = []) {
  if (!creating) return null;
  const value = String(name ?? "");
  if (!value.trim()) return "empty";
  if (value.length > 64 || value.trim() !== value || /[/\\:*?"<>|\u0000-\u001f]/.test(value) || /[. ]$/.test(value)
      || /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value)) return "invalid";
  if (models.some(model => model.name?.toLocaleLowerCase() === value.toLocaleLowerCase())) return "duplicate";
  return null;
}

export function searchTaskCommands(commands, query = "") {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return commands.filter(command => {
    const content = `${command.label ?? ""} ${command.shortLabel ?? ""} ${command.description ?? ""} ${command.id}`.toLocaleLowerCase();
    return terms.every(term => content.includes(term));
  });
}

// Presentation only: these paths never become executable task arguments.
export function taskOutputLocations(command, workspace = {}, parameters = {}) {
  const id = command?.id ?? "";
  const side = ["src", "dst"].includes(command?.side) ? command.side : null;
  if (side && id.endsWith("extract_restored")) return { paths: [`data_${side}/aligned_restored/${parameters.restorationTaskId || "<修复记录>"}`] };
  if (id === "runtime.prepare_vision") return { paths: [], note: "runtime" };
  if (id === "xseg.train") return { paths: ["xseg_model"] };
  if (["train.me", "export.dfm_me", "model.import_me_tf"].includes(id) && parameters.forceModelName) {
    return { paths: [`model/${parameters.forceModelName}`] };
  }
  if (["training", "model"].includes(command?.category)) return { paths: ["model"] };
  if (id === "merge.preview_me") return { paths: [".webui/merge-previews/<本次小样>/merged", ".webui/merge-previews/<本次小样>/merged_mask"], note: "independent-merge-preview" };
  if (command?.category === "merge") return { paths: ["data_dst/merged", "data_dst/merged_mask"] };
  if (command?.category === "encode") {
    if (id === "encode.quality") return { paths: ["result.nut", "result_mask.nut", "result.mp4", "result_mask.mp4", "result.nut.media.json", "result_mask.nut.media.json", "result.mp4.media.json", "result_mask.mp4.media.json"], note: "overwrite-video" };
    const extension = id === "encode.master" ? "nut" : id === "encode.avi" ? "avi" : id === "encode.mov_lossless" ? "mov" : "mp4";
    const paths = [`result.${extension}`, `result_mask.${extension}`];
    if (id === "encode.master") paths.push(...paths.map(name => `${name}.media.json`));
    return { paths, note: "overwrite-video" };
  }
  if (side && id.startsWith("video.cut_")) {
    const extension = /\.[a-z0-9]+$/i.exec(workspace.materials?.[side]?.name ?? "")?.[0] ?? ".*";
    return { paths: [`data_${side}_cut${extension}`] };
  }
  if (side && (id.endsWith("extract_frames") || id.endsWith("denoise_frames"))) return { paths: [`data_${side}`] };
  if (side && id.endsWith("faces_enhance")) return { paths: [`data_${side}/aligned_enhanced/<独立批次>`], note: "independent-enhancement" };
  if (side && id.endsWith("faces_resize")) return { paths: [`data_${side}/aligned_resized`], note: "merge-back" };
  if (side && id.endsWith("fetch_labels")) return { paths: [`data_${side}/aligned_xseg`] };
  if (side) return { paths: [`data_${side}/aligned`] };
  return { paths: [""] };
}

export function displayWorkspacePath(workspacePath, relativePath) {
  if (!relativePath) return workspacePath;
  const separator = workspacePath?.includes("\\") ? "\\" : "/";
  return `${String(workspacePath ?? "").replace(/[\\/]+$/, "")}${separator}${relativePath.replaceAll("/", separator)}`;
}
