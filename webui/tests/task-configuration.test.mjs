import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import {
  commandModelFamily, createTaskConfiguration, displayWorkspacePath,
  modelNameIssue, restoreMEModelConfiguration, searchTaskCommands, selectTaskModel, taskModels, taskOutputLocations,
} from "../src/domain/task-configuration.js";
import { buildCommand, getCommandDefinition, listCommands, validateCommandParameters } from "../server/command-registry.mjs";
import { PATHS } from "../server/paths.mjs";
import { ME_CONFIG_PARAMETERS, ME_DEFAULT_CONFIG } from "../shared/me-training-options.mjs";

const commands = listCommands();
const command = id => commands.find(item => item.id === id);

test("new extraction uses verified quality defaults while retaining explicit compatibility choices", () => {
  assert.deepEqual(taskOutputLocations(command("encode.master")).paths, ["result.nut", "result_mask.nut", "result.nut.media.json", "result_mask.nut.media.json"]);
  const detector = command("src.extract_faces").parameters.find(item => item.id === "detector");
  assert.equal(detector.default, "yolo26s-face");
  assert.deepEqual(detector.options.map(item => item.value), ["s3fd", "yolo11m-face", "yolo12l-face", "yolo26s-face", "manual"]);
  for (const id of ["src.extract_faces", "dst.extract_faces", "src.extract_restored", "dst.extract_restored"]) {
    const defaults = createTaskConfiguration(command(id)).parameters;
    assert.equal(defaults.detector, "yolo26s-face", id);
    assert.equal(defaults.landmarkModel, "tufa", id);
    assert.equal(defaults.jpegQuality, 100, id);
    assert.deepEqual(command(id).parameters.find(item => item.id === "landmarkModel").options.map(item => item.value), ["tufa", "fan"]);
  }
  const compatibility = createTaskConfiguration(command("src.extract_faces"), { detector: "s3fd", landmarkModel: "fan", jpegQuality: 92 });
  assert.equal(compatibility.parameters.detector, "s3fd");
  assert.equal(compatibility.parameters.landmarkModel, "fan");
  assert.equal(compatibility.parameters.jpegQuality, 92);
});

test("quality export declares all four video files and four verification records", () => {
  const output = taskOutputLocations(command("encode.quality"));
  assert.deepEqual(output.paths, ["result.nut", "result_mask.nut", "result.mp4", "result_mask.mp4",
    "result.nut.media.json", "result_mask.nut.media.json", "result.mp4.media.json", "result_mask.mp4.media.json"]);
  assert.equal(output.paths.filter(name => !name.endsWith(".media.json")).length, 4);
  assert.equal(output.paths.filter(name => name.endsWith(".media.json")).length, 4);
  assert.equal(output.note, "overwrite-video");
  for (const side of ["src", "dst"]) {
    assert.deepEqual(taskOutputLocations(command(`${side}.extract_restored`), {}, { restorationTaskId: "rst-verified" }).paths,
      [`data_${side}/aligned_restored/rst-verified`]);
  }
});

test("merge defaults to original DST aligned and contains explicitly selected reviewed copies", () => {
  const defaults = createTaskConfiguration(command("merge.me")).parameters;
  assert.equal(defaults.dstFaceset, "data_dst/aligned");
  const launchFor = dstFaceset => buildCommand(getCommandDefinition("merge.me"), {
    launchMode: "guided", parameters: validateCommandParameters("merge.me", { dstFaceset }, "guided"),
  }).launch;
  for (const selected of [defaults.dstFaceset, "data_dst/aligned_restored/rst-verified",
    path.join(PATHS.workspaceRoot, "data_dst", "aligned_assisted", "reviewed-copy")]) {
    const launch = launchFor(selected);
    const aligned = launch.args[launch.args.indexOf("--aligned-dir") + 1];
    assert.equal(aligned, path.resolve(PATHS.workspaceRoot, selected));
    assert.equal(launch.args[launch.args.indexOf("--input-dir") + 1], path.join(PATHS.workspaceRoot, "data_dst"));
  }
  for (const outside of ["../outside", path.resolve(PATHS.workspaceRoot, "..", "outside")]) {
    assert.throws(() => launchFor(outside), error => error.code === "PARAMETER_INVALID");
  }
});
const saved = [
  { name: "interview", type: "ME", format: "me-pytorch", files: ["metadata.json", "me.pt"], config: { resolution: 128, batch_size: 4 } },
  { name: "other-family", type: "SAEHD", files: ["other-family_SAEHD_data.dat"] },
];
const fullTraining = {
  id: "train.me", category: "training", parameters: [
    ...ME_CONFIG_PARAMETERS,
    ...["forceModelName", "initializeFrom", "pretrainingDataDir"].map(id => ({ id, type: "text", default: "" })),
    ...["allowConfigChange", "resetOptimizer", "resetDataState", "silentStart", "cpuOnly"].map(id => ({ id, type: "boolean", default: false })),
    { id: "gpuIndexes", type: "text", default: "" },
  ],
};

test("model choices match fixed command families and exclude incomplete or synthetic entries", () => {
  for (const item of commands.filter(item => item.parameters.some(parameter => parameter.id === "forceModelName"))) {
    assert.ok(commandModelFamily(item), `${item.id} needs a model family`);
  }
  assert.equal(commandModelFamily(command("export.dfm_me")), "ME");
  assert.equal(commandModelFamily(command("merge.preview_me")), "ME");
  assert.deepEqual(taskModels(command("merge.preview_me"), saved), [saved[0]]);
  assert.equal(createTaskConfiguration(command("merge.preview_me"), {}, saved).parameters.forceModelName, saved[0].name);
  assert.deepEqual(taskModels(command("train.me"), [...saved,
    { name: "weights-only", type: "ME", files: ["weights-only_ME_encoder.npy"] },
  ]), [saved[0]]);
  assert.deepEqual(taskModels(command("xseg.train"), [{ name: "xseg_model", type: "XSeg" }]), []);
});

test("task defaults preserve explicit resume parameters and never carry unregistered arguments", () => {
  const result = createTaskConfiguration(command("train.me"), {
    forceModelName: "interview", cpuOnly: true, gpuIndexes: "0,1", targetIterations: 1234, executable: "outside.exe",
  }, saved);
  assert.equal(result.newModel, false);
  assert.equal(result.parameters.forceModelName, "interview");
  assert.equal(result.parameters.targetIterations, 1234);
  assert.equal(result.parameters.cpuOnly, true);
  assert.equal(result.parameters.gpuIndexes, "");
  assert.equal(Object.hasOwn(result.parameters, "executable"), false);
  const unique = createTaskConfiguration(command("train.me"), null, saved);
  assert.equal(unique.parameters.forceModelName, "interview");
  assert.equal(unique.parameters.silentStart, true);
  assert.equal(createTaskConfiguration(command("train.me"), { forceModelName: "" }, saved).parameters.forceModelName, "");
  const multiple = createTaskConfiguration(command("train.me"), {}, [...saved, { ...saved[0], name: "second" }]);
  assert.equal(multiple.parameters.forceModelName, "");
});

test("new-model intent cannot silently continue a same-name model on Windows", () => {
  const fresh = createTaskConfiguration(command("train.me"), {}, []);
  assert.equal(fresh.newModel, true);
  assert.equal(fresh.parameters.silentStart, false);
  assert.equal(modelNameIssue("", true, saved), "empty");
  assert.equal(modelNameIssue("INTERVIEW", true, saved), "duplicate");
  assert.equal(modelNameIssue("../interview", true, saved), "invalid");
  assert.equal(modelNameIssue(" interview ", true, saved), "invalid");
  assert.equal(modelNameIssue("访谈第二版", true, saved), null);
  assert.equal(modelNameIssue("interview", false, saved), null);
});

test("task search counts true matches and combines trimmed case-insensitive terms", () => {
  assert.deepEqual(searchTaskCommands(commands, "  "), commands);
  assert.equal(searchTaskCommands(commands, "a-task-that-does-not-exist").length, 0);
  const found = searchTaskCommands(commands, "  ME    训练 ");
  assert.ok(found.some(item => item.id === "train.me"));
  assert.equal(found.some(item => item.id === "merge.me"), false);
});

test("output summaries follow the actual fixed workflow destinations", () => {
  assert.deepEqual(taskOutputLocations(command("train.me"), {}, { forceModelName: "interview" }).paths, ["model/interview"]);
  assert.deepEqual(taskOutputLocations(command("xseg.train")).paths, ["xseg_model"]);
  assert.deepEqual(taskOutputLocations(command("dst.denoise_frames")).paths, ["data_dst"]);
  assert.deepEqual(taskOutputLocations(command("merge.me")).paths, ["data_dst/merged", "data_dst/merged_mask"]);
  assert.deepEqual(taskOutputLocations(command("encode.mov_lossless")).paths, ["result.mov", "result_mask.mov"]);
  assert.deepEqual(taskOutputLocations(command("src.faces_resize")).paths, ["data_src/aligned_resized"]);
  assert.deepEqual(taskOutputLocations(command("src.faces_enhance")), {
    paths: ["data_src/aligned_enhanced/<独立批次>"], note: "independent-enhancement",
  });
  assert.deepEqual(taskOutputLocations(command("xseg.dst_fetch_labels")).paths, ["data_dst/aligned_xseg"]);
  assert.deepEqual(taskOutputLocations(command("video.cut_src"), { materials: { src: { name: "data_src.MP4" } } }).paths, ["data_src_cut.MP4"]);
  assert.equal(taskOutputLocations(command("runtime.prepare_vision")).note, "runtime");
  assert.equal(displayWorkspacePath("C:\\DFL\\workspace\\", "data_src/aligned"), "C:\\DFL\\workspace\\data_src\\aligned");
  assert.equal(displayWorkspacePath("/home/dfl/workspace/", "xseg_model"), "/home/dfl/workspace/xseg_model");
});

test("ME resume hydrates the entire checkpoint configuration including legacy defaults", () => {
  const model = { ...saved[0], config: { archi: "df-tc", resolution: 192, batch_size: 6, ae_dims: 512,
    face_type: "wf", use_rg: true, gan_power: 0.2, random_warp: false, lr_dropout: "cpu" } };
  const result = createTaskConfiguration(fullTraining, {}, [model]);
  for (const schema of ME_CONFIG_PARAMETERS) {
    assert.equal(result.parameters[schema.id], model.config[schema.configKey] ?? schema.default, schema.id);
  }
  assert.equal(result.parameters.silentStart, true);
  assert.equal(result.parameters.use_fp16, ME_DEFAULT_CONFIG.use_fp16);
  assert.equal(result.newModel, false);
});

test("ME deliberate resume overrides survive hydration and structural mismatches remain visible", () => {
  const model = { ...saved[0], config: { archi: "df-ud", resolution: 192, batch_size: 6, lr: 0.00008, use_rg: true } };
  const supplied = { forceModelName: model.name, allowConfigChange: true, batchSize: 2, resolution: 128,
    lr: 0.00001, use_rg: false, resetOptimizer: true, ct_mode: "rct" };
  const result = createTaskConfiguration(fullTraining, supplied, [model]);
  for (const [key, value] of Object.entries(supplied)) assert.equal(result.parameters[key], value, key);
  assert.equal(result.parameters.archi, "df-ud");
  assert.equal(result.parameters.use_fp16, false);
});

test("selecting another ME model restores all settings and clears phase/reset/source flags", () => {
  const model = { ...saved[0], name: "second", config: { archi: "df", resolution: 96, batch_size: 8, use_fp16: true } };
  const current = createTaskConfiguration(fullTraining, { forceModelName: "first", archi: "liae-t", allowConfigChange: true,
    resetOptimizer: true, resetDataState: true, initializeFrom: "interview", pretrainingDataDir: "pretrain/faces", pretrain: true }, []).parameters;
  const result = selectTaskModel(fullTraining, current, model.name, [model]);
  assert.equal(result.newModel, false);
  assert.equal(result.parameters.archi, "df");
  assert.equal(result.parameters.batchSize, 8);
  assert.equal(result.parameters.use_fp16, true);
  for (const key of ["allowConfigChange", "resetOptimizer", "resetDataState", "pretrain"]) assert.equal(result.parameters[key], false, key);
  assert.equal(result.parameters.initializeFrom, "");
  assert.equal(result.parameters.pretrainingDataDir, "");
  const fresh = selectTaskModel(fullTraining, result.parameters, "__new__", [model]);
  assert.equal(fresh.newModel, true);
  assert.equal(fresh.parameters.forceModelName, "");
  assert.equal(fresh.parameters.archi, ME_DEFAULT_CONFIG.archi);
  assert.equal(fresh.parameters.use_fp16, false);
});

test("native weight initialization hydrates source structure and starts a new phase", () => {
  const model = { ...saved[0], config: { archi: "df-dtc", resolution: 192, ae_dims: 320, batch_size: 7, pretrain: true, use_rg: true } };
  const result = createTaskConfiguration(fullTraining, { forceModelName: "derived", initializeFrom: model.name, lr: 0.00001 }, [model]);
  assert.equal(result.newModel, true);
  assert.equal(result.parameters.archi, "df-dtc");
  assert.equal(result.parameters.ae_dims, 320);
  assert.equal(result.parameters.batchSize, 7);
  assert.equal(result.parameters.pretrain, false);
  assert.equal(result.parameters.lr, 0.00001);
  assert.equal(restoreMEModelConfiguration(fullTraining, result.parameters, model).initializeFrom, "");
});

test("TF importer always creates a separate model and never auto-selects a checkpoint", () => {
  const importer = { id: "model.import_me_tf", category: "model", parameters: [
    ...ME_CONFIG_PARAMETERS.filter(schema => schema.structural),
    { id: "forceModelName", default: "", type: "text" },
  ] };
  const result = createTaskConfiguration(importer, {}, saved);
  assert.equal(commandModelFamily(importer), "ME");
  assert.equal(result.newModel, true);
  assert.equal(result.parameters.forceModelName, "");
  assert.equal(modelNameIssue("INTERVIEW", result.newModel, saved), "duplicate");
  assert.deepEqual(taskOutputLocations(importer, {}, { forceModelName: "converted" }).paths, ["model/converted"]);
  assert.equal(selectTaskModel(importer, { forceModelName: "converted" }, "interview", saved).parameters.forceModelName, "converted");
});
