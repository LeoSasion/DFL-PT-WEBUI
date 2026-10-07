import { createHash, randomBytes } from "node:crypto";
import { copyFile, lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { PATHS, assertWithin, pathExists } from "./paths.mjs";
import { buildDflEnvironment } from "./environment.mjs";
import { runHelperProcess } from "./helper-process.mjs";

export const MASK_ASSIST_PROFILE = "bisenet-face-preserve-v1";
const ID = /^mask-[a-f0-9]{24}$/;
const JPEG = /^[^<>:"/\\|?*\u0000-\u001f]{1,220}\.(?:jpe?g)$/i;
const IMAGES = /^[^<>:"/\\|?*\u0000-\u001f]{1,220}\.(?:jpe?g|png)$/i;
const HASH = /^[a-f0-9]{64}$/;

export class MaskAssistError extends Error {
  constructor(message, code = "MASK_ASSIST_ERROR", status = 400) {
    super(message); this.name = "MaskAssistError"; this.code = code; this.status = status;
  }
}

function roots(side) {
  if (!["src", "dst"].includes(side)) throw new MaskAssistError("数据集类型无效", "SIDE_INVALID");
  return {
    aligned: path.join(PATHS.workspaceRoot, `data_${side}`, "aligned"),
    drafts: path.join(PATHS.runtimeRoot, "mask-assist", side),
    published: path.join(PATHS.workspaceRoot, `data_${side}`, "aligned_assisted"),
  };
}

function draftPath(side, id) {
  if (!ID.test(id)) throw new MaskAssistError("辅助遮罩记录无效", "MASK_DRAFT_INVALID");
  return path.join(roots(side).drafts, id);
}

function checkAbort(signal) {
  if (signal?.aborted) throw Object.assign(new Error("辅助遮罩操作已取消"), { name: "AbortError", code: "ABORT_ERR" });
}

async function plainPath(target, { directory = false, create = false } = {}) {
  assertWithin(PATHS.workspaceRoot, target, "辅助遮罩路径");
  const relative = path.relative(PATHS.workspaceRoot, target);
  const components = [PATHS.workspaceRoot];
  for (const component of relative.split(path.sep).filter(Boolean)) components.push(path.join(components.at(-1), component));
  for (let index = 0; index < components.length; index += 1) {
    const component = components[index];
    let info;
    try { info = await lstat(component); } catch (error) {
      if (error.code !== "ENOENT" || !create || index === 0) throw error;
      await mkdir(component);
      info = await lstat(component);
    }
    const wantsDirectory = index < components.length - 1 || directory;
    if (info.isSymbolicLink() || (wantsDirectory ? !info.isDirectory() : !info.isFile())) {
      throw new MaskAssistError("辅助遮罩路径不是安全的普通文件或目录", "MASK_PATH_UNSAFE");
    }
  }
  assertWithin(await realpath(PATHS.workspaceRoot), await realpath(target), "辅助遮罩真实路径");
  return target;
}

async function digest(target) { return createHash("sha256").update(await readFile(target)).digest("hex"); }

function validateNames(names) {
  if (!Array.isArray(names) || names.length < 1 || names.length > 500
      || names.some((name) => typeof name !== "string" || !JPEG.test(name))
      || new Set(names.map((name) => name.toLocaleLowerCase("en-US"))).size !== names.length) {
    throw new MaskAssistError("每批选择 1–500 张不同的 DFL WF aligned JPEG", "MASK_NAMES_INVALID");
  }
  return names;
}

async function helper(aligned, output, names, device, { signal, onProgress, runProcess } = {}) {
  const args = [path.join(PATHS.webuiRoot, "python", "mask_assist.py"), "--input-root", aligned, "--output", output,
    "--assets", path.join(PATHS.repositoryRoot, "workspace", ".vision-models", "masks"), "--device", device];
  await runPython(args, { signal, onProgress, runProcess, input: { names } });
}

// Internal fixed-runtime runner, also exercised with a stdlib-only process fixture.
export async function runPython(args, { signal, onProgress, input = {}, timeoutMs = 30 * 60_000,
  maxResponseBytes = 1024 * 1024, runProcess = runHelperProcess } = {}) {
  checkAbort(signal);
  let pending = "", result;
  try {
    result = await runProcess(PATHS.python, args, { cwd: PATHS.repositoryRoot, env: buildDflEnvironment("pytorch"),
      signal, input, timeoutMs, maxBytes: maxResponseBytes, label: "辅助遮罩",
      onStdout: chunk => {
        pending += chunk.toString("utf8");
        const lines = pending.split(/\r?\n/); pending = lines.pop();
        if (pending.length > 64_000 || lines.some(line => line.length > 64_000))
          throw new MaskAssistError("辅助遮罩响应超出限制", "MASK_RESPONSE_TOO_LARGE", 422);
        for (const line of lines) {
          try {
            const progress = JSON.parse(line)?.progress;
            if (progress) onProgress?.({ stage: "生成辅助遮罩", current: progress.completed, total: progress.total, detail: "保留原图，写入新副本" });
          } catch { /* Ignore non-protocol model diagnostics. */ }
        }
      },
    });
  } catch (error) {
    if (error.code === "HELPER_STOP_UNCONFIRMED") throw new MaskAssistError(error.message, "MASK_STOP_UNCONFIRMED", 409);
    if (error.code === "OPERATION_CANCELLED") throw Object.assign(new Error("辅助遮罩操作已取消"), { name: "AbortError", code: "ABORT_ERR" });
    if (error.code === "HELPER_TIMEOUT") throw new MaskAssistError("辅助遮罩生成超时", "MASK_INFERENCE_TIMEOUT", 408);
    if (error.code === "HELPER_OUTPUT_TOO_LARGE") throw new MaskAssistError("辅助遮罩响应超出限制", "MASK_RESPONSE_TOO_LARGE", 422);
    throw error;
  }
  if (result.code !== 0) throw new MaskAssistError(`辅助遮罩生成失败：${result.stderr.slice(-4000)}`, "MASK_INFERENCE_FAILED", 422);
}

export async function createMaskAssistDraft(side, { names, device = "cuda", signal, onProgress, runProcess } = {}) {
  validateNames(names);
  if (!["cuda", "cpu"].includes(device)) throw new MaskAssistError("推理设备无效", "MASK_DEVICE_INVALID");
  const root = roots(side);
  await plainPath(root.aligned, { directory: true });
  for (const name of names) await plainPath(path.join(root.aligned, name));
  await plainPath(root.drafts, { directory: true, create: true });
  const id = `mask-${randomBytes(12).toString("hex")}`;
  const staging = path.join(root.drafts, `.building-${id}`);
  try {
    await helper(root.aligned, staging, names, device, { signal, onProgress, runProcess });
    checkAbort(signal);
    const report = JSON.parse(await readFile(path.join(staging, "report.json"), "utf8"));
    Object.assign(report, { id, side, createdAt: new Date().toISOString(), selectedCount: names.length });
    await writeFile(path.join(staging, "report.json"), JSON.stringify(report, null, 2) + "\n");
    await rename(staging, draftPath(side, id));
    return await readMaskAssistDraft(side, id);
  } catch (error) {
    if (["MASK_STOP_UNCONFIRMED", "HELPER_STOP_UNCONFIRMED"].includes(error.code)) {
      error.details = { ...error.details, retainedStaging: staging };
      throw error;
    }
    if (await pathExists(staging)) { await plainPath(staging, { directory: true }); await rm(staging, { recursive: true, force: true }); }
    throw error;
  }
}

export async function readMaskAssistDraft(side, id) {
  const directory = draftPath(side, id);
  await plainPath(directory, { directory: true });
  const file = path.join(directory, "report.json");
  await plainPath(file);
  const info = await lstat(file);
  if (info.size > 16 * 1024 * 1024) throw new MaskAssistError("辅助遮罩记录过大", "MASK_DRAFT_INVALID");
  const report = JSON.parse(await readFile(file, "utf8"));
  if (report.schemaVersion !== 1 || report.profile !== MASK_ASSIST_PROFILE || report.status !== "ready" || report.id !== id || report.side !== side) {
    throw new MaskAssistError("辅助遮罩记录格式无效", "MASK_DRAFT_INVALID");
  }
  validateNames(report.entries?.map((entry) => entry.file));
  if (report.entries.some((entry) => !HASH.test(entry.inputSha256) || !HASH.test(entry.copySha256))) throw new MaskAssistError("辅助遮罩哈希无效", "MASK_DRAFT_INVALID");
  return { ...report, publication: await inspectMaskAssistPublication(side, id) };
}

export async function inspectMaskAssistPublication(side, id) {
  draftPath(side, id);
  const directory = path.join(roots(side).published, id);
  if (!(await pathExists(directory))) return null;
  await plainPath(directory, { directory: true });
  const marker = path.join(directory, "mask-assist-provenance.json");
  await plainPath(marker);
  if ((await lstat(marker)).size > 32 * 1024 * 1024) throw new MaskAssistError("副本发布记录过大", "MASK_PUBLICATION_INVALID");
  const receipt = JSON.parse(await readFile(marker, "utf8"));
  if (receipt.schemaVersion !== 1 || receipt.id !== id || receipt.side !== side || receipt.profile !== MASK_ASSIST_PROFILE
      || receipt.status !== "published" || receipt.facesetPath !== directory || !Number.isInteger(receipt.copiedCount)
      || receipt.copiedCount < 1 || !Number.isInteger(receipt.assistedCount) || receipt.assistedCount < 1) {
    throw new MaskAssistError("副本发布记录无效", "MASK_PUBLICATION_INVALID");
  }
  return receipt;
}

export async function listMaskAssistDrafts(side) {
  const root = roots(side).drafts;
  if (!(await pathExists(root))) return [];
  await plainPath(root, { directory: true });
  const drafts = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!ID.test(entry.name) || !entry.isDirectory()) continue;
    const report = await readMaskAssistDraft(side, entry.name);
    drafts.push({ id: report.id, side, profile: report.profile, status: report.publication ? "published" : report.status,
      selectedCount: report.entries.length, createdAt: report.createdAt, facesetPath: report.publication?.facesetPath ?? null });
  }
  return drafts.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function resolveMaskAssistFile(side, id, entryIndex, kind) {
  const report = await readMaskAssistDraft(side, id);
  if (!Number.isInteger(entryIndex) || entryIndex < 0 || entryIndex >= report.entries.length || !["original", "copy", "overlay", "mask"].includes(kind)) {
    throw new MaskAssistError("辅助遮罩预览无效", "MASK_PREVIEW_INVALID");
  }
  const entry = report.entries[entryIndex];
  const relative = kind === "original" ? path.join("original", entry.file) : kind === "copy" ? path.join("copies", entry.file)
    : path.join(kind === "overlay" ? "overlay" : "masks", `${String(entryIndex).padStart(4, "0")}.png`);
  const file = path.join(draftPath(side, id), relative);
  await plainPath(file);
  const expected = entry[kind === "original" ? "inputSha256" : kind === "copy" ? "copySha256" : `${kind}Sha256`];
  if (!HASH.test(expected) || await digest(file) !== expected) throw new MaskAssistError("预览文件已改变", "MASK_DRAFT_CHANGED", 409);
  return file;
}

export async function publishMaskAssistCopies(side, id, { reviewed = false, signal, onProgress, runProcess = runHelperProcess } = {}) {
  if (reviewed !== true) throw new MaskAssistError("请先复查遮挡、眼睛与嘴部预览", "MASK_REVIEW_REQUIRED", 409);
  checkAbort(signal);
  const report = await readMaskAssistDraft(side, id);
  if (report.publication) return { ...report.publication, reused: true };
  const root = roots(side);
  await plainPath(root.aligned, { directory: true });
  await plainPath(root.published, { directory: true, create: true });
  const target = path.join(root.published, id);
  if (await pathExists(target)) throw new MaskAssistError("副本目标已存在，请复查发布记录", "MASK_ALREADY_PUBLISHED", 409);
  const entries = (await readdir(root.aligned, { withFileTypes: true })).filter((entry) => IMAGES.test(entry.name));
  if (!entries.length || entries.some((entry) => !entry.isFile() || entry.isSymbolicLink())) throw new MaskAssistError("aligned 数据集为空或包含不安全图片", "MASK_DATASET_INVALID");
  const selected = new Map(report.entries.map((entry) => [entry.file.toLocaleLowerCase("en-US"), entry]));
  if (new Set(entries.map((entry) => entry.name.toLocaleLowerCase("en-US"))).size !== entries.length
      || report.entries.some((entry) => !entries.some((source) => source.name === entry.file))) throw new MaskAssistError("原数据集已变化", "MASK_SOURCE_CHANGED", 409);
  const staging = path.join(root.published, `.building-${id}-${randomBytes(5).toString("hex")}`);
  await mkdir(staging);
  const copied = [];
  try {
    const bindingResult = await runProcess(PATHS.python, [path.join(PATHS.webuiRoot, "python", "mask_source_binding.py"),
      "--aligned-root", path.join(draftPath(side, id), "copies"), "--frames-root", path.join(PATHS.workspaceRoot, `data_${side}`)], {
      cwd: PATHS.repositoryRoot, env: buildDflEnvironment("pytorch"), input: { names: report.entries.map(entry => entry.file) }, signal,
      label: "复核遮罩来源校验", timeoutMs: 120_000,
    });
    if (bindingResult.code !== 0) throw new MaskAssistError(bindingResult.stderr.trim() || "复核遮罩来源校验失败", "MASK_SOURCE_BINDING_FAILED", 409);
    const bindings = new Map(JSON.parse(bindingResult.stdout).map(entry => [entry.file, entry]));
    for (const [index, entry] of entries.entries()) {
      checkAbort(signal);
      const source = path.join(root.aligned, entry.name);
      await plainPath(source);
      const sourceHash = await digest(source);
      const assisted = selected.get(entry.name.toLocaleLowerCase("en-US"));
      if (assisted && sourceHash !== assisted.inputSha256) throw new MaskAssistError("已复查原图发生变化，请重新生成", "MASK_SOURCE_CHANGED", 409);
      const copySource = assisted ? await resolveMaskAssistFile(side, id, report.entries.indexOf(assisted), "copy") : source;
      const destination = path.join(staging, entry.name);
      await copyFile(copySource, destination);
      if (await digest(destination) !== (assisted?.copySha256 ?? sourceHash) || await digest(source) !== sourceHash) throw new MaskAssistError("复制时数据集发生变化", "MASK_SOURCE_CHANGED", 409);
      const binding = assisted ? bindings.get(entry.name) : null;
      copied.push({ file: entry.name, sourceSha256: sourceHash, outputSha256: assisted?.copySha256 ?? sourceHash, assisted: Boolean(assisted),
        ...(assisted ? { sourceFrame: binding?.sourceFrame ?? null, sourceBindingUnavailable: binding?.reason ?? null } : {}) });
      onProgress?.({ stage: "发布完整副本数据集", current: index + 1, total: entries.length, detail: "原 aligned 保留，未修改图片独立复制" });
    }
    checkAbort(signal);
    const currentNames = (await readdir(root.aligned)).filter((name) => IMAGES.test(name)).sort();
    if (JSON.stringify(currentNames) !== JSON.stringify(copied.map((entry) => entry.file).sort())) throw new MaskAssistError("原数据集成员已变化", "MASK_SOURCE_CHANGED", 409);
    for (const entry of copied) {
      checkAbort(signal);
      await plainPath(path.join(root.aligned, entry.file));
      if (await digest(path.join(root.aligned, entry.file)) !== entry.sourceSha256) throw new MaskAssistError("原数据集内容已变化", "MASK_SOURCE_CHANGED", 409);
      if (entry.sourceFrame) {
        const frame = path.join(PATHS.workspaceRoot, `data_${side}`, entry.sourceFrame.file);
        await plainPath(frame);
        if (await digest(frame) !== entry.sourceFrame.sha256) throw new MaskAssistError("原始帧在发布时发生变化", "MASK_SOURCE_CHANGED", 409);
      }
    }
    const receipt = { schemaVersion: 1, id, side, profile: MASK_ASSIST_PROFILE, status: "published", reviewed: true,
      facesetPath: target, copiedCount: copied.length, assistedCount: report.entries.length, subset: false, publishedAt: new Date().toISOString(),
      originalRetained: true, independentCopies: true, embeddedMaskCompatible: true, semantics: report.semantics, entries: copied,
      reviewedMergeAvailable: copied.every(entry => entry.assisted && entry.sourceFrame),
      reviewedMergeReason: copied.every(entry => entry.assisted && entry.sourceFrame) ? null : "全部待合成人脸须经过复核并绑定原始帧；当前副本仍可用于训练" };
    await writeFile(path.join(staging, "mask-assist-provenance.json"), JSON.stringify(receipt, null, 2) + "\n");
    checkAbort(signal);
    await plainPath(root.published, { directory: true });
    await rename(staging, target);
    return receipt;
  } catch (error) {
    if (["MASK_STOP_UNCONFIRMED", "HELPER_STOP_UNCONFIRMED"].includes(error.code)) {
      error.details = { ...error.details, retainedStaging: staging };
      throw error;
    }
    await plainPath(staging, { directory: true }); await rm(staging, { recursive: true, force: true }); throw error;
  }
}
