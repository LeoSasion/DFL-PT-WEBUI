import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile, realpath, readdir } from "node:fs/promises";
import path from "node:path";
import { PATHS, assertWithin, pathExists } from "./paths.mjs";

const JOB_ID = /^[a-z0-9][a-z0-9-]{5,63}$/i;
const HASH = /^[a-f0-9]{64}$/;
const IMAGE_NAME = /^[^/\\:\u0000-\u001f]+\.(?:png|jpe?g|tiff?)$/i;

export class MergePreviewError extends Error {
  constructor(message, code = "MERGE_PREVIEW_INVALID", status = 409) {
    super(message); this.code = code; this.status = status;
  }
}

export function mergePreviewDirectory(jobId) {
  if (!JOB_ID.test(jobId ?? "")) throw new MergePreviewError("小样任务标识不合法", "MERGE_PREVIEW_ID_INVALID", 400);
  return assertWithin(PATHS.runtimeRoot, path.join(PATHS.runtimeRoot, "merge-previews", jobId), "合成小样目录");
}

export async function assertMergePreviewLocation(jobId) {
  const directory = mergePreviewDirectory(jobId);
  const workspaceReal = await realpath(PATHS.workspaceRoot);
  for (const candidate of [PATHS.runtimeRoot, path.dirname(directory), directory, path.join(directory, "merged"), path.join(directory, "merged_mask")]) {
    if (!(await pathExists(candidate))) continue;
    const info = await lstat(candidate);
    if (!info.isDirectory() || info.isSymbolicLink()) {
      throw new MergePreviewError("小样目录必须是当前项目中的普通受管目录");
    }
    try { assertWithin(workspaceReal, await realpath(candidate), "小样目录"); }
    catch { throw new MergePreviewError("小样目录超出当前项目"); }
  }
  return directory;
}

async function verifyImage(directory, name, expectedHash) {
  if (!IMAGE_NAME.test(name ?? "") || !HASH.test(expectedHash ?? "")) throw new MergePreviewError("小样回执成员或哈希无效");
  return verifyFile(assertWithin(directory, path.join(directory, name), "小样图像"), expectedHash);
}

async function verifyFile(target, expectedHash) {
  if (!HASH.test(expectedHash ?? "")) throw new MergePreviewError("小样回执哈希无效");
  const info = await lstat(target).catch(() => null);
  if (!info?.isFile() || info.isSymbolicLink()) throw new MergePreviewError("小样图像缺失或路径已变化");
  try { assertWithin(await realpath(PATHS.workspaceRoot), await realpath(target), "小样成员"); }
  catch { throw new MergePreviewError("小样成员超出当前项目"); }
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(target)) hash.update(chunk);
  if (hash.digest("hex") !== expectedHash) throw new MergePreviewError("小样素材或结果已变化，请重新生成独立小样", "MERGE_PREVIEW_SOURCE_CHANGED");
  return target;
}

async function loadAudit(jobId) {
  const directory = await assertMergePreviewLocation(jobId);
  const auditPath = path.join(directory, "merged", "merge.audit.json");
  const info = await lstat(auditPath).catch(() => null);
  if (!info?.isFile() || info.isSymbolicLink()) throw new MergePreviewError("小样尚未生成可读取回执", "MERGE_PREVIEW_PENDING", 404);
  if (info.size > 2 * 1024 * 1024) throw new MergePreviewError("小样回执过大");
  let audit;
  try { audit = JSON.parse(await readFile(auditPath, "utf8")); } catch { throw new MergePreviewError("小样回执无法读取"); }
  const preview = audit?.preview;
  if (audit.kind !== "merge-quality-audit" || preview?.kind !== "bounded-independent-preview"
    || !Number.isInteger(preview.frameStart) || preview.frameStart < 1
    || !Number.isInteger(preview.frameCount) || preview.frameCount < 1 || preview.frameCount > 20
    || !Array.isArray(audit.frames) || audit.frames.length !== preview.frameCount
    || !["prepared", "complete", "incomplete", "failed"].includes(audit.status)) throw new MergePreviewError("小样回执范围或状态无效");
  return { directory, audit };
}

export async function inspectMergePreview(jobId) {
  const { directory, audit } = await loadAudit(jobId);
  const entries = await readdir(path.join(PATHS.workspaceRoot, "data_dst"), { withFileTypes: true });
  // Python pathex uses Unicode code-point lexical order, rather than numeric or
  // locale sorting; preserve the same explicit frame window across both layers.
  const compareNames = (left, right) => {
    const a = [...left].map(char => char.codePointAt(0)), b = [...right].map(char => char.codePointAt(0));
    for (let index = 0; index < Math.min(a.length, b.length); index++) if (a[index] !== b[index]) return a[index] - b[index];
    return a.length - b.length;
  };
  const names = entries.filter(entry => entry.isFile() && IMAGE_NAME.test(entry.name)).map(entry => entry.name).sort(compareNames);
  const inventoryHash = createHash("sha256").update(names.join("\n"), "utf8").digest("hex");
  if (names.length !== audit.preview.totalSourceFrames || inventoryHash !== audit.preview.sourceFrameInventorySha256
      || names.slice(audit.preview.frameStart - 1, audit.preview.frameStart - 1 + audit.preview.frameCount).some((name, index) => name !== audit.frames[index]?.file)) throw new MergePreviewError("DST 帧清单或小样范围已变化，请重新生成小样", "MERGE_PREVIEW_SOURCE_CHANGED");
  if (typeof audit.preview.modelName !== "string" || /[/\\:*?"<>|\u0000-\u001f]/.test(audit.preview.modelName) || !audit.preview.modelName.trim()) throw new MergePreviewError("小样模型记录无效");
  await verifyFile(assertWithin(PATHS.workspaceRoot, path.join(PATHS.workspaceRoot, "model", audit.preview.modelName, "me.pt")), audit.preview.modelSha256);
  if (typeof audit.preview.alignedPath !== "string") throw new MergePreviewError("小样人脸集记录无效");
  const alignedPath = assertWithin(PATHS.workspaceRoot, path.resolve(PATHS.workspaceRoot, audit.preview.alignedPath), "小样人脸集");
  const items = [];
  for (const frame of audit.frames) {
    await verifyImage(path.join(PATHS.workspaceRoot, "data_dst"), frame.file, frame.sourceSha256);
    for (const alignment of frame.alignments ?? []) {
      if (alignment.storage !== "plain") throw new MergePreviewError("小样必须使用解包后的人脸集");
      await verifyImage(alignedPath, alignment.file, alignment.sha256);
    }
    const resultName = `${path.parse(frame.file).name}.png`;
    const complete = HASH.test(frame.outputSha256 ?? "") && HASH.test(frame.outputMaskSha256 ?? "");
    if (complete) {
      await verifyImage(path.join(directory, "merged"), resultName, frame.outputSha256);
      await verifyImage(path.join(directory, "merged_mask"), resultName, frame.outputMaskSha256);
    } else if (audit.status === "complete") throw new MergePreviewError("完整小样回执缺少结果哈希");
    const base = `/api/tools/merge-preview/${encodeURIComponent(jobId)}`;
    items.push({ name: frame.file, resultName, complete, sourceUrl: `${base}/source/${encodeURIComponent(frame.file)}`,
      mergedUrl: complete ? `${base}/merged/${encodeURIComponent(resultName)}` : null,
      maskUrl: complete ? `${base}/mask/${encodeURIComponent(resultName)}` : null });
  }
  return { jobId, status: audit.status, preview: audit.preview, configuration: audit.configuration,
    outputLocation: `.webui/merge-previews/${jobId}/`, completeCount: items.filter(item => item.complete).length, items };
}

export async function resolveMergePreviewAsset(jobId, slot, encodedName) {
  let name;
  try { name = decodeURIComponent(encodedName); } catch { throw new MergePreviewError("图像名称编码无效", "MERGE_PREVIEW_NAME_INVALID", 400); }
  if (!IMAGE_NAME.test(name) || !["source", "merged", "mask"].includes(slot)) throw new MergePreviewError("小样图像请求无效", "MERGE_PREVIEW_NAME_INVALID", 400);
  const { directory, audit } = await loadAudit(jobId);
  const frame = audit.frames.find(item => slot === "source" ? item.file === name : `${path.parse(item.file).name}.png` === name);
  if (!frame) throw new MergePreviewError("图像不属于此小样回执", "MERGE_PREVIEW_MEMBER_MISSING", 404);
  const target = slot === "source" ? assertWithin(path.join(PATHS.workspaceRoot, "data_dst"), path.join(PATHS.workspaceRoot, "data_dst", name))
    : assertWithin(directory, path.join(directory, slot === "mask" ? "merged_mask" : "merged", name));
  const expectedHash = slot === "source" ? frame.sourceSha256 : slot === "mask" ? frame.outputMaskSha256 : frame.outputSha256;
  const info = await lstat(target).catch(() => null);
  if (!info?.isFile() || info.isSymbolicLink() || info.size > 32 * 1024 * 1024) throw new MergePreviewError("小样图像缺失、过大或路径已变化");
  try { assertWithin(await realpath(PATHS.workspaceRoot), await realpath(target), "小样图像"); }
  catch { throw new MergePreviewError("小样图像超出当前项目"); }
  const bytes = await readFile(target);
  if (createHash("sha256").update(bytes).digest("hex") !== expectedHash) throw new MergePreviewError("小样图像已变化，请重新生成", "MERGE_PREVIEW_SOURCE_CHANGED");
  // Serve exactly the verified bytes rather than reopening a path after hashing.
  return { bytes, path: target, name, mimeType: /\.png$/i.test(name) ? "image/png" : /\.jpe?g$/i.test(name) ? "image/jpeg" : "image/tiff" };
}
