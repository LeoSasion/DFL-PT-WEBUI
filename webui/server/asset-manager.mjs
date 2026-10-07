import {
  copyFile,
  link,
  lstat,
  mkdir,
  open,
  readdir,
  realpath,
  rename,
  stat,
  unlink,
} from "node:fs/promises";
import { createHash, randomBytes } from "node:crypto";
import { runHelperProcess } from "./helper-process.mjs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { buildDflEnvironment } from "./environment.mjs";
import { PATHS, assertWithin, pathExists } from "./paths.mjs";

const IMAGE_NAME = /^[^<>:"/\\|?*\u0000-\u001f]{1,220}\.(?:jpe?g|png)$/i;
const QUARANTINE_TOKEN = /^\d{14}-[a-f0-9]{10}$/;
const MAX_HELPER_OUTPUT = 4 * 1024 * 1024;
const ANALYSIS_CACHE_TTL_MS = 30_000;
const MAX_ANALYSIS_CACHE_ENTRIES = 64;
const analysisCache = new Map();

export class AssetError extends Error {
  constructor(message, code = "ASSET_ERROR", status = 400, details) {
    super(message);
    this.name = "AssetError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function alignedDirectory(side) {
  if (!["src", "dst"].includes(side)) {
    throw new AssetError("数据集类型不受支持", "SIDE_INVALID");
  }
  return path.join(PATHS.workspaceRoot, `data_${side}`, "aligned");
}

function quarantineDirectory(side) {
  alignedDirectory(side);
  return assertWithin(
    PATHS.runtimeRoot,
    path.join(PATHS.runtimeRoot, "quarantine", side),
    "隔离目录",
  );
}

function decodeImageName(encodedName) {
  let name;
  try {
    name = decodeURIComponent(encodedName);
  } catch {
    throw new AssetError("图片文件名无效", "IMAGE_NAME_INVALID");
  }
  if (!IMAGE_NAME.test(name) || path.basename(name) !== name) {
    throw new AssetError("图片文件名不在允许范围内", "IMAGE_NAME_INVALID");
  }
  return name;
}

function validateQuarantineToken(token) {
  if (!QUARANTINE_TOKEN.test(token)) {
    throw new AssetError("隔离记录无效", "QUARANTINE_TOKEN_INVALID");
  }
  return token;
}

async function lstatIfPresent(target, options) {
  try {
    return await lstat(target, options);
  } catch (error) {
    if (error?.code === "ENOENT" || error?.code === "ENOTDIR") return null;
    throw error;
  }
}

async function verifyPlainDirectory(target, label, code = "QUARANTINE_PATH_UNSAFE") {
  const info = await lstatIfPresent(target);
  if (!info) {
    throw new AssetError(`${label}不存在`, code, 404);
  }
  if (!info.isDirectory() || info.isSymbolicLink()) {
    throw new AssetError(`${label}不是安全的本地目录`, code, 400);
  }
}

async function verifyAlignedDirectory(side) {
  const directory = alignedDirectory(side);
  const directoryInfo = await lstatIfPresent(directory);
  if (!directoryInfo) return null;
  const dataDirectory = path.dirname(directory);
  await verifyPlainDirectory(dataDirectory, "数据集目录", "ALIGNED_PATH_UNSAFE");
  if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) {
    throw new AssetError("aligned 目录不是安全的本地目录", "ALIGNED_PATH_UNSAFE", 400);
  }
  const [workspaceRealPath, directoryRealPath] = await Promise.all([
    realpath(PATHS.workspaceRoot),
    realpath(directory),
  ]);
  try {
    assertWithin(workspaceRealPath, directoryRealPath, "aligned 目录");
  } catch {
    throw new AssetError("aligned 目录超出工作区", "ALIGNED_PATH_UNSAFE", 400);
  }
  return directory;
}

async function resolveExistingAlignedImage(side, encodedName) {
  const target = resolveAlignedImage(side, encodedName);
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    throw new AssetError("aligned 图片不存在", "IMAGE_MISSING", 404);
  }
  const fileInfo = await lstatIfPresent(target);
  if (!fileInfo) {
    throw new AssetError("aligned 图片不存在", "IMAGE_MISSING", 404);
  }
  if (!fileInfo.isFile() || fileInfo.isSymbolicLink()) {
    throw new AssetError("aligned 图片不是安全的本地图片", "ALIGNED_PATH_UNSAFE", 400);
  }
  const [directoryRealPath, targetRealPath] = await Promise.all([
    realpath(directory),
    realpath(target),
  ]);
  try {
    assertWithin(directoryRealPath, targetRealPath, "aligned 图片");
  } catch {
    throw new AssetError("aligned 图片超出工作区", "ALIGNED_PATH_UNSAFE", 400);
  }
  return target;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

async function openVerifiedImage(
  target,
  label,
  unsafeCode,
  { containmentRoot = path.dirname(target), boundaryRoot = PATHS.workspaceRoot } = {},
) {
  let fileHandle;
  try {
    fileHandle = await open(target, "r");
  } catch (error) {
    if (error?.code === "ENOENT" || error?.code === "ENOTDIR") {
      throw new AssetError(`${label}不存在`, "IMAGE_MISSING", 404);
    }
    throw error;
  }
  try {
    const openedInfo = await fileHandle.stat({ bigint: true });
    const [boundaryRealPath, containmentRealPath, targetRealPath] = await Promise.all([
      realpath(boundaryRoot),
      realpath(containmentRoot),
      realpath(target),
    ]);
    try {
      assertWithin(boundaryRealPath, containmentRealPath, label);
      assertWithin(containmentRealPath, targetRealPath, label);
    } catch {
      throw new AssetError(`${label}超出允许范围`, unsafeCode, 400);
    }
    // Check the path after canonicalization so an ancestor swap cannot redirect the opened handle.
    const pathInfo = await lstat(target, { bigint: true });
    if (
      !openedInfo.isFile()
      || !pathInfo.isFile()
      || pathInfo.isSymbolicLink()
      || !sameFileIdentity(openedInfo, pathInfo)
    ) {
      throw new AssetError(`${label}在读取前发生了不安全的路径变化`, unsafeCode, 400);
    }
    return { fileHandle, fileInfo: openedInfo };
  } catch (error) {
    await fileHandle.close().catch(() => {});
    if (error?.code === "ENOENT" || error?.code === "ENOTDIR") {
      throw new AssetError(`${label}不存在`, "IMAGE_MISSING", 404);
    }
    throw error;
  }
}

async function ensureSafeAlignedDirectory(side) {
  const directory = alignedDirectory(side);
  const dataDirectory = path.dirname(directory);
  const dataInfo = await lstatIfPresent(dataDirectory);
  if (dataInfo) {
    await verifyPlainDirectory(dataDirectory, "数据集目录", "ALIGNED_PATH_UNSAFE");
  } else {
    await mkdir(dataDirectory);
  }
  const alignedInfo = await lstatIfPresent(directory);
  if (alignedInfo) {
    await verifyPlainDirectory(directory, "aligned 目录", "ALIGNED_PATH_UNSAFE");
  } else {
    await mkdir(directory);
  }
  return verifyAlignedDirectory(side);
}

async function verifyQuarantineRoot(side) {
  const root = quarantineDirectory(side);
  if (!(await pathExists(root))) return null;
  const base = path.dirname(root);
  await verifyPlainDirectory(PATHS.runtimeRoot, "运行时目录");
  await verifyPlainDirectory(base, "隔离根目录");
  await verifyPlainDirectory(root, "数据集隔离目录");
  const [workspaceRealPath, runtimeRealPath, rootRealPath] = await Promise.all([
    realpath(PATHS.workspaceRoot),
    realpath(PATHS.runtimeRoot),
    realpath(root),
  ]);
  try {
    assertWithin(workspaceRealPath, runtimeRealPath, "运行时目录");
    assertWithin(runtimeRealPath, rootRealPath, "数据集隔离目录");
  } catch {
    throw new AssetError("隔离目录超出运行时目录", "QUARANTINE_PATH_UNSAFE", 400);
  }
  return root;
}

async function ensureQuarantineRoot(side) {
  const root = quarantineDirectory(side);
  const base = path.dirname(root);
  const runtimeInfo = await lstatIfPresent(PATHS.runtimeRoot);
  if (runtimeInfo) await verifyPlainDirectory(PATHS.runtimeRoot, "运行时目录");
  else await mkdir(PATHS.runtimeRoot);
  const baseInfo = await lstatIfPresent(base);
  if (baseInfo) await verifyPlainDirectory(base, "隔离根目录");
  else await mkdir(base);
  const rootInfo = await lstatIfPresent(root);
  if (rootInfo) await verifyPlainDirectory(root, "数据集隔离目录");
  else await mkdir(root);
  await verifyQuarantineRoot(side);
  return root;
}

async function createQuarantineTokenDirectory(side) {
  const root = await ensureQuarantineRoot(side);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const token = recoveryToken();
    const directory = assertWithin(root, path.join(root, token), "隔离记录目录");
    try {
      await mkdir(directory);
      return { token, directory };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }
  }
  throw new AssetError("无法创建唯一的隔离记录", "QUARANTINE_TOKEN_COLLISION", 500);
}

export function resolveAlignedImage(side, encodedName) {
  const name = decodeImageName(encodedName);
  return assertWithin(alignedDirectory(side), path.join(alignedDirectory(side), name), "aligned 图片");
}

export async function resolveQuarantinedImage(side, token, encodedName) {
  validateQuarantineToken(token);
  const name = decodeImageName(encodedName);
  const root = await verifyQuarantineRoot(side);
  if (!root) {
    throw new AssetError("隔离文件不存在", "QUARANTINE_MISSING", 404);
  }
  const tokenDirectory = assertWithin(root, path.join(root, token), "隔离记录目录");
  const target = assertWithin(tokenDirectory, path.join(tokenDirectory, name), "隔离文件");
  if (!(await pathExists(tokenDirectory)) || !(await pathExists(target))) {
    throw new AssetError("隔离文件不存在", "QUARANTINE_MISSING", 404);
  }
  await verifyPlainDirectory(tokenDirectory, "隔离记录目录");
  const fileInfo = await lstat(target);
  if (!fileInfo.isFile() || fileInfo.isSymbolicLink()) {
    throw new AssetError("隔离文件不是安全的本地图片", "QUARANTINE_PATH_UNSAFE", 400);
  }
  const [rootRealPath, tokenRealPath, targetRealPath] = await Promise.all([
    realpath(root),
    realpath(tokenDirectory),
    realpath(target),
  ]);
  try {
    assertWithin(rootRealPath, tokenRealPath, "隔离记录目录");
    assertWithin(tokenRealPath, targetRealPath, "隔离文件");
  } catch {
    throw new AssetError("隔离文件超出允许范围", "QUARANTINE_PATH_UNSAFE", 400);
  }
  return target;
}

async function runAssetHelper(args, input, { signal, onProgress, timeoutMs = 120_000 } = {}) {
  let pending = "";
  const progress = line => {
    if (line.startsWith("DFL_PROGRESS ")) {
      try { onProgress?.(JSON.parse(line.slice(13))); } catch { /* malformed progress is not a result */ }
    }
  };
  const result = await runHelperProcess(PATHS.python, [path.join(PATHS.webuiRoot, "python", "dfl_asset_tool.py"), ...args], {
    cwd: PATHS.currentDflRoot, env: buildDflEnvironment("current"), input, signal, timeoutMs,
    label: "DFL 分析", maxBytes: MAX_HELPER_OUTPUT,
    onStderr: chunk => { pending += chunk.toString("utf8"); const lines = pending.split(/\r?\n/); pending = lines.pop() || ""; lines.forEach(progress); },
  });
  if (pending) progress(pending);
  if (result.code !== 0) throw new AssetError(result.stderr.split(/\r?\n/).filter(line => !line.startsWith("DFL_PROGRESS ")).join("\n").trim() || "读取 DFL 元数据失败", "DFL_METADATA_FAILED", 422);
  try { return JSON.parse(result.stdout); } catch { throw new AssetError("DFL 分析响应不是有效 JSON", "DFL_METADATA_INVALID", 500); }
}

export async function listAlignedAssets(side, { offset = 0, limit = 60 } = {}) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    return { side, total: 0, offset: 0, limit, items: [] };
  }
  const result = await runAssetHelper([
    "list",
    "--directory",
    directory,
    "--offset",
    String(Math.max(Number(offset) || 0, 0)),
    "--limit",
    String(Math.min(Math.max(Number(limit) || 60, 1), 200)),
  ]);
  return {
    side,
    ...result,
    items: result.items.map((item) => ({
      ...item,
      imageUrl: `/api/assets/${side}/aligned/${encodeURIComponent(item.name)}`,
    })),
  };
}

export async function summarizeXSegLabels(side) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    return {
      side,
      total: 0,
      polygonCount: 0,
      appliedMaskCount: 0,
      usableLabelCount: 0,
      invalidCount: 0,
    };
  }
  return {
    side,
    ...await runAssetHelper(["xseg-label-summary", "--directory", directory]),
  };
}

async function cachedAnalysis(side, key, refresh, loader, { fingerprint } = {}) {
  const cacheKey = `${side}:${key}`;
  const cached = analysisCache.get(cacheKey);
  if (!refresh && cached && Date.now() - cached.createdAt < ANALYSIS_CACHE_TTL_MS
      && (fingerprint === undefined || cached.value.fingerprint === fingerprint)) {
    return { ...cached.value, cached: true };
  }
  const value = await loader();
  analysisCache.set(cacheKey, { createdAt: Date.now(), value });
  while (analysisCache.size > MAX_ANALYSIS_CACHE_ENTRIES) {
    analysisCache.delete(analysisCache.keys().next().value);
  }
  return { ...value, cached: false };
}

function invalidateAnalysis(side) {
  for (const key of analysisCache.keys()) {
    if (key.startsWith(`${side}:`)) analysisCache.delete(key);
  }
}

export async function buildAlignedPoseAtlas(side, { signal, onProgress } = {}) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    return {
      side,
      total: 0,
      validCount: 0,
      invalidCount: 0,
      lowQualityCount: 0,
      meanSharpness: 0,
      coverage: 0,
      occupiedCells: 0,
      cellCount: 117,
      lowQualityThreshold: 0.24,
      yawTicks: [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90],
      pitchTicks: [60, 45, 30, 15, 0, -15, -30, -45, -60],
      cells: [],
    };
  }
  const result = await runAssetHelper(
    ["atlas", "--directory", directory],
    undefined,
    { signal, onProgress, timeoutMs: 600_000 },
  );
  return {
    side,
    ...result,
    cells: result.cells.map((cell) => ({
      ...cell,
      samples: cell.samples.map((sample) => ({
        ...sample,
        hasDflMetadata: true,
        polygonCount: 0,
        pointCount: 0,
        imageUrl: `/api/assets/${side}/aligned/${encodeURIComponent(sample.name)}`,
      })),
    })),
  };
}

function recoveryToken() {
  return `${new Date().toISOString().replace(/\D/g, "").slice(0, 14)}-${randomBytes(5).toString("hex")}`;
}

export async function buildAlignedSimilarityGroups(
  side,
  { refresh = false, threshold = 0.86, limit = 500, offset = 0, compareOffset = null,
    signal, onProgress } = {},
) {
  const parameters = normalizeSimilarityParameters({ threshold, limit, offset, compareOffset });
  const { threshold: safeThreshold, limit: safeLimit, offset: safeOffset,
    compareOffset: safeCompareOffset, windowSize, mode } = parameters;
  const directory = await verifyAlignedDirectory(side);
  const before = await roleDatasetSnapshot(side);
  if (!directory) {
    return {
      side, workspaceKey: PATHS.workspaceRoot, threshold: safeThreshold, total: 0, analyzedCount: 0, invalidCount: 0,
      truncated: false, groupCount: 0, groupedCount: 0, ungroupedCount: 0,
      method: "dct-hsv-edge-ssim-complete-link-v2", groups: [], cached: false,
      fingerprint: before.fingerprint, mode, offset: safeOffset,
      compareOffset: safeCompareOffset, limit: safeLimit, windowSize,
      pageCount: 0, pageIndex: Math.floor(safeOffset / windowSize),
      comparePageIndex: safeCompareOffset === null ? null : Math.floor(safeCompareOffset / windowSize),
      hasPrevious: safeOffset > 0, hasNext: false, selectedCount: 0,
      windows: [safeOffset, ...(safeCompareOffset === null ? [] : [safeCompareOffset])]
        .map((windowOffset, batch) => ({ batch, offset: windowOffset, count: 0,
          analyzedCount: 0, invalidCount: 0, start: null, end: null })),
    };
  }
  const result = await cachedAnalysis(
    side,
    `similarity:${JSON.stringify([safeThreshold, safeLimit, safeOffset, safeCompareOffset])}`,
    refresh,
    async () => ({ ...await runAssetHelper([
      "similarity", "--directory", directory, "--threshold", String(safeThreshold),
      "--limit", String(safeLimit), "--offset", String(safeOffset),
      ...(safeCompareOffset === null ? [] : ["--compare-offset", String(safeCompareOffset)]),
    ], undefined, { signal, onProgress, timeoutMs: 600_000 }), fingerprint: before.fingerprint }),
    { fingerprint: before.fingerprint },
  );
  const after = await roleDatasetSnapshot(side);
  if (before.fingerprint !== after.fingerprint) {
    invalidateAnalysis(side);
    throw new AssetError("分析期间素材已变化，请重新分析", "SIMILARITY_DATASET_CHANGED", 409);
  }
  return {
    side,
    ...result,
    workspaceKey: PATHS.workspaceRoot,
    groups: result.groups.map((group) => ({
      ...group,
      members: group.members.map((member) => ({
        ...member,
        imageUrl: `/api/assets/${side}/aligned/${encodeURIComponent(member.name)}`,
      })),
    })),
  };
}

function normalizeSimilarityParameters({ threshold = 0.86, limit = 500, offset = 0,
  compareOffset = null } = {}) {
  const numeric = (value, fallback, label, integer = false) => {
    const number = value === undefined || value === null ? fallback : Number(value);
    if ((value !== undefined && value !== null && !["number", "string"].includes(typeof value))
        || (typeof value === "string" && !value.trim())
        || !Number.isFinite(number) || (integer && !Number.isSafeInteger(number))) {
      throw new AssetError(`${label}需要有效${integer ? "整数" : "数值"}`, "SIMILARITY_PARAMETERS_INVALID");
    }
    return number;
  };
  const safeThreshold = Math.min(Math.max(numeric(threshold, 0.86, "相似阈值"), 0.72), 0.98);
  const safeLimit = Math.min(Math.max(numeric(limit, 500, "分析上限", true), 2), 500);
  const safeOffset = numeric(offset, 0, "批次起点", true);
  const safeCompareOffset = compareOffset === undefined || compareOffset === null
    ? null : numeric(compareOffset, 0, "比较批次起点", true);
  if (safeOffset < 0 || (safeCompareOffset !== null && safeCompareOffset < 0)) {
    throw new AssetError("批次起点不能为负数", "SIMILARITY_PARAMETERS_INVALID");
  }
  const windowSize = safeCompareOffset === null ? safeLimit : Math.min(Math.floor(safeLimit / 2), 250);
  if (safeCompareOffset !== null && Math.abs(safeOffset - safeCompareOffset) < windowSize) {
    throw new AssetError("请选择两个不重叠的图片批次", "SIMILARITY_WINDOWS_OVERLAP");
  }
  return { threshold: safeThreshold, limit: safeLimit, offset: safeOffset,
    compareOffset: safeCompareOffset, windowSize, mode: safeCompareOffset === null ? "batch" : "paired" };
}

export async function roleDatasetSnapshot(side, { includeIdentities = false } = {}) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) return { names: [], fingerprint: createHash("sha256").update("[]").digest("hex"),
    ...(includeIdentities ? { identities: new Map() } : {}) };
  const entries = (await readdir(directory, { withFileTypes: true }))
    .filter((entry) => IMAGE_NAME.test(entry.name)).sort((a, b) => a.name.localeCompare(b.name, "en"));
  const records = [];
  const identities = new Map();
  for (const entry of entries) {
    const target = await resolveExistingAlignedImage(side, encodeURIComponent(entry.name));
    const info = await lstat(target, { bigint: true });
    records.push([entry.name, String(info.size), String(info.mtimeNs), String(info.ctimeNs)]);
    if (includeIdentities) identities.set(entry.name, info);
  }
  return { names: records.map((record) => record[0]),
    fingerprint: createHash("sha256").update(JSON.stringify(records)).digest("hex"),
    ...(includeIdentities ? { identities } : {}) };
}

function sameReviewedImage(info, expected, { afterMove = false } = {}) {
  return Boolean(expected) && info.isFile() && !info.isSymbolicLink()
    && sameFileIdentity(info, expected) && info.size === expected.size
    && info.mtimeNs === expected.mtimeNs && (afterMove || info.ctimeNs === expected.ctimeNs);
}

export async function buildAlignedRoleGroups(side, { threshold = 0.5, signal, onProgress } = {}) {
  const value = Number(threshold);
  if (!Number.isFinite(value) || value < 0.3 || value > 0.85) {
    throw new AssetError("角色相似阈值需要在 0.30–0.85 之间", "ROLE_THRESHOLD_INVALID");
  }
  const before = await roleDatasetSnapshot(side);
  const directory = await verifyAlignedDirectory(side);
  const result = directory ? await runAssetHelper([
    "roles", "--directory", directory, "--threshold", String(value), "--limit", "2000",
  ], undefined, { signal, onProgress, timeoutMs: 600_000 }) : {
    total: 0, analyzedCount: 0, invalidCount: 0, invalid: [], groupCount: 0, groups: [], truncated: false,
  };
  const after = await roleDatasetSnapshot(side);
  if (before.fingerprint !== after.fingerprint) {
    throw new AssetError("分析期间素材已变化，请重新分析", "ROLE_DATASET_CHANGED", 409);
  }
  return { ...result, side, fingerprint: after.fingerprint,
    groups: result.groups.map((group) => ({ ...group, members: group.members.map((member) => ({
      ...member, imageUrl: `/api/assets/${side}/aligned/${encodeURIComponent(member.name)}`,
    })) })) };
}

export async function retainAlignedRoles(side, { names, fingerprint } = {}) {
  if (!Array.isArray(names) || !names.length || names.length > 2000
      || names.some((name) => typeof name !== "string" || !IMAGE_NAME.test(name))) {
    throw new AssetError("请先选择要保留的角色图片（最多 2000 张）", "ROLE_SELECTION_INVALID");
  }
  const snapshot = await roleDatasetSnapshot(side);
  if (snapshot.names.length > 2000 || snapshot.fingerprint !== fingerprint) {
    throw new AssetError("素材已变化或超过单次分析上限，请重新分批分析", "ROLE_DATASET_CHANGED", 409);
  }
  const available = new Set(snapshot.names);
  if (names.some((name) => !available.has(name))) {
    throw new AssetError("所选图片不属于当前数据集", "ROLE_SELECTION_INVALID");
  }
  const keep = new Set(names);
  const excluded = snapshot.names.filter((name) => !keep.has(name));
  const result = excluded.length
    ? await quarantineAlignedImages(side, excluded, { maximum: 2000 })
    : { side, count: 0, token: null, recoverable: true };
  return { ...result, keptCount: keep.size };
}

export async function buildAlignedPoseProbe(side, { datasetPath } = {}) {
  let directory;
  if (datasetPath !== undefined) {
    if (typeof datasetPath !== "string" || !path.isAbsolute(datasetPath)) {
      throw new AssetError("评测人脸集路径必须是项目内绝对路径", "ALIGNED_PATH_UNSAFE", 400);
    }
    directory = assertWithin(PATHS.workspaceRoot, path.resolve(datasetPath), "评测人脸集");
    try {
      assertWithin(await realpath(PATHS.workspaceRoot), await realpath(directory), "评测人脸集");
    } catch {
      throw new AssetError("评测人脸集超出项目工作区或不存在", "ALIGNED_PATH_UNSAFE", 400);
    }
  } else {
    directory = await verifyAlignedDirectory(side);
  }
  if (!directory) {
    return {
      schemaVersion: 1,
      side,
      datasetFingerprint: null,
      totalCount: 0,
      validCount: 0,
      invalidCount: 0,
      sampleCount: 0,
      maxSamples: 180,
      maxSamplesPerCell: 3,
      yawTicks: [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90],
      pitchTicks: [60, 45, 30, 15, 0, -15, -30, -45, -60],
      cells: [],
      samples: [],
    };
  }
  return runAssetHelper([
    "probe-manifest",
    "--directory",
    directory,
    "--side",
    side,
  ], undefined, { timeoutMs: 600_000 });
}

export async function auditAlignedAssets(side, {
  refresh = false, offset = 0, limit = 120, signal, onProgress,
} = {}) {
  const directory = await verifyAlignedDirectory(side);
  const safeOffset = Math.max(Number(offset) || 0, 0);
  const safeLimit = Math.min(Math.max(Number(limit) || 120, 1), 500);
  if (!directory) {
    return {
      side,
      total: 0,
      offset: safeOffset,
      limit: safeLimit,
      analyzedCount: 0,
      validMetadataCount: 0,
      invalidMetadataCount: 0,
      maskedCount: 0,
      xsegSharpnessCount: 0,
      usableCount: 0,
      issueItemCount: 0,
      severeIssueCount: 0,
      uniqueSourceCount: 0,
      duplicateSourceGroupCount: 0,
      meanQualityScore: 0,
      meanSharpness: 0,
      meanFullSharpness: 0,
      issueCounts: {},
      items: [],
      cached: false,
    };
  }
  const result = await cachedAnalysis(side, `audit:${safeOffset}:${safeLimit}`, refresh, () => (
    runAssetHelper([
      "audit",
      "--directory",
      directory,
      "--offset",
      String(safeOffset),
      "--limit",
      String(safeLimit),
    ], undefined, { signal, onProgress, timeoutMs: 600_000 })
  ));
  return {
    side,
    ...result,
    items: result.items.map((item) => ({
      ...item,
      imageUrl: `/api/assets/${side}/aligned/${encodeURIComponent(item.name)}`,
    })),
  };
}

export async function inspectAlignedPack(side, { refresh = false, signal } = {}) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    return { side, present: false, status: "aligned_missing", warnings: [], cached: false };
  }
  const result = await cachedAnalysis(side, "pack", refresh, () => (
    runAssetHelper(["pack-inspect", "--directory", directory], undefined, { signal })
  ));
  return { side, ...result };
}

export async function inspectExtractionCoverage(
  side,
  { refresh = false, offset = 0, limit = 120, signal, onProgress } = {},
) {
  const directory = await verifyAlignedDirectory(side);
  const frames = path.join(PATHS.workspaceRoot, `data_${side}`);
  const safeOffset = Math.max(Number(offset) || 0, 0);
  const safeLimit = Math.min(Math.max(Number(limit) || 120, 1), 500);
  if (!directory || !(await pathExists(frames))) {
    return {
      side,
      total: 0,
      offset: safeOffset,
      limit: safeLimit,
      analyzedCount: 0,
      coveredCount: 0,
      uncoveredCount: 0,
      multiFaceCount: 0,
      orphanAlignmentCount: 0,
      items: [],
      cached: false,
    };
  }
  const result = await cachedAnalysis(side, `coverage:${safeOffset}:${safeLimit}`, refresh, () => (
    runAssetHelper([
      "coverage",
      "--frames",
      frames,
      "--directory",
      directory,
      "--offset",
      String(safeOffset),
      "--limit",
      String(safeLimit),
    ], undefined, { signal, onProgress, timeoutMs: 600_000 })
  ));
  return {
    side,
    ...result,
    items: result.items.map((item) => ({
      ...item,
      frameUrl: `/api/workspace/review/${side}-frame/${encodeURIComponent(item.name)}`,
      faces: item.faces.map((face) => ({
        ...face,
        alignedUrl: `/api/assets/${side}/aligned/${encodeURIComponent(face.alignedName)}`,
      })),
    })),
  };
}

export async function inspectAlignedAnnotation(side, encodedName) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  return runAssetHelper(["inspect", "--file", target]);
}

export async function saveAlignedAnnotation(side, encodedName, payload) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  const result = await runAssetHelper(["save", "--file", target], payload);
  invalidateAnalysis(side);
  return result;
}

export async function previewAlignedRepair(side, encodedName, payload) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  const frames = path.join(PATHS.workspaceRoot, `data_${side}`);
  return runAssetHelper([
    "alignment-preview", "--file", target, "--frames", frames,
  ], { landmarks: payload?.landmarks });
}

export async function applyAlignedRepair(side, encodedName, payload) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  const frames = path.join(PATHS.workspaceRoot, `data_${side}`);
  const token = recoveryToken();
  const backupDirectory = assertWithin(
    PATHS.runtimeRoot,
    path.join(PATHS.runtimeRoot, "alignment-backups", side, token),
    "对齐恢复目录",
  );
  await mkdir(backupDirectory, { recursive: true });
  const result = await runAssetHelper([
    "alignment-apply", "--file", target, "--frames", frames,
    "--backup-directory", backupDirectory,
  ], { landmarks: payload?.landmarks });
  invalidateAnalysis(side);
  return { side, token, ...result, recoverable: true };
}

export async function listAlignedRepairBackups(side) {
  alignedDirectory(side);
  const root = path.join(PATHS.runtimeRoot, "alignment-backups", side);
  if (!(await pathExists(root))) return [];
  const tokens = await readdir(root, { withFileTypes: true });
  const result = [];
  for (const tokenEntry of tokens) {
    if (!tokenEntry.isDirectory() || !QUARANTINE_TOKEN.test(tokenEntry.name)) continue;
    const directory = path.join(root, tokenEntry.name);
    const files = await readdir(directory, { withFileTypes: true });
    for (const file of files) {
      if (file.isFile() && IMAGE_NAME.test(file.name)) {
        result.push({ side, token: tokenEntry.name, name: file.name });
      }
    }
  }
  return result.sort((a, b) => b.token.localeCompare(a.token));
}

export async function restoreAlignedRepair(side, token, encodedName) {
  await ensureSafeAlignedDirectory(side);
  const target = resolveAlignedImage(side, encodedName);
  if (!QUARANTINE_TOKEN.test(token)) {
    throw new AssetError("对齐备份记录无效", "ALIGNMENT_BACKUP_INVALID");
  }
  const name = path.basename(target);
  const root = path.join(PATHS.runtimeRoot, "alignment-backups", side);
  const backup = assertWithin(root, path.join(root, token, name), "对齐备份");
  if (!(await pathExists(backup))) {
    throw new AssetError("对齐备份不存在", "ALIGNMENT_BACKUP_MISSING", 404);
  }
  const restoreToken = recoveryToken();
  const undoDirectory = path.join(PATHS.runtimeRoot, "alignment-restores", side, restoreToken);
  await mkdir(undoDirectory, { recursive: true });
  if (await pathExists(target)) {
    await resolveExistingAlignedImage(side, encodedName);
    await copyFile(target, path.join(undoDirectory, name));
  }
  const temporary = `${target}.${process.pid}.${Date.now()}.restore`;
  try {
    await copyFile(backup, temporary);
    await rename(temporary, target);
  } catch (error) {
    if (await pathExists(temporary)) await unlink(temporary);
    throw error;
  }
  invalidateAnalysis(side);
  return { side, token, name, restored: true, undoToken: restoreToken };
}

async function streamImageFile(response, target, { label, unsafeCode }) {
  const { fileHandle, fileInfo } = await openVerifiedImage(target, label, unsafeCode);
  try {
    const contentType = path.extname(target).toLowerCase() === ".png" ? "image/png" : "image/jpeg";
    response.writeHead(200, {
      "Content-Type": contentType,
      "Content-Length": Number(fileInfo.size),
      "Cache-Control": "private, max-age=30",
    });
    try {
      await pipeline(fileHandle.createReadStream({ autoClose: false }), response);
    } catch {
      // Headers may already be committed; terminate only this response and keep the runtime alive.
      if (!response.destroyed) response.destroy();
    }
  } finally {
    await fileHandle.close().catch(() => {});
  }
}

export async function streamAlignedImage(response, side, encodedName) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  return streamImageFile(response, target, {
    label: "aligned 图片",
    unsafeCode: "ALIGNED_PATH_UNSAFE",
  });
}

export async function inspectQuarantinedAnnotation(side, token, encodedName) {
  const target = await resolveQuarantinedImage(side, token, encodedName);
  return runAssetHelper(["inspect", "--file", target]);
}

export async function streamQuarantinedImage(response, side, token, encodedName) {
  const target = await resolveQuarantinedImage(side, token, encodedName);
  return streamImageFile(response, target, {
    label: "隔离图片",
    unsafeCode: "QUARANTINE_PATH_UNSAFE",
  });
}

export async function streamAlignedPoster(response, side) {
  const directory = await verifyAlignedDirectory(side);
  if (!directory) {
    throw new AssetError("aligned 预览图不存在", "IMAGE_MISSING", 404);
  }
  const names = (await readdir(directory, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && !entry.isSymbolicLink() && IMAGE_NAME.test(entry.name))
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right, "en", { numeric: true }));
  if (!names.length) {
    throw new AssetError("aligned 预览图不存在", "IMAGE_MISSING", 404);
  }
  return streamAlignedImage(response, side, encodeURIComponent(names[0]));
}

export async function quarantineAlignedImage(side, encodedName) {
  const target = await resolveExistingAlignedImage(side, encodedName);
  const { token, directory } = await createQuarantineTokenDirectory(side);
  await rename(target, path.join(directory, path.basename(target)));
  invalidateAnalysis(side);
  return { side, token, name: path.basename(target), recoverable: true };
}

export async function quarantineAlignedImages(side, names, { maximum = 500, review } = {}) {
  if (!Array.isArray(names) || !names.length || names.length > maximum) {
    throw new AssetError(`批量隔离需要 1–${maximum} 个文件`, "QUARANTINE_BATCH_INVALID");
  }
  const uniqueNames = [...new Set(names.map((name) => String(name)))];
  let reviewedFingerprint = null;
  let reviewedIdentities = null;
  if (review !== undefined) {
    if (!review || typeof review !== "object" || Array.isArray(review)
        || typeof review.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(review.fingerprint)
        || review.side !== side || review.workspaceKey !== PATHS.workspaceRoot
        || names.some((name) => typeof name !== "string")) {
      throw new AssetError("相似图复核记录不属于当前项目与数据集，请重新分析", "SIMILARITY_REVIEW_INVALID", 409);
    }
    const snapshot = await roleDatasetSnapshot(side, { includeIdentities: true });
    if (snapshot.fingerprint !== review.fingerprint) {
      throw new AssetError("素材已变化，请重新分析后隔离", "SIMILARITY_DATASET_CHANGED", 409);
    }
    const result = await buildAlignedSimilarityGroups(side, normalizeSimilarityParameters(review));
    if (result.fingerprint !== review.fingerprint) {
      throw new AssetError("素材已变化，请重新分析后隔离", "SIMILARITY_DATASET_CHANGED", 409);
    }
    const eligible = new Set(result.groups.flatMap((group) => group.members
      .filter((member) => !member.representative).map((member) => member.name)));
    if (uniqueNames.some((name) => !eligible.has(name))) {
      throw new AssetError("只能隔离本次候选组中已复核的非代表图", "SIMILARITY_SELECTION_INVALID", 409);
    }
    reviewedFingerprint = result.fingerprint;
    reviewedIdentities = snapshot.identities;
  }
  const targets = await Promise.all(uniqueNames.map((name) => (
    resolveExistingAlignedImage(side, encodeURIComponent(name))
  )));
  if (reviewedFingerprint !== null
      && (await roleDatasetSnapshot(side)).fingerprint !== reviewedFingerprint) {
    throw new AssetError("素材已变化，请重新分析后隔离", "SIMILARITY_DATASET_CHANGED", 409);
  }
  const { token, directory: destinationDirectory } = await createQuarantineTokenDirectory(side);
  const moved = [];
  try {
    // Token creation awaits filesystem work after the earlier review check.
    // Check again before moving, and retain each selected file's identity for
    // later moves; our own completed moves will change the whole inventory.
    if (reviewedFingerprint !== null
        && (await roleDatasetSnapshot(side)).fingerprint !== reviewedFingerprint) {
      throw new AssetError("素材已变化，请重新分析后隔离", "SIMILARITY_DATASET_CHANGED", 409);
    }
    for (const target of targets) {
      const name = path.basename(target);
      if (reviewedIdentities !== null) {
        await resolveExistingAlignedImage(side, encodeURIComponent(name));
        if (!sameReviewedImage(await lstat(target, { bigint: true }), reviewedIdentities.get(name))) {
          throw new AssetError("所选图片在隔离前已变化，请重新分析后隔离", "SIMILARITY_DATASET_CHANGED", 409);
        }
      }
      const destination = path.join(destinationDirectory, name);
      await rename(target, destination);
      moved.push(name);
      if (reviewedIdentities !== null
          && !sameReviewedImage(await lstat(destination, { bigint: true }), reviewedIdentities.get(name), { afterMove: true })) {
        throw new AssetError("所选图片在隔离期间已变化，请检查恢复记录", "SIMILARITY_DATASET_CHANGED", 409);
      }
    }
  } catch (error) {
    const rollbackErrors = [];
    for (const name of moved) {
      try {
        if (reviewedIdentities !== null
            && !sameReviewedImage(await lstat(path.join(destinationDirectory, name), { bigint: true }), reviewedIdentities.get(name), { afterMove: true })) {
          throw new AssetError("恢复区图片已变化，原文件保留；请检查恢复记录", "QUARANTINE_IMAGE_CHANGED", 409);
        }
        await restoreAlignedImage(side, token, encodeURIComponent(name));
      } catch (failure) {
        rollbackErrors.push({ name, code: failure.code ?? "QUARANTINE_ROLLBACK_FAILED" });
      }
    }
    invalidateAnalysis(side);
    if (rollbackErrors.length) {
      throw new AssetError("隔离未完成，部分图片保留在恢复区；请先检查恢复记录", "QUARANTINE_RECOVERY_REQUIRED", 500,
        { side, token, originalCode: error.code ?? "QUARANTINE_MOVE_FAILED", rollbackErrors });
    }
    throw error;
  }
  invalidateAnalysis(side);
  return { side, token, names: moved, count: moved.length, recoverable: true };
}

export async function listAlignedQuarantine(side, { offset = 0, limit = 60 } = {}) {
  const safeOffset = Math.max(Number(offset) || 0, 0);
  const safeLimit = Math.min(Math.max(Number(limit) || 60, 1), 200);
  const root = await verifyQuarantineRoot(side);
  if (!root) {
    return { side, total: 0, offset: safeOffset, limit: safeLimit, items: [] };
  }
  const result = await runAssetHelper([
    "quarantine-list",
    "--directory",
    root,
    "--offset",
    String(safeOffset),
    "--limit",
    String(safeLimit),
  ]);
  return {
    side,
    ...result,
    items: result.items.map((item) => ({
      ...item,
      imageUrl: (
        `/api/assets/${side}/quarantine/${encodeURIComponent(item.token)}`
        + `/${encodeURIComponent(item.name)}`
      ),
    })),
  };
}

export async function restoreAlignedImage(side, token, encodedName) {
  const destination = resolveAlignedImage(side, encodedName);
  const name = path.basename(destination);
  const source = await resolveQuarantinedImage(side, token, encodedName);
  await ensureSafeAlignedDirectory(side);
  const { fileHandle, fileInfo: sourceInfo } = await openVerifiedImage(
    source,
    "隔离图片",
    "QUARANTINE_PATH_UNSAFE",
  );
  let installed = false;
  try {
    try {
      // A hard-link install is complete-or-absent and refuses to overwrite an existing name.
      await link(source, destination);
      installed = true;
    } catch (error) {
      if (error?.code === "EEXIST") {
        throw new AssetError("aligned 中已有同名图片，不能覆盖恢复", "RESTORE_CONFLICT", 409);
      }
      throw error;
    }
    const {
      fileHandle: destinationHandle,
      fileInfo: destinationInfo,
    } = await openVerifiedImage(destination, "恢复目标", "RESTORE_INSTALL_UNSAFE");
    try {
      if (!sameFileIdentity(sourceInfo, destinationInfo)) {
        throw new AssetError("恢复目标未能安全提交", "RESTORE_INSTALL_UNSAFE", 500);
      }
    } finally {
      await destinationHandle.close().catch(() => {});
    }
  } catch (error) {
    if (installed) await unlink(destination).catch(() => {});
    throw error;
  } finally {
    await fileHandle.close().catch(() => {});
  }
  try {
    await unlink(source);
  } catch (error) {
    try {
      await unlink(destination);
    } catch {
      // Keep the original unlink failure; the source remains the recovery authority.
    }
    throw error;
  }
  invalidateAnalysis(side);
  return { side, token, name, restored: true };
}
