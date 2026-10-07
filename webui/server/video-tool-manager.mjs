import { runHelperProcess } from "./helper-process.mjs";
import { createHash, randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readdir, rename, rm, stat } from "node:fs/promises";
import path from "node:path";
import { PATHS, assertWithin, pathExists, readJson, writeJsonAtomic } from "./paths.mjs";
import { inspectWorkspace, resolveWorkspaceMaterial } from "./workspace-manager.mjs";

const IMAGE_NAME = /^[^<>:"/\\|?*\u0000-\u001f]{1,220}\.(?:jpe?g|png)$/i;
const SEGMENT_ID = /^seg-[a-f0-9]{10}$/;
const ARCHIVE_TOKEN = /^\d{14}-[a-f0-9]{10}$/;
const MAX_SEGMENTS = 100;
const FRAME_TIMELINE = "frames.timeline.json";

export class VideoToolError extends Error {
  constructor(message, code = "VIDEO_TOOL_ERROR", status = 400, details) {
    super(message);
    this.name = "VideoToolError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function assertSide(side) {
  if (!new Set(["src", "dst"]).has(side)) {
    throw new VideoToolError("素材类型不受支持", "SIDE_INVALID");
  }
}

function videoRoot() {
  return path.join(PATHS.runtimeRoot, "video");
}

function manifestPath(side, kind) {
  assertSide(side);
  return assertWithin(videoRoot(), path.join(videoRoot(), `${side}-${kind}.json`), "视频清单");
}

function finite(value) {
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

export function normalizeSegments(segments, duration) {
  if (!Array.isArray(segments) || segments.length > MAX_SEGMENTS) {
    throw new VideoToolError(`分段数量必须在 0–${MAX_SEGMENTS} 之间`, "SEGMENTS_INVALID");
  }
  const normalized = segments.map((segment, index) => {
    const start = finite(segment?.start);
    const end = finite(segment?.end);
    if (start === null || end === null || start < 0 || end <= start || (!segment?.sourceFrameRange && end - start < 0.08)) {
      throw new VideoToolError(`第 ${index + 1} 个分段时间范围无效`, "SEGMENT_RANGE_INVALID");
    }
    if (duration && end > duration + 0.05) {
      throw new VideoToolError(`第 ${index + 1} 个分段超出视频时长`, "SEGMENT_RANGE_INVALID");
    }
    let sourceFrameRange;
    if (segment?.sourceFrameRange != null) {
      const range = segment.sourceFrameRange;
      if (!/^[a-f0-9]{64}$/.test(range.sourceSha256) || !Number.isInteger(range.startIndex) || range.startIndex < 0
          || !Number.isInteger(range.endIndexExclusive) || range.endIndexExclusive <= range.startIndex
          || !Number.isInteger(range.startPts) || !Number.isInteger(range.endPts) || range.endPts <= range.startPts
          || !Array.isArray(range.timeBase) || range.timeBase.length !== 2 || range.timeBase.some(value => !Number.isInteger(value) || value <= 0)) {
        throw new VideoToolError("场景分段的原始帧范围无效", "SEGMENT_RANGE_INVALID");
      }
      sourceFrameRange = { sourceSha256: range.sourceSha256, startIndex: range.startIndex,
        endIndexExclusive: range.endIndexExclusive, startPts: range.startPts, endPts: range.endPts, timeBase: [...range.timeBase] };
    }
    return {
      id: SEGMENT_ID.test(segment?.id) ? segment.id : `seg-${randomBytes(5).toString("hex")}`,
      start,
      end,
      label: String(segment?.label ?? `片段 ${index + 1}`).trim().slice(0, 48) || `片段 ${index + 1}`,
      selected: segment?.selected !== false,
      ...(sourceFrameRange ? { sourceFrameRange } : {}),
    };
  });
  return normalized.sort((a, b) => a.start - b.start || a.end - b.end);
}

async function fingerprintMaterial(target) {
  const fileStat = await stat(target);
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(target)) digest.update(chunk);
  const after = await stat(target);
  if (after.size !== fileStat.size || after.mtimeMs !== fileStat.mtimeMs) {
    throw new VideoToolError("素材在读取时发生变化，请重新检查", "VIDEO_SOURCE_CHANGED", 409);
  }
  return {
    name: path.basename(target),
    bytes: fileStat.size,
    modifiedAtMs: Math.round(fileStat.mtimeMs),
    sha256: digest.digest("hex"),
  };
}

function sameMaterial(manifest, fingerprint) {
  return manifest?.material?.name === fingerprint.name
    && manifest.material.bytes === fingerprint.bytes
    && manifest.material.sha256 === fingerprint.sha256;
}

export async function sceneCapabilities() {
  const { stdout } = await runProcess(PATHS.python, [
    path.join(PATHS.webuiRoot, "python", "scene_detection.py"), "--capabilities",
  ], { timeoutMs: 60_000 });
  return JSON.parse(stdout);
}

export function bindSceneCuts(timeline, detection) {
  if (!detection || detection.schemaVersion !== 2 || detection.source?.sha256 !== timeline.source?.sha256) {
    return { ...timeline, sceneDetectionApplied: false, sceneDetectionReason: "没有与当前素材 SHA256 绑定的场景结果" };
  }
  const base = JSON.stringify(detection.timeBase);
  const cuts = detection.cuts ?? [];
  if (!Array.isArray(cuts) || !Number.isInteger(detection.sourceFrameCount)
      || cuts.some((cut, index) => !Number.isInteger(cut.sourceFrameIndex) || cut.sourceFrameIndex <= 0
        || cut.sourceFrameIndex >= detection.sourceFrameCount || !Number.isInteger(cut.pts)
        || JSON.stringify(cut.timeBase) !== base || (index && cuts[index - 1].sourceFrameIndex >= cut.sourceFrameIndex))) {
    throw new VideoToolError("场景整数帧索引或时间基无效", "SCENE_TIMELINE_INVALID", 422);
  }
  let previous;
  const frames = timeline.frames.map(frame => {
    if (JSON.stringify(frame.timeBase) !== base || !Number.isInteger(frame.sourceFrameIndex)) {
      throw new VideoToolError("提帧和场景检测的源时间线不一致", "SCENE_TIMELINE_INVALID", 422);
    }
    let low = 0, high = cuts.length;
    while (low < high) { const middle = (low + high) >>> 1; if (cuts[middle].sourceFrameIndex <= frame.sourceFrameIndex) low = middle + 1; else high = middle; }
    if (low && cuts[low - 1].sourceFrameIndex === frame.sourceFrameIndex && cuts[low - 1].pts !== frame.pts) {
      throw new VideoToolError("场景边界的整数 PTS 与提帧不一致", "SCENE_TIMELINE_INVALID", 422);
    }
    const result = { ...frame, shotId: low,
      cutBefore: !previous || previous.shotId !== low || previous.segmentIndex !== frame.segmentIndex
        || frame.sourceFrameIndex <= previous.sourceFrameIndex };
    previous = result;
    return result;
  });
  return { ...timeline, frames, sceneDetectionApplied: true, sceneDetection: {
    schemaVersion: 2, sourceSha256: detection.source.sha256, algorithm: detection.algorithm,
    parameters: detection.parameters, cutCount: cuts.length, cuts,
  } };
}

async function runProcess(executable, args, { timeoutMs = 30 * 60_000, signal, onStderr } = {}) {
  const result = await runHelperProcess(executable, args, { timeoutMs, signal, onStderr, label: "视频工具", maxBytes: 8 * 1024 * 1024 });
  if (result.code !== 0) throw new VideoToolError(result.stderr.trim() || "视频工具执行失败", "VIDEO_TOOL_FAILED", 422);
  return result;
}


async function readManifest(side, kind) {
  const target = manifestPath(side, kind);
  if (!(await pathExists(target))) return null;
  try {
    return await readJson(target);
  } catch {
    return null;
  }
}

export async function listFrameArchives(side) {
  assertSide(side);
  const root = path.join(PATHS.archiveRoot, "frames", side);
  if (!(await pathExists(root))) return [];
  const entries = await readdir(root, { withFileTypes: true });
  const archives = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !ARCHIVE_TOKEN.test(entry.name)) continue;
    const files = (await readdir(path.join(root, entry.name), { withFileTypes: true }))
      .filter((file) => file.isFile() && IMAGE_NAME.test(file.name));
    if (files.length) archives.push({ side, token: entry.name, frameCount: files.length });
  }
  return archives.sort((left, right) => right.token.localeCompare(left.token));
}

export async function inspectVideoTimeline(side) {
  assertSide(side);
  const workspace = await inspectWorkspace();
  const material = workspace.materials?.[side] ?? null;
  if (!material) {
    return { side, material: null, scenes: [], segments: [], archives: await listFrameArchives(side), sceneThreshold: 0.32, sceneAlgorithm: "ffmpeg" };
  }
  const fingerprint = await fingerprintMaterial(material.path);
  const [sceneManifest, segmentManifest, archives, capabilities] = await Promise.all([
    readManifest(side, "scenes"),
    readManifest(side, "segments"),
    listFrameArchives(side),
    sceneCapabilities(),
  ]);
  const currentScenes = sameMaterial(sceneManifest, fingerprint) ? sceneManifest : null;
  return {
    side,
    material: {
      ...material,
      path: undefined,
      url: `/api/workspace/materials/${side}`,
    },
    scenes: currentScenes?.scenes ?? [],
    sceneThreshold: currentScenes?.threshold ?? 0.32,
    sceneAlgorithm: currentScenes?.algorithm ?? "ffmpeg",
    sceneSummary: currentScenes ? { total: currentScenes.total, truncated: currentScenes.truncated,
      displayed: currentScenes.scenes.length, sourceFrameCount: currentScenes.sourceFrameCount,
      cutCount: currentScenes.cuts.length, sourceSha256: currentScenes.source.sha256 } : null,
    sceneAlgorithms: capabilities.algorithms,
    segments: sameMaterial(segmentManifest, fingerprint) ? segmentManifest.segments ?? [] : [],
    archives,
  };
}

export async function detectVideoScenes(side, { algorithm = "ffmpeg", threshold, signal, onProgress } = {}) {
  assertSide(side);
  if (!["ffmpeg", "adaptive", "transnetv2"].includes(algorithm)) throw new VideoToolError("场景算法无效", "SCENE_ALGORITHM_INVALID");
  const safeThreshold = threshold == null ? ({ ffmpeg: 0.32, adaptive: 3, transnetv2: 0.5 })[algorithm] : Number(threshold);
  if (!Number.isFinite(safeThreshold) || safeThreshold < 0 || safeThreshold > (algorithm === "adaptive" ? 255 : 1)
      || (algorithm === "adaptive" && safeThreshold === 0)) throw new VideoToolError("场景算法阈值无效", "SCENE_THRESHOLD_INVALID");
  if (!(await pathExists(PATHS.ffmpeg))) {
    throw new VideoToolError("内置 ffmpeg 不存在", "FFMPEG_MISSING", 503);
  }
  const material = await resolveWorkspaceMaterial(side);
  const fingerprint = await fingerprintMaterial(material);
  const { stdout } = await runProcess(PATHS.python, [
    path.join(PATHS.webuiRoot, "python", "scene_detection.py"), "--source", material,
    "--ffmpeg", PATHS.ffmpeg, "--ffprobe", PATHS.ffprobe, "--algorithm", algorithm, "--threshold", String(safeThreshold),
  ], { signal, onStderr: chunk => onProgress?.({ stage: "检测视频场景", message: chunk.toString().trim() }) });
  const detection = JSON.parse(stdout);
  const after = await fingerprintMaterial(material);
  if (detection.schemaVersion !== 2 || detection.source?.sha256 !== fingerprint.sha256 || after.sha256 !== fingerprint.sha256
      || !Array.isArray(detection.scenes) || !Array.isArray(detection.cuts)) throw new VideoToolError("场景结果与当前素材不一致", "VIDEO_SOURCE_CHANGED", 409);
  if (signal?.aborted) throw new VideoToolError("场景检测已取消", "OPERATION_CANCELLED", 409);
  const manifest = {
    ...detection,
    side,
    material: fingerprint,
    threshold: safeThreshold,
    analyzedAt: new Date().toISOString(),
    scenes: detection.scenes.map((scene, index) => ({ ...scene, id: `scene-${String(index + 1).padStart(3, "0")}` })),
  };
  await mkdir(videoRoot(), { recursive: true });
  await writeJsonAtomic(manifestPath(side, "scenes"), manifest);
  return { side, threshold: safeThreshold, algorithm, scenes: manifest.scenes,
    sceneSummary: { total: detection.total, truncated: detection.truncated, displayed: manifest.scenes.length,
      sourceFrameCount: detection.sourceFrameCount, cutCount: detection.cuts.length, sourceSha256: detection.source.sha256 } };
}

export async function saveVideoSegments(side, { segments } = {}) {
  assertSide(side);
  const [material, workspace] = await Promise.all([resolveWorkspaceMaterial(side), inspectWorkspace()]);
  const duration = Number(workspace.materials?.[side]?.durationSeconds) || 0;
  const normalized = normalizeSegments(segments, duration);
  const manifest = {
    schemaVersion: 1,
    side,
    material: await fingerprintMaterial(material),
    updatedAt: new Date().toISOString(),
    segments: normalized,
  };
  await mkdir(videoRoot(), { recursive: true });
  await writeJsonAtomic(manifestPath(side, "segments"), manifest);
  return { side, segments: normalized };
}

export async function extractVideoSegments(side, { segments, fps = 0 } = {}) {
  assertSide(side);
  if (!(await pathExists(PATHS.ffmpeg))) {
    throw new VideoToolError("内置 ffmpeg 不存在", "FFMPEG_MISSING", 503);
  }
  const [material, workspace] = await Promise.all([resolveWorkspaceMaterial(side), inspectWorkspace()]);
  const duration = Number(workspace.materials?.[side]?.durationSeconds) || 0;
  const sourceFingerprint = await fingerprintMaterial(material);
  const savedSegments = await readManifest(side, "segments");
  if (segments == null && !sameMaterial(savedSegments, sourceFingerprint)) {
    throw new VideoToolError("分段清单属于其他素材，请重新保存", "VIDEO_SOURCE_CHANGED", 409);
  }
  const requested = segments ?? savedSegments.segments;
  const selected = normalizeSegments(requested, duration).filter((segment) => segment.selected);
  if (!selected.length) throw new VideoToolError("至少选择一个分段", "SEGMENTS_EMPTY");
  const safeFps = Number(fps) === 0 ? 0 : Math.min(Math.max(Number(fps) || 0, 1), 60);
  const token = `${new Date().toISOString().replace(/\D/g, "").slice(0, 14)}-${randomBytes(5).toString("hex")}`;
  const root = videoRoot();
  const staging = assertWithin(root, path.join(root, "staging", `${side}-${token}`), "提帧暂存目录");
  const frameRoot = path.join(PATHS.workspaceRoot, `data_${side}`);
  const archive = assertWithin(PATHS.archiveRoot, path.join(PATHS.archiveRoot, "frames", side, token), "帧归档目录");
  await mkdir(staging, { recursive: true });
  const movedExisting = [];
  const installed = [];
  try {
    // The native extractor records the decoded integer PTS and real source
    // frame index. Drop-only sampling does not manufacture CFR duplicates.
    await runProcess(PATHS.python, [
      path.join(path.dirname(PATHS.currentMain), "core", "media_timeline.py"),
      "--source", material, "--output", staging, "--ffmpeg", PATHS.ffmpeg,
      "--ffprobe", PATHS.ffprobe, "--fps", String(safeFps), "--segments-json", JSON.stringify(selected),
    ]);
    const timing = await readJson(path.join(staging, FRAME_TIMELINE));
    if (timing.source?.sha256 !== sourceFingerprint.sha256
        || (await fingerprintMaterial(material)).sha256 !== sourceFingerprint.sha256) {
      throw new VideoToolError("提帧期间素材发生变化", "VIDEO_SOURCE_CHANGED", 409);
    }
    const detection = await readManifest(side, "scenes");
    const annotatedTiming = bindSceneCuts(timing, sameMaterial(detection, sourceFingerprint) ? detection : null);
    await writeJsonAtomic(path.join(staging, FRAME_TIMELINE), annotatedTiming);
    const stagedFiles = (await readdir(staging, { withFileTypes: true }))
      .filter((entry) => entry.isFile() && (IMAGE_NAME.test(entry.name) || entry.name === FRAME_TIMELINE));
    if (!stagedFiles.some((entry) => IMAGE_NAME.test(entry.name))) throw new VideoToolError("选定分段没有生成任何帧", "NO_FRAMES_EXTRACTED", 422);
    await Promise.all([mkdir(frameRoot, { recursive: true }), mkdir(archive, { recursive: true })]);
    const existing = (await readdir(frameRoot, { withFileTypes: true }))
      .filter((entry) => entry.isFile() && (IMAGE_NAME.test(entry.name) || entry.name === FRAME_TIMELINE));
    for (const entry of existing) {
      await rename(path.join(frameRoot, entry.name), path.join(archive, entry.name));
      movedExisting.push(entry.name);
    }
    for (const entry of stagedFiles) {
      await rename(path.join(staging, entry.name), path.join(frameRoot, entry.name));
      installed.push(entry.name);
    }
    await rm(staging, { recursive: true, force: true });
    return {
      side,
      token,
      segmentCount: selected.length,
      frameCount: installed.filter((name) => IMAGE_NAME.test(name)).length,
      archivedFrameCount: movedExisting.filter((name) => IMAGE_NAME.test(name)).length,
      recoverableArchive: archive,
      fps: safeFps,
      timingManifest: FRAME_TIMELINE,
      imageFormat: "png",
      sceneDetectionApplied: annotatedTiming.sceneDetectionApplied,
    };
  } catch (error) {
    if (error.code === "HELPER_STOP_UNCONFIRMED") {
      error.details = { ...error.details, retainedStaging: staging };
      throw error;
    }
    for (const name of installed) {
      const target = path.join(frameRoot, name);
      if (await pathExists(target)) await rename(target, path.join(staging, name));
    }
    for (const name of movedExisting) {
      const source = path.join(archive, name);
      if (await pathExists(source)) await rename(source, path.join(frameRoot, name));
    }
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

export async function restoreFrameArchive(side, token) {
  assertSide(side);
  if (!ARCHIVE_TOKEN.test(token)) throw new VideoToolError("帧归档令牌无效", "FRAME_ARCHIVE_INVALID");
  const archiveRoot = path.join(PATHS.archiveRoot, "frames", side);
  const source = assertWithin(archiveRoot, path.join(archiveRoot, token), "帧归档");
  if (!(await pathExists(source))) throw new VideoToolError("帧归档不存在", "FRAME_ARCHIVE_MISSING", 404);
  const archived = (await readdir(source, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && (IMAGE_NAME.test(entry.name) || entry.name === FRAME_TIMELINE));
  if (!archived.some((entry) => IMAGE_NAME.test(entry.name))) throw new VideoToolError("帧归档为空", "FRAME_ARCHIVE_EMPTY", 409);
  const frameRoot = path.join(PATHS.workspaceRoot, `data_${side}`);
  const undoToken = `${new Date().toISOString().replace(/\D/g, "").slice(0, 14)}-${randomBytes(5).toString("hex")}`;
  const undo = assertWithin(archiveRoot, path.join(archiveRoot, undoToken), "当前帧撤销归档");
  await Promise.all([mkdir(frameRoot, { recursive: true }), mkdir(undo, { recursive: true })]);
  const current = (await readdir(frameRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && (IMAGE_NAME.test(entry.name) || entry.name === FRAME_TIMELINE));
  const movedCurrent = [];
  const restored = [];
  try {
    for (const entry of current) {
      await rename(path.join(frameRoot, entry.name), path.join(undo, entry.name));
      movedCurrent.push(entry.name);
    }
    for (const entry of archived) {
      await rename(path.join(source, entry.name), path.join(frameRoot, entry.name));
      restored.push(entry.name);
    }
    return { side, token, restoredFrameCount: restored.filter((name) => IMAGE_NAME.test(name)).length, undoToken, recoverable: true };
  } catch (error) {
    for (const name of restored) {
      const target = path.join(frameRoot, name);
      if (await pathExists(target)) await rename(target, path.join(source, name));
    }
    for (const name of movedCurrent) {
      const backup = path.join(undo, name);
      if (await pathExists(backup)) await rename(backup, path.join(frameRoot, name));
    }
    throw error;
  }
}
