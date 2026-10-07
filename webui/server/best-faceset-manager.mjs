import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, readdir, realpath } from "node:fs/promises";
import path from "node:path";
import { PATHS, assertWithin, pathExists } from "./paths.mjs";
import { buildDflEnvironment } from "./environment.mjs";
import { runHelperProcess } from "./helper-process.mjs";

const PLAN = /^plan-[a-f0-9]{32}$/;
const HASH = /^[a-f0-9]{64}$/;
const CLASSES = new Set(["selected", "review", "rejected"]);
export class BestFacesetError extends Error {
  constructor(message, code = "BEST_FACESET_INVALID", status = 400) { super(message); this.name = "BestFacesetError"; this.code = code; this.status = status; }
}
function member(value) {
  if (typeof value !== "string" || value.length > 1024 || /[\\:\u0000-\u001f]/.test(value)
      || value.split("/").some(part => !part || part === "." || part === "..") || !/\.(jpe?g)$/i.test(value))
    throw new BestFacesetError("人脸成员路径无效");
  return value;
}
function range(offset = 0, limit = 120) {
  if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 500)
    throw new BestFacesetError("每次读取或分析最多 500 张，请使用有效批次范围", "BEST_FACESET_RANGE_INVALID");
  return { offset, limit };
}
function references(values = []) {
  if (!Array.isArray(values) || values.length > 32 || new Set(values).size !== values.length)
    throw new BestFacesetError("请确认至多 32 张不同的同人参照", "BEST_FACESET_REFERENCES_INVALID");
  return values.map(member);
}
function digest(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

export class BestFacesetManager {
  constructor({ paths = PATHS, runProcess = runHelperProcess } = {}) { this.paths = paths; this.runProcess = runProcess; this.previewEntries = new Map(); }
  async status(options = {}) {
    const result = await this.helper("status", [], options);
    if (result.analysisBatchLimit !== 500 || !["foreground_tenengrad", "efficient-fiqa"].includes(result.qualityDefault))
      throw new BestFacesetError("质量资源状态无效", "BEST_FACESET_RESPONSE_INVALID", 422);
    return result;
  }
  roots(side) {
    if (!["src", "dst"].includes(side)) throw new BestFacesetError("数据集类型无效", "SIDE_INVALID");
    return { aligned: path.join(this.paths.workspaceRoot, `data_${side}`, "aligned"),
      frames: path.join(this.paths.workspaceRoot, `data_${side}`),
      plans: path.join(this.paths.runtimeRoot, "best-facesets", side),
      outputs: path.join(this.paths.workspaceRoot, `data_${side}`, "best-training") };
  }
  planPath(side, id) {
    if (!PLAN.test(id)) throw new BestFacesetError("训练子集记录编号无效", "BEST_FACESET_PLAN_INVALID");
    return path.join(this.roots(side).plans, id);
  }
  outputPath(side, id) { this.planPath(side, id); return path.join(this.roots(side).outputs, `best-training-${id.slice(5)}`); }
  async plain(target, { directory = false, create = false } = {}) {
    const root = this.paths.workspaceRoot;
    assertWithin(root, target, "训练子集路径");
    const components = [root];
    for (const piece of path.relative(root, target).split(path.sep).filter(Boolean)) components.push(path.join(components.at(-1), piece));
    for (const [index, component] of components.entries()) {
      let info;
      try { info = await lstat(component); } catch (error) {
        if (error.code !== "ENOENT" || !create || index === 0) throw error;
        await mkdir(component); info = await lstat(component);
      }
      const wantsDirectory = index < components.length - 1 || directory;
      if (info.isSymbolicLink() || (wantsDirectory ? !info.isDirectory() : !info.isFile()))
        throw new BestFacesetError("训练子集路径包含链接或不支持的文件类型", "BEST_FACESET_PATH_UNSAFE");
    }
    assertWithin(await realpath(root), await realpath(target), "训练子集真实路径");
    return target;
  }
  async metadata(side, id) {
    const directory = this.planPath(side, id); await this.plain(directory, { directory: true });
    const file = path.join(directory, "plan.json"); await this.plain(file);
    if ((await lstat(file)).size > 1024 * 1024) throw new BestFacesetError("训练子集记录过大");
    const data = JSON.parse(await readFile(file, "utf8"));
    const expected = this.roots(side).aligned;
    if (data.schemaVersion !== 1 || data.planId !== id || path.resolve(data.dataset ?? "") !== expected
        || !Number.isInteger(data.total) || data.total < 1 || data.total > 500000 || !HASH.test(data.datasetFingerprint)
        || !["foreground_tenengrad", "efficient-fiqa"].includes(data.qualityModel))
      throw new BestFacesetError("训练子集记录与当前项目不匹配", "BEST_FACESET_PLAN_INVALID", 409);
    references(data.confirmedReferenceMembers);
    return data;
  }
  async helper(action, args = [], { signal, onProgress } = {}) {
    let pending = "";
    const result = await this.runProcess(this.paths.python,
      [path.join(this.paths.webuiRoot, "python", "best_training_faceset.py"), action, ...args], {
        cwd: this.paths.repositoryRoot, env: buildDflEnvironment("pytorch"), signal,
        label: "最佳训练人脸", timeoutMs: 30 * 60_000, maxBytes: 8 * 1024 * 1024,
        onStderr: chunk => {
          pending += chunk.toString("utf8");
          const lines = pending.split(/\r?\n/); pending = lines.pop().slice(-64000);
          for (const line of lines) if (line.startsWith("DFL_PROGRESS ")) {
            try { onProgress?.(JSON.parse(line.slice(13))); } catch { /* Other model diagnostics have no protocol authority. */ }
          }
        },
      });
    if (result.code !== 0) {
      const stderr = String(result.stderr ?? "").trim();
      const conflict = /(?:^|\n)(?:[\w.]+\.)?PlanConflict:\s*([^\r\n]+)\s*$/.exec(stderr);
      throw new BestFacesetError(conflict ? conflict[1].slice(0, 1000) : stderr.slice(-2000) || "最佳训练人脸处理失败",
        conflict ? "BEST_FACESET_CONFLICT" : "BEST_FACESET_PROCESS_FAILED", conflict ? 409 : 422);
    }
    try { return JSON.parse(result.stdout.trim()); }
    catch { throw new BestFacesetError("训练子集工具返回了无效记录", "BEST_FACESET_RESPONSE_INVALID", 422); }
  }
  async inspect(side, id, options = {}) {
    const selected = range(options.offset, options.limit); const meta = await this.metadata(side, id);
    const status = options.status ?? null;
    if (status && !CLASSES.has(status)) throw new BestFacesetError("查看分类无效");
    const result = await this.helper("inspect", ["--plan", this.planPath(side, id), "--offset", String(selected.offset), "--limit", String(selected.limit), ...(status ? ["--status", status] : [])], options);
    if (result.schemaVersion !== 1 || result.planId !== id || result.total !== meta.total || !Array.isArray(result.items) || result.items.length > selected.limit
        || !Number.isInteger(result.completedCount) || result.completedCount < 0 || result.completedCount > result.total
        || result.globalReady !== (result.completedCount === result.total))
      throw new BestFacesetError("训练子集状态无效", "BEST_FACESET_RESPONSE_INVALID", 422);
    for (const [index, item] of result.items.entries()) {
      member(item.member); if (!HASH.test(item.sha256)) throw new BestFacesetError("训练子集图片校验无效");
      const position = item.position ?? (status ? NaN : selected.offset + index);
      if (!Number.isInteger(position) || position < 0 || position >= meta.total) throw new BestFacesetError("训练子集成员位置无效");
      this.previewEntries.set(`${side}/${id}/${position}`, { ...item, position, fingerprint: meta.datasetFingerprint });
    }
    while (this.previewEntries.size > 5000) this.previewEntries.delete(this.previewEntries.keys().next().value);
    const publication = await this.publication(side, id, { verifyFiles: false });
    if (publication) {
      publication.trainingAllowed = meta.qualityModel === "foreground_tenengrad" || (await this.status(options)).efficientFiqa?.admission === "production";
      publication.trainingAdmissionReason = publication.trainingAllowed ? null : "Efficient-FIQA 本次未显示替换收益，结果仅用于对照评测";
    }
    let sourceValidation = {};
    if (options.verifySource === true) {
      try { const verification = await this.helper("verify-source", ["--plan", this.planPath(side, id)], options);
        sourceValidation = { sourceValid: verification.sourceValid === true && verification.planId === id, sourceValidationMessage: null }; }
      catch (error) { sourceValidation = { sourceValid: false, sourceValidationMessage: error.message }; }
    }
    return { ...result, ...sourceValidation, state: result.state === "published" && !publication ? "withdrawn" : result.state, side, targetCount: meta.targetCount, qualityModel: meta.qualityModel,
      identityReferences: meta.confirmedReferenceMembers, publication, selectedDirectory: publication ? path.join(publication.outputDirectory, "selected") : null,
      items: result.items.map((item, index) => ({ ...item, position: item.position ?? selected.offset + index,
        imageUrl: `/api/best-facesets/${side}/plans/${id}/files/${item.position ?? selected.offset + index}/original` })) };
  }
  async list(side) {
    const directory = this.roots(side).plans;
    if (!(await pathExists(directory))) return [];
    await this.plain(directory, { directory: true });
    const results = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (!PLAN.test(entry.name)) continue;
      const meta = await this.metadata(side, entry.name);
      const publication = meta.state === "published" ? await this.publication(side, entry.name, { verifyFiles: false }) : null;
      results.push({ planId: entry.name, state: meta.state === "published" && !publication ? "withdrawn" : meta.state, total: meta.total, targetCount: meta.targetCount,
        qualityModel: meta.qualityModel, identityReferences: meta.confirmedReferenceMembers, createdAt: meta.createdAt ?? null,
        parentPlanId: meta.parentPlanId ?? null, reviewRootPlanId: meta.reviewRootPlanId ?? entry.name, reviewRevision: meta.reviewRevision ?? 0, reviewVersion: meta.reviewVersion ?? 0,
        counts: meta.selection?.counts ?? meta.publication?.counts ?? null });
    }
    return results.sort((a, b) => String(b.createdAt ?? "").localeCompare(String(a.createdAt ?? "")));
  }
  async create(side, { targetCount = 2000, identityReferences = [], qualityModel = "foreground_tenengrad", ...options } = {}) {
    if (!Number.isInteger(targetCount) || targetCount < 1 || targetCount > 500000) throw new BestFacesetError("数量上限须为 1–500000", "BEST_FACESET_TARGET_INVALID");
    if (!["foreground_tenengrad", "efficient-fiqa"].includes(qualityModel)) throw new BestFacesetError("质量模型不受支持", "BEST_FACESET_QUALITY_INVALID");
    const refs = references(identityReferences), root = this.roots(side);
    await this.plain(root.aligned, { directory: true }); await this.plain(root.plans, { directory: true, create: true });
    for (const name of refs) await this.plain(path.join(root.aligned, name));
    const result = await this.helper("init", ["--input", root.aligned, "--plan-root", root.plans, "--frames", root.frames,
      "--target-count", String(targetCount), "--quality-model", qualityModel, ...refs.flatMap(name => ["--reference", name])], options);
    this.planPath(side, result.planId);
    return this.inspect(side, result.planId, options);
  }
  async analyze(side, id, { offset = 0, limit = 500, device = "cpu", ...options } = {}) {
    range(offset, limit); if (!["cpu", "cuda"].includes(device)) throw new BestFacesetError("推理设备无效");
    await this.metadata(side, id);
    await this.helper("analyze", ["--plan", this.planPath(side, id), "--offset", String(offset), "--limit", String(limit), "--device", device], options);
    return this.inspect(side, id, { ...options, offset, limit });
  }
  async analyzeAll(side, id, { device = "cpu", signal, onProgress } = {}) {
    let state = await this.inspect(side, id, { signal });
    while (!state.globalReady) {
      const offset = state.nextOffset;
      state = await this.analyze(side, id, { offset, limit: 500, device, signal,
        onProgress: update => onProgress?.({ ...update, current: offset + (update.current ?? 0), total: state.total,
          detail: `当前批 ${offset + 1}–${Math.min(offset + 500, state.total)} / ${state.total}；每批最多 500 张` }) });
      onProgress?.({ stage: "全量特征分析", current: state.completedCount, total: state.total,
        detail: `已分析 ${state.completedCount} 张，剩余 ${state.total - state.completedCount} 张` });
    }
    return state;
  }
  async select(side, id, { identityReferences = [], ...options } = {}) {
    const refs = references(identityReferences); const state = await this.inspect(side, id, options);
    if (!state.globalReady) throw new BestFacesetError(`还剩 ${state.total - state.completedCount} 张待分析；请继续第 ${state.nextOffset + 1} 张起的批次`, "BEST_FACESET_INCOMPLETE", 409);
    if (state.parentPlanId) throw new BestFacesetError("人工复核版本请使用保留、排除或暂缓决定；重新筛选需建立新计划。", "BEST_FACESET_REVIEW_IMMUTABLE", 409);
    await this.helper("confirm", ["--plan", this.planPath(side, id), ...refs.flatMap(name => ["--reference", name])], options);
    await this.helper("finalize", ["--plan", this.planPath(side, id)], options);
    return this.inspect(side, id, options);
  }
  async publish(side, id, { dryRun = false, expectedReferences, expectedRevision, ...options } = {}) {
    const state = await this.inspect(side, id, options);
    if (state.qualityModel !== "foreground_tenengrad") throw new BestFacesetError("Efficient-FIQA 本次未显示替换收益，计划仅供对照评测，不能发布为生产训练子集", "BEST_FACESET_COMPARISON_ONLY", 409);
    if (!Array.isArray(expectedReferences)) throw new BestFacesetError("发布必须携带当前已确认的身份参照，请刷新选集后重试", "BEST_FACESET_REFERENCES_REQUIRED", 409);
    const expected = references(expectedReferences);
    const recorded = new Set(state.identityReferences);
    if (expected.length !== recorded.size || expected.some(name => !recorded.has(name)))
      throw new BestFacesetError("身份参照已变化，请重新生成代表性训练子集后再发布", "BEST_FACESET_REFERENCES_CHANGED", 409);
    if (!state.globalReady || !["finalized", "published"].includes(state.state)) throw new BestFacesetError("请先完成全量分析与代表性选集", "BEST_FACESET_NOT_SELECTED", 409);
    if (state.parentPlanId && (!Number.isSafeInteger(expectedRevision) || expectedRevision !== state.reviewRevision))
      throw new BestFacesetError("复核版本已变化，请刷新后重新预演发布。", "BEST_FACESET_REVISION_CONFLICT", 409);
    const root = this.roots(side); await this.plain(root.outputs, { directory: true, create: true });
    const result = await this.helper("publish", ["--plan", this.planPath(side, id), "--output-root", root.outputs,
      ...(state.parentPlanId ? ["--expected-revision", String(expectedRevision)] : []), ...(dryRun ? ["--dry-run"] : [])], options);
    if (dryRun) {
      if (result.dryRun !== true || path.resolve(result.outputDirectory ?? "") !== this.outputPath(side, id)) throw new BestFacesetError("发布预演记录无效");
      return result;
    }
    return this.publication(side, id);
  }
  async publication(side, id, { verifyFiles = true } = {}) {
    const directory = this.outputPath(side, id); if (!(await pathExists(directory))) return null;
    await this.plain(directory, { directory: true }); const file = path.join(directory, "receipt.json"); await this.plain(file);
    if ((await lstat(file)).size > 256 * 1024 * 1024) throw new BestFacesetError("发布记录过大");
    const receipt = JSON.parse(await readFile(file, "utf8"));
    if (receipt.schemaVersion === 1 && receipt.kind === "best-training-faceset" && receipt.planId === id
        && receipt.batchId === path.basename(directory) && receipt.state === "withdrawn") return null;
    const metadata = await this.metadata(side, id);
    if (receipt.schemaVersion !== 1 || receipt.kind !== "best-training-faceset" || receipt.planId !== id
        || receipt.batchId !== path.basename(directory) || receipt.state !== "committed"
        || path.resolve(receipt.source?.dataset ?? "") !== this.roots(side).aligned || !HASH.test(receipt.source?.fingerprint)
        || !Array.isArray(receipt.entries) || receipt.entries.length !== metadata.total || receipt.source.fingerprint !== metadata.datasetFingerprint
        || metadata.globalSelection !== true || !["finalized", "published"].includes(metadata.state))
      throw new BestFacesetError("训练子集发布记录无效或已撤回", "BEST_FACESET_RECEIPT_INVALID", 409);
    const counts = { selected: 0, review: 0, rejected: 0 }; const seen = new Set();
    const expectedFiles = new Set();
    for (const entry of receipt.entries) {
      const name = member(entry.member); const destination = entry.destination;
      if (!CLASSES.has(entry.status) || destination !== `${entry.status}/${name}` || !HASH.test(entry.sha256)
          || seen.has(name.toLowerCase())) throw new BestFacesetError("训练子集成员记录无效", "BEST_FACESET_RECEIPT_INVALID", 409);
      seen.add(name.toLowerCase()); counts[entry.status]++;
      expectedFiles.add(destination);
      if (verifyFiles) {
        const target = path.join(directory, ...destination.split("/")); await this.plain(target);
        if (digest(await readFile(target)) !== entry.sha256) throw new BestFacesetError("已发布人脸已变化，请重新分析", "BEST_FACESET_COPY_CHANGED", 409);
        if (entry.sidecarSha256) {
          if (!HASH.test(entry.sidecarSha256)) throw new BestFacesetError("辅助记录校验无效");
          const sidecar = `${target}.landmarks.json`; await this.plain(sidecar);
          if (digest(await readFile(sidecar)) !== entry.sidecarSha256) throw new BestFacesetError("已发布辅助记录已变化", "BEST_FACESET_COPY_CHANGED", 409);
          expectedFiles.add(`${destination}.landmarks.json`);
        }
      }
    }
    if (Object.keys(counts).some(key => counts[key] !== receipt.counts?.[key] || counts[key] !== metadata.selection?.counts?.[key]))
      throw new BestFacesetError("发布数量与完整计划成员不匹配", "BEST_FACESET_RECEIPT_INVALID", 409);
    if (verifyFiles) {
      const actualFiles = new Set();
      const inspectDirectory = async (folder, prefix) => {
        await this.plain(folder, { directory: true });
        for (const entry of await readdir(folder, { withFileTypes: true })) {
          const relative = `${prefix}/${entry.name}`;
          if (entry.isDirectory()) await inspectDirectory(path.join(folder, entry.name), relative);
          else if (entry.isFile() && !entry.isSymbolicLink()) actualFiles.add(relative);
          else throw new BestFacesetError("发布目录包含链接或特殊文件", "BEST_FACESET_PATH_UNSAFE");
        }
      };
      for (const status of CLASSES) await inspectDirectory(path.join(directory, status), status);
      if (actualFiles.size !== expectedFiles.size || [...actualFiles].some(file => !expectedFiles.has(file)))
        throw new BestFacesetError("发布目录包含新增或缺失文件", "BEST_FACESET_COPY_CHANGED", 409);
    }
    return { schemaVersion: 1, planId: id, outputDirectory: directory, receiptPath: file, counts, selectedDirectory: path.join(directory, "selected"),
      allOriginalsRetained: true, independentCopies: true };
  }
  async file(side, id, position, kind = "original") {
    if (!["original", "published"].includes(kind)) throw new BestFacesetError("预览类型无效");
    range(position, 1); const metadata = await this.metadata(side, id);
    let item = this.previewEntries.get(`${side}/${id}/${position}`);
    if (!item || item.fingerprint !== metadata.datasetFingerprint) item = (await this.inspect(side, id, { offset: position, limit: 1 })).items[0];
    if (!item) throw new BestFacesetError("预览图片不存在", "BEST_FACESET_FILE_MISSING", 404);
    const publication = kind === "published" ? await this.publication(side, id, { verifyFiles: false }) : null;
    const base = kind === "original" ? this.roots(side).aligned : publication?.outputDirectory;
    if (!base || kind === "published" && !CLASSES.has(item.status)) throw new BestFacesetError("结果尚未发布");
    const file = path.join(base, ...(kind === "published" ? [item.status] : []), ...item.member.split("/")); await this.plain(file);
    const bytes = await readFile(file); if (digest(bytes) !== item.sha256) throw new BestFacesetError("人脸预览来源已改变", "BEST_FACESET_SOURCE_CHANGED", 409);
    return { bytes, path: file, name: item.member, mimeType: "image/jpeg" };
  }
  async recover(side, id, { dryRun = false, ...options } = {}) {
    const publication = await this.publication(side, id);
    if (!publication) throw new BestFacesetError("没有已发布结果可撤回");
    return this.helper("recover", ["--receipt", publication.receiptPath, ...(dryRun ? ["--dry-run"] : [])], options);
  }
  async review(side, id, action, { expectedRevision, requestId, identityReferences = [], members = [], decision, ...options } = {}) {
    if (!["create", "decide", "undo"].includes(action) || !Number.isSafeInteger(expectedRevision) || expectedRevision < 0
      || typeof requestId !== "string" || !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(requestId)) throw new BestFacesetError("复核请求或版本无效");
    await this.metadata(side, id);
    const refs = references(identityReferences);
    const args = ["--plan", this.planPath(side, id), "--expected-revision", String(expectedRevision), "--request-id", requestId];
    if (action !== "create") args.push(...refs.flatMap(name => ["--reference", name]));
    if (action === "decide") {
      if (!["keep", "exclude", "defer"].includes(decision) || !Array.isArray(members) || !members.length || members.length > 500 || new Set(members).size !== members.length)
        throw new BestFacesetError("每次复核须选择 1–500 张不同的人脸与有效决定");
      args.push("--decision", decision, ...members.map(member).flatMap(name => ["--member", name]));
    }
    const result = await this.helper(`review-${action}`, args, options);
    const nextId = result.planId; this.planPath(side, nextId);
    for (const key of this.previewEntries.keys()) if (key.startsWith(`${side}/${nextId}/`)) this.previewEntries.delete(key);
    return { ...await this.inspect(side, nextId, options), reviewOperation: result.reviewOperation };
  }
  async requireSelected(target) {
    const relative = path.relative(this.paths.workspaceRoot, target).split(path.sep);
    const index = relative.findIndex(piece => /^best-training-[a-f0-9]{32}$/.test(piece));
    if (index === -1 && !relative.some(piece => piece === "best-training" || piece.startsWith("best-training-"))) return null;
    if (relative.length !== 4 || index !== 2 || relative[1] !== "best-training" || relative[3] !== "selected" || !/^data_(src|dst)$/.test(relative[0]))
      throw new BestFacesetError("最佳训练人脸只允许使用完整发布的 selected 目录", "BEST_FACESET_TRAINING_PATH_INVALID", 409);
    const side = relative[0].slice(5), id = `plan-${relative[2].slice(14)}`;
    const metadata = await this.metadata(side, id);
    if (metadata.qualityModel !== "foreground_tenengrad" && (await this.status()).efficientFiqa?.admission !== "production")
      throw new BestFacesetError("Efficient-FIQA 本次未显示替换收益，结果仅供对照评测，不能作为推荐训练子集", "BEST_FACESET_COMPARISON_ONLY", 409);
    const publication = await this.publication(side, id);
    if (!publication || publication.counts.selected < 1) throw new BestFacesetError("推荐训练子集为空或尚未发布", "BEST_FACESET_TRAINING_EMPTY", 409);
    const source = await this.helper("verify-source", ["--plan", this.planPath(side, id)]);
    if (source.sourceValid !== true || source.planId !== id) throw new BestFacesetError("原始素材校验失败，请重新检查选集。", "BEST_FACESET_SOURCE_CHANGED", 409);
    return publication;
  }
}
export const bestFacesetManager = new BestFacesetManager();
