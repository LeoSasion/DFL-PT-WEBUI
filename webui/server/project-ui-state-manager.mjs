import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile, readdir, rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { PATHS } from "./paths.mjs";
import { bestFacesetManager } from "./best-faceset-manager.mjs";

const HASH = /^[a-f0-9]{64}$/;
const PLAN = /^plan-[a-f0-9]{32}$/;
const CLASSES = new Set(["all", "selected", "review", "rejected"]);
export class ProjectUiStateError extends Error {
  constructor(message, code = "PROJECT_UI_STATE_INVALID", status = 400) { super(message); this.code = code; this.status = status; }
}
const fail = (message, code, status) => { throw new ProjectUiStateError(message, code, status); };
function safeMember(name) {
  if (typeof name !== "string" || name.length > 1024 || /[\\:\u0000-\u001f]/.test(name)
    || name.split("/").some(part => !part || part === "." || part === "..") || !/\.jpe?g$/i.test(name)) fail("身份参照文件名无效");
  return name;
}
function revision(value) {
  if (!Number.isSafeInteger(value) || value < 0) fail("草稿版本无效");
  return value;
}

export class ProjectUiStateManager {
  constructor({ paths = PATHS, facesetManager = bestFacesetManager, inventory } = {}) {
    this.paths = paths; this.facesets = facesetManager; this.readInventory = inventory ?? (side => this.inventory(side)); this.writes = new Map();
  }
  project(key, { required = false } = {}) {
    if ((required && typeof key !== "string") || (key != null && key !== this.paths.workspaceRoot))
      fail("项目已切换，请重新读取当前项目的选择。", "PROJECT_UI_STATE_PROJECT_CHANGED", 409);
  }
  side(side) { if (!["src", "dst"].includes(side)) fail("请选择 SRC 或 DST"); return side; }
  file(name) { return path.join(this.paths.runtimeRoot, "ui-state", name); }
  async inventory(side) {
    const root = this.facesets.roots(this.side(side)).aligned;
    try { await this.facesets.plain(root, { directory: true }); } catch (error) {
      if (error.code === "ENOENT") return { names: [], fingerprint: createHash("sha256").update("[]").digest("hex") };
      throw error;
    }
    const records = [];
    const walk = async (folder, prefix = "") => {
      await this.facesets.plain(folder, { directory: true });
      for (const entry of (await readdir(folder, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name, "en"))) {
        const name = prefix + entry.name, target = path.join(folder, entry.name);
        if (entry.isSymbolicLink()) fail("人脸集包含链接，请先整理素材。", "PROJECT_UI_STATE_SOURCE_UNSAFE", 409);
        if (entry.isDirectory()) { await walk(target, `${name}/`); continue; }
        if (!/\.jpe?g$/i.test(name)) continue;
        safeMember(name); await this.facesets.plain(target);
        const before = await lstat(target, { bigint: true });
        const hash = createHash("sha256");
        for await (const chunk of createReadStream(target)) hash.update(chunk);
        const after = await lstat(target, { bigint: true });
        if (before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || before.ino !== after.ino)
          fail("读取时素材发生变化，请重试。", "PROJECT_UI_STATE_SOURCE_CHANGED", 409);
        let sidecarHash = null;
        try {
          const sidecar = `${target}.landmarks.json`; await this.facesets.plain(sidecar);
          sidecarHash = createHash("sha256").update(await readFile(sidecar)).digest("hex");
        } catch (error) { if (error.code !== "ENOENT") throw error; }
        records.push([name, hash.digest("hex"), sidecarHash]);
      }
    };
    await walk(root);
    return { names: records.map(record => record[0]), fingerprint: createHash("sha256").update(JSON.stringify(records)).digest("hex") };
  }
  async read(name, fallback) {
    const file = this.file(name);
    try {
      await this.facesets.plain(file);
      if ((await lstat(file)).size > 256 * 1024) fail("项目界面记录过大");
      const value = JSON.parse(await readFile(file, "utf8"));
      if (value.schemaVersion !== 1 || value.projectKey !== this.paths.workspaceRoot) fail("项目界面记录不匹配", "PROJECT_UI_STATE_INVALID", 409);
      revision(value.revision); return value;
    } catch (error) { if (error.code === "ENOENT") return fallback; throw error; }
  }
  async write(name, value) {
    const file = this.file(name); await this.facesets.plain(path.dirname(file), { directory: true, create: true });
    try { await this.facesets.plain(file); } catch (error) { if (error.code !== "ENOENT") throw error; }
    const temporary = `${file}.${randomUUID()}.tmp`;
    try { await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" }); await rename(temporary, file); }
    finally { await unlink(temporary).catch(error => { if (error.code !== "ENOENT") throw error; }); }
  }
  serial(name, action) {
    const previous = this.writes.get(name) ?? Promise.resolve();
    const pending = previous.catch(() => {}).then(action);
    this.writes.set(name, pending);
    void pending.finally(() => { if (this.writes.get(name) === pending) this.writes.delete(name); }).catch(() => {});
    return pending;
  }
  empty() { return { schemaVersion: 1, projectKey: this.paths.workspaceRoot, revision: 0 }; }
  normalizeDraft(draft) {
    if (!draft || typeof draft !== "object" || Array.isArray(draft)) fail("选集草稿无效");
    const refs = draft.identityReferences ?? [];
    if (!Array.isArray(refs) || refs.length > 32 || new Set(refs).size !== refs.length) fail("请选择至多 32 张不同的身份参照");
    const planId = draft.planId ?? null;
    if (planId !== null && !PLAN.test(planId)) fail("选集记录无效");
    const offset = draft.offset ?? 0, targetCount = draft.targetCount ?? 2000;
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > 500000 || !Number.isInteger(targetCount) || targetCount < 1 || targetCount > 500000) fail("选集浏览范围或数量无效");
    if (!CLASSES.has(draft.category ?? "all") || !["foreground_tenengrad", "efficient-fiqa"].includes(draft.qualityModel ?? "foreground_tenengrad")) fail("选集分类或质量方案无效");
    if (draft.sourceFingerprint != null && !HASH.test(draft.sourceFingerprint)) fail("素材校验无效");
    return { planId, identityReferences: refs.map(safeMember), confirmed: draft.confirmed === true, offset,
      category: draft.category ?? "all", targetCount, qualityModel: draft.qualityModel ?? "foreground_tenengrad", sourceFingerprint: draft.sourceFingerprint ?? null };
  }
  async presentDraft(side, value, snapshot) {
    const current = snapshot ?? await this.readInventory(side);
    const draft = value.draft ? this.normalizeDraft(value.draft) : null;
    let sourceValid = !draft || draft.sourceFingerprint === current.fingerprint;
    let message = sourceValid ? null : "素材已变化，上次身份确认已失效，请重新检查参照。";
    if (draft && draft.identityReferences.some(name => !current.names.includes(name))) { sourceValid = false; message = "部分身份参照已移走，请重新选择。"; }
    let planSourceValid = draft?.planId ? false : null;
    if (draft?.planId) {
      try {
        await this.facesets.metadata(side, draft.planId);
        const source = await this.facesets.helper("verify-source", ["--plan", this.facesets.planPath(side, draft.planId)]);
        if (source.sourceValid !== true || source.planId !== draft.planId) fail("选集来源已失效");
        planSourceValid = true;
      }
      catch { sourceValid = false; message = "上次选集记录暂不可用，请选择其他记录或重新创建。"; }
    }
    const identityConfirmationValid = Boolean(sourceValid && draft?.confirmed && draft.identityReferences.length);
    return { ...value, draft: draft ? { ...draft, confirmed: identityConfirmationValid } : null,
      sourceFingerprint: current.fingerprint, sourceValid, planSourceValid, identityConfirmationValid, validationMessage: message };
  }
  async draft(side, { projectKey } = {}) {
    this.side(side); this.project(projectKey);
    return this.presentDraft(side, await this.read(`best-${side}.json`, { ...this.empty(), draft: null }));
  }
  async saveDraft(side, { projectKey, expectedRevision, draft } = {}) {
    this.side(side); this.project(projectKey, { required: true }); revision(expectedRevision);
    const normalized = draft === null ? null : this.normalizeDraft(draft);
    const name = `best-${side}.json`;
    return this.serial(name, async () => {
      const previous = await this.read(name, { ...this.empty(), draft: null });
      if (previous.revision !== expectedRevision) fail("草稿已在其他页面更新，请重新读取后继续。", "PROJECT_UI_STATE_REVISION_CONFLICT", 409);
      const snapshot = await this.readInventory(side);
      if (normalized?.confirmed && (normalized.sourceFingerprint !== snapshot.fingerprint || normalized.identityReferences.some(item => !snapshot.names.includes(item))))
        fail("素材或身份参照已变化，请重新确认。", "PROJECT_UI_STATE_SOURCE_CHANGED", 409);
      if (normalized?.confirmed && !normalized.identityReferences.length) fail("请先选择身份参照");
      if (normalized?.planId) await this.facesets.metadata(side, normalized.planId);
      const value = { ...this.empty(), revision: previous.revision + 1, updatedAt: new Date().toISOString(), draft: normalized };
      await this.write(name, value); return this.presentDraft(side, value, snapshot);
    });
  }
  async presentInput(side, selection = { kind: "aligned" }) {
    const snapshot = await this.readInventory(side);
    if (selection.kind === "aligned") return { kind: "aligned", planId: null, path: `data_${side}/aligned`, count: snapshot.names.length,
      valid: snapshot.names.length > 0, validationMessage: snapshot.names.length ? null : "尚无人脸素材，请先提取。", sourceFingerprint: snapshot.fingerprint };
    if (selection.kind !== "selected" || !PLAN.test(selection.planId)) fail("训练输入记录无效", "PROJECT_UI_STATE_INVALID", 409);
    try {
      const target = path.join(this.facesets.outputPath(side, selection.planId), "selected");
      const publication = await this.facesets.requireSelected(target);
      if (snapshot.fingerprint !== selection.sourceFingerprint) fail("原始素材已变化，请重新检查选集。", "PROJECT_UI_STATE_SOURCE_CHANGED", 409);
      return { ...selection, path: path.relative(this.paths.workspaceRoot, target).split(path.sep).join("/"), count: publication.counts.selected, valid: true, validationMessage: null };
    } catch (error) { return { ...selection, valid: false, validationMessage: error.message }; }
  }
  async inputs({ projectKey } = {}) {
    this.project(projectKey);
    const value = await this.read("training-inputs.json", { ...this.empty(), sides: {} });
    const [src, dst] = await Promise.all([this.presentInput("src", value.sides?.src), this.presentInput("dst", value.sides?.dst)]);
    return { ...value, sides: { src, dst } };
  }
  async saveInput({ projectKey, expectedRevision, side, selection } = {}) {
    this.project(projectKey, { required: true }); this.side(side); revision(expectedRevision);
    if (!selection || !["aligned", "selected"].includes(selection.kind)) fail("请选择原始 aligned 或已发布 selected");
    return this.serial("training-inputs.json", async () => {
      const previous = await this.read("training-inputs.json", { ...this.empty(), sides: {} });
      if (previous.revision !== expectedRevision) fail("训练输入已在其他页面更新，请重新读取。", "PROJECT_UI_STATE_REVISION_CONFLICT", 409);
      let next = { kind: "aligned" };
      if (selection.kind === "selected") {
        if (!PLAN.test(selection.planId)) fail("请选择已发布选集");
        const publication = await this.facesets.requireSelected(path.join(this.facesets.outputPath(side, selection.planId), "selected"));
        const meta = await this.facesets.metadata(side, selection.planId);
        next = { kind: "selected", planId: selection.planId, count: publication.counts.selected,
          path: path.relative(this.paths.workspaceRoot, publication.selectedDirectory).split(path.sep).join("/"),
          sourceFingerprint: (await this.readInventory(side)).fingerprint, createdAt: meta.createdAt ?? null, identityReferences: meta.confirmedReferenceMembers };
      }
      const value = { ...this.empty(), revision: previous.revision + 1, updatedAt: new Date().toISOString(), sides: { ...previous.sides, [side]: next } };
      await this.write("training-inputs.json", value); return this.inputs({ projectKey });
    });
  }
}
export const projectUiStateManager = new ProjectUiStateManager();
