import { randomBytes, createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, readdir, realpath, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { PATHS } from "./paths.mjs";
import { terminateChildProcessTree } from "./process-tree.mjs";
import { runHelperProcess } from "./helper-process.mjs";

const TASK_ID = /^rst-[a-f0-9]{32}$/;
const IMAGE_NAME = /^[^<>:"/\\|?*\u0000-\u001f]{1,220}\.(?:png|jpe?g)$/i;
const MAX_IMAGES = 500;
const MAX_BYTES = 50 * 1024 * 1024;
const MODELS = ["swinir-psnr", "realesrgan-x4plus", "gfpgan-v1.4"];
const WEIGHTS = {
  "swinir-psnr": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
  "realesrgan-x4plus": "RealESRGAN_x4plus.pth", "gfpgan-v1.4": "GFPGANv1.4.pth",
};
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const now = () => new Date().toISOString();
function within(root, target) {
  const relative = path.relative(path.resolve(root), path.resolve(target));
  return !relative || relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export class RestorationError extends Error {
  constructor(message, code = "RESTORATION_INVALID", status = 400) {
    super(message); this.name = "RestorationError"; this.code = code; this.status = status;
  }
}

async function atomicJson(target, value) {
  const temporary = `${target}.${randomBytes(6).toString("hex")}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  await rename(temporary, target);
}

export class RestorationManager {
  constructor({ workspaceRoot = PATHS.workspaceRoot, runtimeRoot = PATHS.runtimeRoot,
    projectRoot = PATHS.repositoryRoot, pythonPath = PATHS.python,
    assetsRoot = path.join(projectRoot, "workspace", ".vision-models", "restoration"),
    assertCanStart = async () => {}, onActiveChange = () => {}, processRunner = null,
    terminateProcessTree = terminateChildProcessTree, runProcess = runHelperProcess } = {}) {
    this.workspaceRoot = path.resolve(workspaceRoot); this.runtimeRoot = path.resolve(runtimeRoot);
    this.projectRoot = path.resolve(projectRoot); this.pythonPath = pythonPath; this.assetsRoot = assetsRoot;
    if (this.runtimeRoot !== path.join(this.workspaceRoot, ".webui")) throw new RestorationError("修复运行目录必须位于当前项目 .webui");
    this.root = path.join(this.runtimeRoot, "restoration");
    this.assertCanStart = assertCanStart; this.onActiveChange = onActiveChange;
    this.processRunner = processRunner; this.tasks = new Map(); this.activeId = null; this.controllers = new Map(); this.pending = new Map();
    this.terminateProcessTree = terminateProcessTree; this.stopUnconfirmed = null;
    this.runProcess = runProcess;
  }

  async safePath(target, { directory = false, create = false } = {}) {
    target = path.resolve(target);
    if (!within(this.workspaceRoot, target)) throw new RestorationError("修复路径超出当前项目", "RESTORATION_PATH_UNSAFE");
    const parts = path.relative(this.workspaceRoot, target).split(path.sep).filter(Boolean);
    let current = this.workspaceRoot;
    for (const part of [null, ...parts]) {
      if (part !== null) current = path.join(current, part);
      let info;
      try { info = await lstat(current); }
      catch (error) {
        if (error.code !== "ENOENT" || !create) throw error;
        await mkdir(current); info = await lstat(current);
      }
      if (info.isSymbolicLink() || current !== target && !info.isDirectory()) throw new RestorationError("修复路径不是安全的普通目录", "RESTORATION_PATH_UNSAFE");
    }
    const [workspaceReal, targetReal] = await Promise.all([realpath(this.workspaceRoot), realpath(target)]);
    if (!within(workspaceReal, targetReal)) throw new RestorationError("修复路径超出真实项目目录", "RESTORATION_PATH_UNSAFE");
    if (directory && !(await lstat(target)).isDirectory()) throw new RestorationError("修复目录不存在", "RESTORATION_PATH_UNSAFE");
    return target;
  }

  async modelAssetsRoot(modelId) {
    const research = path.join(this.projectRoot, "workspace", ".vision-models", "restoration");
    if (path.resolve(this.assetsRoot) !== research) return this.assetsRoot;
    const installed = path.join(this.projectRoot, "_internal", "vision_models", "production", modelId);
    try {
      const info = await lstat(installed);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new RestorationError("模型资源目录须为独立普通目录", "RESTORATION_ASSET_INVALID");
      return installed;
    } catch (error) { if (error.code !== "ENOENT") throw error; return research; }
  }

  async initialize() {
    await this.safePath(this.root, { directory: true, create: true });
    for (const name of ["tasks", "requests", "staging", "outputs"]) await this.safePath(path.join(this.root, name), { directory: true, create: true });
    for (const entry of await readdir(path.join(this.root, "tasks"), { withFileTypes: true })) {
      if (!entry.isFile() || !/^rst-[a-f0-9]{32}\.json$/.test(entry.name)) continue;
      const file = await this.safePath(path.join(this.root, "tasks", entry.name));
      let task;
      try { task = JSON.parse(await readFile(file, "utf8")); } catch { continue; }
      if (!TASK_ID.test(task.taskId) || `${task.taskId}.json` !== entry.name) continue;
      if (["running", "queued"].includes(task.status)) {
        task.status = "interrupted"; task.error = "服务重启中断了修复；原图保留，请重新创建任务。"; task.updatedAt = now();
        await atomicJson(file, task);
      }
      this.tasks.set(task.taskId, task);
    }
  }

  async verifiedBytes(target, expectedSha256) {
    await this.safePath(target);
    const before = await lstat(target, { bigint: true });
    if (!before.isFile() || before.size < 1n || before.size > BigInt(MAX_BYTES)) throw new RestorationError("图片不是 50 MB 以内的普通文件", "RESTORATION_IMAGE_INVALID");
    const handle = await open(target, "r");
    try {
      const opened = await handle.stat({ bigint: true });
      const bytes = await handle.readFile();
      await this.safePath(target);
      const after = await lstat(target, { bigint: true });
      if (before.dev !== opened.dev || before.ino !== opened.ino || after.dev !== opened.dev || after.ino !== opened.ino
        || before.size !== after.size || before.mtimeNs !== after.mtimeNs || bytes.length > MAX_BYTES)
        throw new RestorationError("图片在读取期间变化", "RESTORATION_SOURCE_CHANGED", 409);
      const digest = sha256(bytes);
      if (expectedSha256 && digest !== expectedSha256) throw new RestorationError("图片内容已改变", "RESTORATION_SOURCE_CHANGED", 409);
      return { bytes, digest };
    } finally { await handle.close(); }
  }

  async listInputs({ side, offset = 0, limit = MAX_IMAGES } = {}) {
    if (!["src", "dst"].includes(side) || !Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > MAX_IMAGES)
      throw new RestorationError("请选择源/目标素材及 1–500 张批次", "RESTORATION_SELECTION_INVALID");
    const directory = path.join(this.workspaceRoot, `data_${side}`);
    let entries;
    try { await this.safePath(directory, { directory: true }); entries = await readdir(directory, { withFileTypes: true }); }
    catch (error) { if (error.code !== "ENOENT") throw error; entries = []; }
    const names = entries.filter(entry => entry.isFile() && IMAGE_NAME.test(entry.name)).map(entry => entry.name).sort((a, b) => a.localeCompare(b, "en"));
    const inputs = [];
    for (const name of names.slice(offset, offset + limit)) {
      const { bytes, digest } = await this.verifiedBytes(path.join(directory, name));
      inputs.push({ name, bytes: bytes.length, sha256: digest });
    }
    const models = [];
    for (const id of MODELS) {
      let available = false;
      try { available = (await lstat(path.join(await this.modelAssetsRoot(id), WEIGHTS[id]))).isFile(); } catch {}
      models.push({ id, available, sourceFramesSupported: id !== "gfpgan-v1.4",
        ...(id === "gfpgan-v1.4" ? { guidance: "GFPGAN 需要已确认的 FFHQ 五点对齐；本入口不把任意源帧直接套入。" } : {}) });
    }
    return { side, stage: "source-frames", total: names.length, offset, limit, maxSelected: MAX_IMAGES,
      range: { start: inputs.length ? offset + 1 : 0, end: offset + inputs.length }, inputs, models,
      fingerprint: sha256(Buffer.from(JSON.stringify({ side, offset, limit, inputs }))), defaultModelId: "swinir-psnr" };
  }

  getTask(taskId) {
    if (!TASK_ID.test(taskId)) throw new RestorationError("修复任务 ID 无效");
    const task = this.tasks.get(taskId);
    if (!task) throw new RestorationError("修复任务不存在", "RESTORATION_NOT_FOUND", 404);
    return structuredClone(task);
  }
  activeTasks() { return this.activeId ? [{ id: this.activeId, taskId: this.activeId, status: "running" }] : []; }
  listTasks() { return [...this.tasks.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).map(task => structuredClone(task)); }

  async createTask({ side, names, fingerprint, offset = 0, limit = MAX_IMAGES, modelId = "swinir-psnr" } = {}) {
    if (this.closed || this.activeId) throw new RestorationError("修复服务关闭或已有任务正在运行", "RESTORATION_BUSY", 409);
    if (!MODELS.includes(modelId) || modelId === "gfpgan-v1.4") throw new RestorationError("源帧修复请选择 SwinIR 或 Real-ESRGAN；GFPGAN 需要单独的 FFHQ 对齐流程", "RESTORATION_STAGE_UNSUPPORTED");
    if (!Array.isArray(names) || names.length < 1 || names.length > MAX_IMAGES || names.some(name => typeof name !== "string" || !IMAGE_NAME.test(name))
      || new Set(names.map(name => name.toLowerCase())).size !== names.length) throw new RestorationError("请选择 1–500 张不同源帧", "RESTORATION_SELECTION_INVALID");
    const taskId = `rst-${randomBytes(16).toString("hex")}`;
    this.activeId = taskId;
    let finishStartup;
    this.startupDone = new Promise(resolve => { finishStartup = resolve; });
    try {
      await this.assertCanStart();
      this.onActiveChange(taskId);
      const inventory = await this.listInputs({ side, offset, limit });
      if (this.closed) throw new RestorationError("修复服务正在关闭", "RESTORATION_BUSY", 409);
      if (typeof fingerprint !== "string" || fingerprint !== inventory.fingerprint) throw new RestorationError("源帧已改变，请刷新所选批次", "RESTORATION_SOURCE_CHANGED", 409);
      const inputs = names.map(name => inventory.inputs.find(item => item.name === name));
      if (inputs.some(item => !item)) throw new RestorationError("所选图片不属于当前可见批次", "RESTORATION_SELECTION_INVALID");
      const outputNames = names.map(name => `${path.parse(name).name.toLowerCase()}.png`);
      if (new Set(outputNames).size !== names.length) throw new RestorationError("所选文件转为 PNG 后重名，请分开修复", "RESTORATION_NAME_COLLISION");
      const task = { taskId, side, stage: "source-frames", modelId, status: "running", createdAt: now(), updatedAt: now(),
        selectedCount: inputs.length, inputs, limit: MAX_IMAGES, range: inventory.range, fingerprint,
        progress: { completed: 0, total: inputs.length }, outputs: [], originalsPreserved: true,
        metadataCopied: false, nextStep: "从已完成源图副本重新提取；不能直接导入 aligned。" };
      const request = { taskId, side, modelId, device: "cuda", inputs };
      const requestFile = path.join(this.root, "requests", `${taskId}.json`);
      await atomicJson(requestFile, request);
      await atomicJson(path.join(this.root, "tasks", `${taskId}.json`), task);
      this.tasks.set(taskId, task);
      const controller = new AbortController(); this.controllers.set(taskId, controller);
      const pending = this.execute(task, requestFile, controller.signal).finally(() => this.pending.delete(taskId));
      this.pending.set(taskId, pending);
      finishStartup();
      return this.getTask(taskId);
    } catch (error) { this.activeId = null; this.onActiveChange(null); finishStartup(); throw error; }
  }

  async execute(task, requestFile, signal) {
    try {
      const onProgress = value => { task.progress = { completed: value.completed, total: task.selectedCount }; task.updatedAt = now(); };
      const manifest = this.processRunner ? await this.processRunner(requestFile, { signal, onProgress })
        : await this.runPython(requestFile, { signal, onProgress });
      if (signal.aborted) throw new RestorationError("修复已取消", "RESTORATION_CANCELLED", 499);
      await this.verifyPublication(task, manifest);
      task.status = "completed"; task.outputs = manifest.outputs; task.report = manifest;
      task.progress = { completed: task.selectedCount, total: task.selectedCount };
      task.outputDirectory = path.join(this.root, "outputs", task.taskId, "images");
    } catch (error) {
      if (error.code === "RESTORATION_STOP_UNCONFIRMED") this.stopUnconfirmed = error;
      task.status = (signal.aborted || error.name === "AbortError") && !this.stopUnconfirmed ? "cancelled" : "failed";
      task.error = String(error.message).slice(-4000); task.outputs = [];
    }
    finally {
      task.updatedAt = now();
      try { await atomicJson(path.join(this.root, "tasks", `${task.taskId}.json`), task); }
      catch (error) { task.status = "failed"; task.error = `修复任务状态保存失败：${error.message}`; task.outputs = []; }
      finally {
        this.controllers.delete(task.taskId);
        // An unconfirmed descendant may still use CUDA. Keep the reservation honest.
        if (!this.stopUnconfirmed) { this.activeId = null; this.onActiveChange(null); }
      }
    }
  }

  async runPython(requestFile, { signal, onProgress }) {
    let remainder = "";
    let result;
    try {
      result = await this.runProcess(this.pythonPath, ["-B", path.join(this.projectRoot, "webui", "python", "restoration_task.py"),
        "--request", requestFile, "--workspace", this.workspaceRoot, "--runtime", this.runtimeRoot, "--assets", this.assetsRoot],
      { cwd: this.projectRoot, env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONDONTWRITEBYTECODE: "1" },
        signal, label: "源帧修复", timeoutMs: 90 * 60_000, maxBytes: 4 * 1024 * 1024,
        terminateProcessTree: this.terminateProcessTree,
        onStderr: chunk => {
          remainder += chunk; const lines = remainder.split(/\r?\n/); remainder = lines.pop().slice(-16000);
          for (const line of lines) if (line.startsWith("RESTORATION_PROGRESS ")) {
            try { onProgress(JSON.parse(line.slice(21))); } catch {}
          }
        },
      });
    } catch (error) {
      // The shared runner reports a failed stop even when the child never closes,
      // and retains its process reservation. execute() keeps our task active too.
      if (error.code === "HELPER_STOP_UNCONFIRMED") throw new RestorationError(error.message, "RESTORATION_STOP_UNCONFIRMED", 503);
      if (error.code === "HELPER_TIMEOUT") throw new RestorationError("修复超过安全时限", "RESTORATION_FAILED", 422);
      if (error.code === "HELPER_OUTPUT_TOO_LARGE") throw new RestorationError("修复响应过大", "RESTORATION_FAILED", 422);
      throw error;
    }
    if (result.code !== 0) throw new RestorationError(result.stderr.trim().slice(-16000) || "修复执行失败", "RESTORATION_FAILED", 422);
    try { return JSON.parse(result.stdout); } catch { throw new RestorationError("修复结果不是有效 JSON", "RESTORATION_FAILED", 422); }
  }

  async cancelTask(taskId) {
    const task = this.getTask(taskId);
    if (task.status !== "running") return task;
    this.controllers.get(taskId)?.abort();
    await this.pending.get(taskId);
    return this.getTask(taskId);
  }

  async close() {
    this.closed = true;
    await this.startupDone;
    if (this.activeId && this.tasks.has(this.activeId)) await this.cancelTask(this.activeId);
    await Promise.allSettled([...this.pending.values()]);
    if (this.stopUnconfirmed) throw this.stopUnconfirmed;
  }

  async verifyPublication(task, manifest) {
    if (manifest.taskId !== task.taskId || manifest.side !== task.side || manifest.modelId !== task.modelId
      || manifest.selectedCount !== task.selectedCount || !Array.isArray(manifest.outputs) || manifest.outputs.length !== task.selectedCount
      || manifest.metadataCopied !== false || manifest.originalsPreserved !== true || manifest.atomicBatch !== true)
      throw new RestorationError("修复批次完整性验证失败", "RESTORATION_PUBLICATION_INVALID", 422);
    const directory = await this.safePath(path.join(this.root, "outputs", task.taskId), { directory: true });
    const disk = JSON.parse(await readFile(await this.safePath(path.join(directory, "manifest.json")), "utf8"));
    if (JSON.stringify(disk) !== JSON.stringify(manifest)) throw new RestorationError("修复清单不一致", "RESTORATION_PUBLICATION_INVALID", 422);
    const images = await this.safePath(path.join(directory, "images"), { directory: true });
    const names = new Set();
    const sourceNames = new Set();
    for (const output of manifest.outputs) {
      const input = task.inputs.find(item => item.name === output.sourceName);
      if (typeof output.name !== "string" || !IMAGE_NAME.test(output.name) || !output.name.toLowerCase().endsWith(".png")
        || names.has(output.name.toLowerCase()) || sourceNames.has(output.sourceName) || !input || input.sha256 !== output.inputSha256
        || !/^[a-f0-9]{64}$/.test(output.outputSha256) || output.mode !== "RGB" || output.metadataCopied !== false)
        throw new RestorationError("修复输出图片清单无效", "RESTORATION_PUBLICATION_INVALID", 422);
      names.add(output.name.toLowerCase());
      sourceNames.add(output.sourceName);
      const { bytes } = await this.verifiedBytes(path.join(images, output.name), output.outputSha256);
      if (!bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
        || bytes.readUInt32BE(16) !== output.width || bytes.readUInt32BE(20) !== output.height || bytes[24] !== 8 || bytes[25] !== 2)
        throw new RestorationError("修复图片 RGB/尺寸不一致", "RESTORATION_PUBLICATION_INVALID", 422);
    }
    if ((await readdir(images)).length !== manifest.outputs.length) throw new RestorationError("修复图片数量不一致", "RESTORATION_PUBLICATION_INVALID", 422);
    return images;
  }

  async resolveOutputDirectory(taskId) {
    const task = this.getTask(taskId);
    if (task.status !== "completed") throw new RestorationError("只有完整完成的修复副本可重新提取", "RESTORATION_NOT_COMPLETE", 409);
    return this.verifyPublication(task, task.report);
  }

  async resultImage(taskId, name) {
    if (typeof name !== "string" || !IMAGE_NAME.test(name)) throw new RestorationError("修复图片名称无效");
    const task = this.getTask(taskId);
    const directory = await this.resolveOutputDirectory(taskId);
    const output = task.outputs.find(item => item.name === name);
    if (!output) throw new RestorationError("修复图片不存在", "RESTORATION_NOT_FOUND", 404);
    const { bytes } = await this.verifiedBytes(path.join(directory, name), output.outputSha256);
    return { path: path.join(directory, name), name, mimeType: "image/png", buffer: bytes, bytes, contentType: "image/png", filename: name };
  }
}
