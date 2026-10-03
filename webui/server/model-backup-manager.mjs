import { execFile } from "node:child_process";
import { lstat, realpath } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { buildDflEnvironment } from "./environment.mjs";
import { PATHS } from "./paths.mjs";

const executeFile = promisify(execFile);
const MODEL_NAME = /^(?!CON(?:\.|$)|PRN(?:\.|$)|AUX(?:\.|$)|NUL(?:\.|$)|COM[1-9](?:\.|$)|LPT[1-9](?:\.|$))[^<>:"/\\|?*\u0000-\u001f]{1,64}$/i;
const BACKUP_ID = /^(?:automatic|manual|legacy)\/iter-\d{8,12}-\d{10,20}$/;

export class ModelBackupError extends Error {
  constructor(message, code = "MODEL_BACKUP_ERROR", status = 400) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export class ModelBackupManager {
  constructor({ workspaceRoot = PATHS.workspaceRoot, python = PATHS.python,
    meMain = PATHS.meMain, executor = executeFile } = {}) {
    this.modelRoot = path.resolve(workspaceRoot, "model");
    this.python = python;
    this.meMain = meMain;
    this.executor = executor;
  }

  async modelDirectory(name) {
    if (typeof name !== "string" || !MODEL_NAME.test(name) || name !== name.trim() || /[. ]$/.test(name)) {
      throw new ModelBackupError("ME 模型名称无效", "MODEL_NAME_INVALID");
    }
    const candidate = path.join(this.modelRoot, name);
    let root, directory, info;
    try {
      [root, directory, info] = await Promise.all([
        realpath(this.modelRoot), realpath(candidate), lstat(candidate),
      ]);
    } catch {
      throw new ModelBackupError("ME 模型不存在", "MODEL_NOT_FOUND", 404);
    }
    const relative = path.relative(root, directory);
    if (!info.isDirectory() || info.isSymbolicLink() || !relative || relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new ModelBackupError("ME 模型目录无效", "MODEL_DIRECTORY_INVALID", 400);
    }
    return directory;
  }

  async run(command, model, backupId = null) {
    const args = [this.meMain, command, "--model", model];
    if (backupId != null) args.push("--backup-id", backupId);
    try {
      const result = await this.executor(this.python, args, {
        windowsHide: true,
        timeout: command === "restore-backup" ? 600_000 : 60_000,
        maxBuffer: 4 * 1024 * 1024,
        env: buildDflEnvironment("pytorch"),
      });
      return JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1));
    } catch (error) {
      throw new ModelBackupError(
        error?.stderr?.trim().split(/\r?\n/).at(-1) || error?.message || "ME 备份命令失败",
        "MODEL_BACKUP_COMMAND_FAILED", 422,
      );
    }
  }

  async list(name) {
    return this.run("list-backups", await this.modelDirectory(name));
  }

  async restore(name, backupId) {
    if (typeof backupId !== "string" || !BACKUP_ID.test(backupId)) {
      throw new ModelBackupError("请选择有效的 ME 恢复点", "MODEL_BACKUP_ID_INVALID");
    }
    return this.run("restore-backup", await this.modelDirectory(name), backupId);
  }
}
