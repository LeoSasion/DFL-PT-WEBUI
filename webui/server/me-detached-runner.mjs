import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { open, readFile, writeFile, appendFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const sidecarScript = fileURLToPath(new URL("./me-runner-sidecar.mjs", import.meta.url));
const POLL_MS = 400;
const START_TIMEOUT_MS = 10_000;
const MAX_LOG_READ_BYTES = 1024 * 1024;
const ORPHAN_STALE_MS = 30_000;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const processAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
};

async function readState(target) {
  try {
    return JSON.parse(await readFile(target, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

export function createMeSupervisorToken() {
  return randomBytes(16).toString("hex");
}

export class MeDetachedRunner {
  constructor({ directory, token, outputOffset = 0, initialState = null, pollMs = POLL_MS,
    orphanStaleMs = ORPHAN_STALE_MS, processAliveImpl = processAlive }) {
    this.directory = directory;
    this.token = token;
    this.outputFile = path.join(directory, "me-output.ndjson");
    this.stateFile = path.join(directory, "me-runner-state.json");
    this.controlFile = path.join(directory, "me-runner-control.jsonl");
    this.outputOffset = Math.max(0, Number(outputOffset) || 0);
    this.pid = initialState?.trainerPid ?? initialState?.launcherPid ?? null;
    this.launcherPid = initialState?.launcherPid ?? null;
    this.supervisorPid = initialState?.supervisorPid ?? null;
    this.state = initialState;
    this.pollMs = pollMs;
    this.orphanStaleMs = orphanStaleMs;
    this.processAliveImpl = processAliveImpl;
    this.dataListener = null;
    this.exitListener = null;
    this.stateListener = null;
    this.pollTimer = null;
    this.pollPending = false;
    this.closed = false;
    this.exited = false;
  }

  onData(listener) {
    this.dataListener = listener;
    this.startPolling();
  }

  onExit(listener) {
    this.exitListener = listener;
    this.startPolling();
  }

  onState(listener) {
    this.stateListener = listener;
  }

  startPolling() {
    if (this.pollTimer || !this.dataListener || !this.exitListener || this.closed) return;
    this.pollTimer = setInterval(() => void this.poll().catch(() => {}), this.pollMs);
    this.pollTimer.unref?.();
    void this.poll().catch(() => {});
  }

  async readOutput() {
    let handle;
    try {
      handle = await open(this.outputFile, "r");
      const size = (await handle.stat()).size;
      if (size < this.outputOffset) this.outputOffset = 0;
      while (this.outputOffset < size) {
        const length = Math.min(size - this.outputOffset, MAX_LOG_READ_BYTES);
        const buffer = Buffer.allocUnsafe(length);
        const { bytesRead } = await handle.read(buffer, 0, length, this.outputOffset);
        if (!bytesRead) break;
        const chunk = buffer.subarray(0, bytesRead);
        const lastNewline = chunk.lastIndexOf(10);
        if (lastNewline < 0) break;
        const startOffset = this.outputOffset;
        let lineStart = 0;
        for (let index = 0; index <= lastNewline; index += 1) {
          if (chunk[index] !== 10) continue;
          const line = chunk.subarray(lineStart, index).toString("utf8");
          const recordOffset = startOffset + index + 1;
          lineStart = index + 1;
          if (!line) continue;
          try {
            const record = JSON.parse(line);
            if (typeof record.data === "string") this.dataListener(record.data, recordOffset);
          } catch {
            // A torn final record remains unread until the next poll.
          }
        }
        this.outputOffset = startOffset + lastNewline + 1;
      }
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    } finally {
      await handle?.close();
    }
  }

  async poll() {
    if (this.pollPending || this.closed || this.exited) return;
    this.pollPending = true;
    try {
      await this.readOutput();
      const state = await readState(this.stateFile);
      if (!state || state.token !== this.token) return;
      const previousStatus = this.state?.status;
      const previousPid = this.pid;
      this.state = state;
      this.pid = state.trainerPid ?? state.launcherPid ?? this.pid;
      this.launcherPid = state.launcherPid ?? this.launcherPid;
      this.supervisorPid = state.supervisorPid ?? this.supervisorPid;
      if (state.status !== previousStatus || this.pid !== previousPid) {
        this.stateListener?.(state);
      }
      if (state.status === "exited" || state.status === "failed") {
        // The sidecar flushes output before publishing terminal state.
        await this.readOutput();
        this.exited = true;
        this.dispose();
        this.exitListener({ exitCode: state.exitCode ?? 1, signal: state.signal ?? null });
      } else if (this.isOrphaned(state)) {
        // The sidecar may die after its final state write fails. Never infer a
        // successful save from a Trainer heartbeat or close acknowledgement.
        this.exited = true;
        this.dispose();
        this.exitListener({ exitCode: 1, signal: null });
      }
    } finally {
      this.pollPending = false;
    }
  }

  isOrphaned(state) {
    const age = Date.now() - Date.parse(state.updatedAt ?? "");
    if (!Number.isFinite(age) || age < this.orphanStaleMs) return false;
    if (!Number.isSafeInteger(state.supervisorPid) || state.supervisorPid <= 0) return false;
    const children = [state.launcherPid, state.trainerPid]
      .filter((pid) => Number.isSafeInteger(pid) && pid > 0);
    if (children.length === 0) return false;
    return [state.supervisorPid, ...children].every((pid) => !this.processAliveImpl(pid));
  }

  async kill() {
    if (this.closed || this.exited) return;
    await appendFile(this.controlFile, `${JSON.stringify({
      operation: "force-kill", token: this.token, requestedAt: new Date().toISOString(),
    })}\n`, "utf8");
  }

  write() {
    throw new Error("ME Web training has no interactive terminal input");
  }

  resize() {
    return false;
  }

  dispose() {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.closed = true;
  }
}

export async function createMeDetachedRunner(options) {
  const { directory, token, outputOffset = 0, attach = false } = options;
  const stateFile = path.join(directory, "me-runner-state.json");
  let initialState = await readState(stateFile);
  if (initialState && initialState.token !== token) {
    throw new Error("ME supervisor identity does not match the persisted job");
  }
  if (attach && !initialState) {
    throw new Error("ME supervisor state is missing");
  }
  if (!attach) {
    if (initialState) throw new Error("ME supervisor already exists for this job");
    const launchFile = path.join(directory, "me-launch.json");
    await writeFile(launchFile, `${JSON.stringify({
      token, executable: options.executable, args: options.args, cwd: options.cwd,
    })}\n`, { encoding: "utf8", flag: "wx" });
    const child = spawn(process.execPath, [sidecarScript, launchFile], {
      cwd: options.cwd,
      env: options.env,
      detached: true,
      shell: false,
      stdio: "ignore",
      windowsHide: true,
    });
    const supervisorPid = await new Promise((resolve, reject) => {
      child.once("spawn", () => resolve(child.pid));
      child.once("error", reject);
    });
    let sidecarExit = null;
    child.once("exit", (exitCode, signal) => { sidecarExit = { exitCode, signal }; });
    child.unref();
    const deadline = Date.now() + START_TIMEOUT_MS;
    while (Date.now() < deadline) {
      initialState = await readState(stateFile);
      if (initialState?.token === token && ["running", "exited", "failed"].includes(initialState.status)) break;
      if (sidecarExit && !initialState) {
        throw new Error(`ME supervisor exited before startup (code ${sidecarExit.exitCode ?? "unknown"})`);
      }
      await delay(100);
    }
    if (!initialState || initialState.token !== token || initialState.status === "starting") {
      throw new Error(`ME supervisor did not start (PID ${supervisorPid ?? "unknown"})`);
    }
    if (initialState.status === "failed") {
      throw new Error(initialState.error || "ME Trainer did not start");
    }
  }
  return new MeDetachedRunner({ directory, token, outputOffset, initialState });
}
