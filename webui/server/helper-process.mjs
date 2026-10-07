import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { terminateChildProcessTree } from "./process-tree.mjs";

const helpers = new Map();
export class HelperProcessError extends Error {
  constructor(message, code, status = 409) { super(message); this.name = "HelperProcessError"; this.code = code; this.status = status; }
}
export function activeHelperProcesses() {
  return [...helpers.values()].map(({ child: _child, cancel: _cancel, done: _done, ...item }) => ({ ...item }));
}
export function assertHelperStopsConfirmed() {
  const unknown = activeHelperProcesses().filter(item => item.state === "stop-unconfirmed");
  if (unknown.length) throw new HelperProcessError("后台工具进程退出尚未确认，工作区保持占用", "HELPER_STOP_UNCONFIRMED");
}
export async function closeHelperProcesses() {
  const running = [...helpers.values()];
  running.forEach(item => item.cancel?.());
  await Promise.allSettled(running.map(item => item.done));
  assertHelperStopsConfirmed();
}

// Fixed-command callers own executable/arguments. A helper is settled only after
// stream close and, when stopping, verified process-tree termination.
export function runHelperProcess(executable, args, {
  cwd, env, input, signal, label = "后台工具", timeoutMs = 120_000,
  maxBytes = 4 * 1024 * 1024, closeWaitMs = 10_000,
  onStdout, onStderr, spawnProcess = spawn, terminateProcessTree = terminateChildProcessTree,
} = {}) {
  if (signal?.aborted) return Promise.reject(Object.assign(new Error("操作已取消"), { name: "AbortError", code: "OPERATION_CANCELLED", status: 499 }));
  const id = `helper-${randomBytes(8).toString("hex")}`;
  let resolveDone;
  const entry = { id, label, state: "running", pid: null, startedAt: new Date().toISOString(), child: null,
    done: new Promise(resolve => { resolveDone = resolve; }) };
  helpers.set(id, entry);
  return new Promise((resolve, reject) => {
    let child, timer, closeTimer, bytes = 0, settled = false, closed = false, closedCode, failure, stopping;
    const stdout = [], stderr = [];
    let resolveClose;
    const closedPromise = new Promise(done => { resolveClose = done; });
    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer); clearTimeout(closeTimer);
      signal?.removeEventListener("abort", abort);
      if (error?.code === "HELPER_STOP_UNCONFIRMED") { entry.state = "stop-unconfirmed"; entry.error = error.message; }
      else helpers.delete(id);
      resolveDone();
      if (error) reject(error); else resolve(value);
    };
    const stop = error => {
      if (stopping || settled) return;
      failure = error;
      entry.state = "stopping";
      stopping = Promise.resolve().then(() => terminateProcessTree(child)).then(async () => {
        if (!closed) await Promise.race([closedPromise, new Promise((_, fail) => {
          closeTimer = setTimeout(() => fail(new Error("进程树已终止，但流关闭未确认")), closeWaitMs);
        })]);
        finish(failure);
      }).catch(error => finish(new HelperProcessError(`${label}进程退出无法确认：${error.message}`, "HELPER_STOP_UNCONFIRMED")));
    };
    const abort = () => stop(Object.assign(new Error("操作已取消，进程树已停止"), { name: "AbortError", code: "OPERATION_CANCELLED", status: 499 }));
    entry.cancel = abort;
    try { child = spawnProcess(executable, args, { cwd, env, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] }); }
    catch (error) { finish(error); return; }
    entry.child = child; entry.pid = child.pid ?? null;
    timer = setTimeout(() => stop(new HelperProcessError(`${label}超过处理时限`, "HELPER_TIMEOUT", 504)), timeoutMs);
    signal?.addEventListener("abort", abort, { once: true });
    const collect = (target, callback) => chunk => {
      if (settled || failure) return;
      bytes += chunk.length;
      if (bytes > maxBytes) { stop(new HelperProcessError(`${label}响应超过允许大小`, "HELPER_OUTPUT_TOO_LARGE", 422)); return; }
      target.push(chunk);
      try { callback?.(chunk); } catch (error) { stop(error); }
    };
    child.stdout.on("data", collect(stdout, onStdout));
    child.stderr.on("data", collect(stderr, onStderr));
    child.stdin?.on("error", error => { if (!closed && !failure) stop(error); });
    child.once("error", error => { failure ??= error; });
    child.once("close", code => {
      closed = true; closedCode = code; resolveClose();
      if (stopping) return;
      if (failure) { finish(failure); return; }
      finish(null, { code: closedCode, stdout: Buffer.concat(stdout).toString("utf8"), stderr: Buffer.concat(stderr).toString("utf8") });
    });
    if (signal?.aborted) abort();
    if (input === undefined) child.stdin?.end();
    else child.stdin?.end(typeof input === "string" || Buffer.isBuffer(input) ? input : JSON.stringify(input));
  });
}
