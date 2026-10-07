import { execFile, spawn } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

const execute = promisify(execFile);
async function windowsProcesses() {
  const executable = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const script = "Get-CimInstance Win32_Process | ForEach-Object { [pscustomobject]@{ pid=$_.ProcessId; parent=$_.ParentProcessId; created=$_.CreationDate.ToUniversalTime().Ticks.ToString() } } | ConvertTo-Json -Compress";
  const { stdout } = await execute(executable, ["-NoProfile", "-NonInteractive", "-Command", script],
    { windowsHide: true, timeout: 15000, maxBuffer: 4 * 1024 * 1024 });
  const value = JSON.parse(stdout.replace(/^\uFEFF/, ""));
  return Array.isArray(value) ? value : [value];
}

function descendants(processes, rootPid) {
  const ids = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const item of processes) if (ids.has(item.parent) && !ids.has(item.pid)) {
      ids.add(item.pid); changed = true;
    }
  }
  return processes.filter(item => ids.has(item.pid));
}

// The Windows venv executable is a launcher; its interpreter is a child process.
// Callers retain their reservation until this promise AND the child's close event.
export async function terminateChildProcessTree(child) {
  if (!Number.isSafeInteger(child?.pid) || child.pid <= 0) return Promise.resolve();
  if (process.platform !== "win32") {
    child.kill("SIGTERM");
    return Promise.resolve();
  }
  const original = descendants(await windowsProcesses(), child.pid);
  const result = await new Promise((resolve, reject) => {
    const executable = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "taskkill.exe");
    const killer = spawn(executable, ["/PID", String(child.pid), "/T", "/F"],
      { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
    let detail = "";
    killer.stderr.on("data", chunk => { detail = (detail + chunk).slice(-2000); });
    killer.once("error", reject);
    killer.once("close", code => resolve({code, detail}));
  });
  // taskkill may report a nonzero code after killing the interpreter makes its
  // launcher exit by itself. Verify every observed process identity, never
  // accept that code alone or confuse a reused PID with the original process.
  for (let attempt = 0; attempt < 3; attempt++) {
    const after = await windowsProcesses();
    const remaining = after.filter(item => original.some(owned => owned.pid === item.pid && owned.created === item.created));
    const rootReused = after.some(item => item.pid === child.pid && original.some(owned => owned.pid === child.pid && owned.created !== item.created));
    const lateChildren = rootReused ? [] : descendants(after, child.pid);
    if (!remaining.length && !lateChildren.length) return;
    if (attempt < 2) await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`无法确认任务进程树已停止（taskkill ${result.code}）：${result.detail.trim()}`);
}
