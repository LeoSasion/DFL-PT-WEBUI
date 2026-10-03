// Owns one fixed ME command independently of the HTTP runtime. The HTTP
// process may restart while this process keeps the trainer and its exit status.
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";
import { createMeSupervisorStateWriter } from "./me-supervisor-state.mjs";

const launchFile = process.argv[2];
const directory = path.dirname(launchFile ?? "");
const stateFile = path.join(directory, "me-runner-state.json");
const outputFile = path.join(directory, "me-output.ndjson");
const controlFile = path.join(directory, "me-runner-control.jsonl");
const trainerHeartbeatFile = path.join(directory, "trainer-heartbeat.json");
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function processAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
}

const atomicState = createMeSupervisorStateWriter(stateFile);
let launchToken = null;

async function run() {
  const launch = JSON.parse(await readFile(launchFile, "utf8"));
  launchToken = typeof launch?.token === "string" ? launch.token : null;
  if (!launch || typeof launch.token !== "string" || typeof launch.executable !== "string"
    || !Array.isArray(launch.args) || typeof launch.cwd !== "string") {
    throw new Error("Invalid ME launch descriptor");
  }
  const startedAt = new Date().toISOString();
  const state = {
    token: launch.token,
    supervisorPid: process.pid,
    launcherPid: null,
    trainerPid: null,
    trainerPidSource: null,
    trainerHeartbeatAt: null,
    trainerPhase: null,
    status: "starting",
    startedAt,
    updatedAt: startedAt,
    endedAt: null,
    exitCode: null,
    signal: null,
    error: null,
  };
  const saveState = () => {
    state.updatedAt = new Date().toISOString();
    return atomicState(state);
  };
  await saveState();

  const output = createWriteStream(outputFile, { flags: "a", encoding: "utf8" });
  let child;
  let outputError = null;
  output.on("error", (error) => {
    outputError = error;
    state.error = `ME output log write failed: ${error.message}`;
    void killLauncherTree().catch(() => {});
  });
  const finishOutput = () => new Promise((resolve) => {
    if (output.destroyed) return resolve();
    output.once("close", resolve);
    output.end();
  });
  const writeOutput = (data) => {
    if (data && !outputError) output.write(`${JSON.stringify({ data })}\n`);
  };
  const syncTrainerPid = async () => {
    let heartbeat;
    try {
      heartbeat = JSON.parse(await readFile(trainerHeartbeatFile, "utf8"));
    } catch (error) {
      if (error.code === "ENOENT" || error instanceof SyntaxError) return false;
      throw error;
    }
    if (!Number.isSafeInteger(heartbeat?.pid) || heartbeat.pid <= 0) return false;
    const heartbeatAt = Date.parse(heartbeat.at ?? "");
    if (!Number.isFinite(heartbeatAt)) return false;
    const heartbeatStartedAt = Date.parse(heartbeat.startedAt ?? "");
    if (!Number.isFinite(heartbeatStartedAt)
      || heartbeatStartedAt < Date.parse(state.startedAt) - 60_000) return false;
    state.trainerHeartbeatAt = heartbeat.at;
    state.trainerPhase = typeof heartbeat.phase === "string" ? heartbeat.phase : null;
    if (state.trainerPid === heartbeat.pid && state.trainerPidSource === "heartbeat") return false;
    state.trainerPid = heartbeat.pid;
    state.trainerPidSource = "heartbeat";
    return true;
  };
  const killLauncherTree = async () => {
    if (!child?.pid) return;
    if (process.platform === "win32") {
      const taskkill = (pid) => new Promise((resolve, reject) => {
        const killer = spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
          shell: false, windowsHide: true, stdio: "ignore",
        });
        killer.once("error", reject);
        killer.once("close", (code) => code === 0 ? resolve() : reject(new Error(`taskkill exited ${code}`)));
      });
      try {
        await taskkill(child.pid);
        return;
      } catch {
        // The launcher can exit while taskkill examines its tree.
      }
      const heartbeatAge = Date.now() - Date.parse(state.trainerHeartbeatAt ?? "");
      if (state.trainerPidSource === "heartbeat" && state.trainerPid !== child.pid
        && Number.isFinite(heartbeatAge) && heartbeatAge >= 0 && heartbeatAge <= 30_000) {
        try {
          await taskkill(state.trainerPid);
          return;
        } catch {
          // The trainer may have exited between heartbeat and termination.
        }
      }
    }
    try { child.kill(); } catch {}
  };
  try {
    child = spawn(launch.executable, launch.args, {
      cwd: launch.cwd,
      env: process.env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    await new Promise((resolve, reject) => {
      child.once("spawn", resolve);
      child.once("error", reject);
    });
  } catch (error) {
    writeOutput(`[WEB] ME Trainer 启动失败：${error.message}\n`);
    await finishOutput();
    state.status = "failed";
    state.error = error.message;
    state.exitCode = 1;
    state.endedAt = new Date().toISOString();
    await saveState();
    return;
  }

  state.launcherPid = child.pid;
  await syncTrainerPid();
  state.status = "running";
  await saveState();
  const streamOutput = (stream) => {
    const decoder = new StringDecoder("utf8");
    stream.on("data", (chunk) => writeOutput(decoder.write(chunk)));
    stream.on("end", () => writeOutput(decoder.end()));
  };
  streamOutput(child.stdout);
  streamOutput(child.stderr);

  let controlOffset = 0;
  let controlPending = "";
  let pollingControl = false;
  const pollControl = async () => {
    if (pollingControl) return;
    pollingControl = true;
    try {
      const contents = await readFile(controlFile, "utf8").catch((error) => {
        if (error.code === "ENOENT") return "";
        throw error;
      });
      if (contents.length < controlOffset) {
        controlOffset = 0;
        controlPending = "";
      }
      const lines = (controlPending + contents.slice(controlOffset)).split("\n");
      controlOffset = contents.length;
      controlPending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let request;
        try { request = JSON.parse(line); } catch { continue; }
        if (request.operation === "force-kill" && request.token === launch.token) {
          writeOutput("[WEB] 已请求强制结束 ME Trainer\n");
          await killLauncherTree();
        }
      }
    } finally {
      pollingControl = false;
    }
  };
  const controlTimer = setInterval(() => void pollControl().catch((error) => {
    writeOutput(`[WEB] 无法读取 ME 控制文件：${error.message}\n`);
  }), 250);
  const identityTimer = setInterval(() => void syncTrainerPid().then((changed) => (
    changed ? saveState() : undefined
  )).catch((error) => {
    writeOutput(`[WEB] 无法读取 ME Trainer 身份：${error.message}\n`);
  }), 1000);
  const heartbeatTimer = setInterval(() => void (async () => {
    await syncTrainerPid();
    await saveState();
  })().catch((error) => {
    writeOutput(`[WEB] 无法保存 ME 监督状态：${error.message}\n`);
  }), 5000);

  const exit = await new Promise((resolve) => {
    child.once("close", (exitCode, signal) => resolve({ exitCode, signal }));
  });
  await syncTrainerPid().catch(() => {});
  if (process.env.DFL_WEB_HEARTBEAT_FILE && !state.trainerPid) {
    // A lightweight launcher can exit before the actual interpreter has
    // published its first heartbeat. Give that child a short startup window.
    for (let attempt = 0; attempt < 20 && !state.trainerPid; attempt += 1) {
      await pause(100);
      await syncTrainerPid().catch(() => {});
    }
  }
  const trainerOutlivedLauncher = state.trainerPidSource === "heartbeat"
    && state.trainerPid !== state.launcherPid && processAlive(state.trainerPid);
  if (trainerOutlivedLauncher) {
    state.launcherExitedAt = new Date().toISOString();
    state.launcherExitCode = exit.exitCode;
    await saveState();
    while (processAlive(state.trainerPid)) {
      await pause(500);
      await syncTrainerPid().catch(() => {});
    }
    await syncTrainerPid().catch(() => {});
  }
  clearInterval(controlTimer);
  clearInterval(identityTimer);
  clearInterval(heartbeatTimer);
  await finishOutput();
  state.status = outputError ? "failed" : "exited";
  state.exitCode = outputError ? 1 : trainerOutlivedLauncher
    ? state.trainerPhase === "finished" ? 0 : 1
    : exit.exitCode;
  state.signal = exit.signal;
  if (trainerOutlivedLauncher && state.exitCode !== 0 && !state.error) {
    state.error = "ME Trainer ended without a completed heartbeat";
  }
  state.endedAt = new Date().toISOString();
  await saveState();
}

run().catch(async (error) => {
  try {
    // Preserve the job identity even when a final state write fails. Without
    // the token, a reconnected Web manager must reject this terminal record.
    if (!launchToken) {
      const launch = JSON.parse(await readFile(launchFile, "utf8"));
      launchToken = typeof launch?.token === "string" ? launch.token : null;
    }
    await atomicState({
      token: launchToken,
      status: "failed",
      supervisorPid: process.pid,
      launcherPid: null,
      trainerPid: null,
      trainerPidSource: null,
      trainerHeartbeatAt: null,
      trainerPhase: null,
      updatedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      exitCode: 1,
      signal: null,
      error: error.message,
    });
  } catch {}
  process.exitCode = 1;
});
