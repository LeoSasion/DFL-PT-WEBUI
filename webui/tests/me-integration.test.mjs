import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { buildCommand, getCommandDefinition, listCommands, validateCommandParameters } from "../server/command-registry.mjs";
import { JobManager } from "../server/job-manager.mjs";
import { PATHS } from "../server/paths.mjs";
import { discoverModels } from "../server/workspace-manager.mjs";

test("face model commands are exclusively ME and use the repository PyTorch runtime", () => {
  const commands = listCommands();
  assert.deepEqual(commands.filter(command => command.category === "training").map(command => command.id), ["train.me", "xseg.train"]);
  for (const id of ["train.saehd", "train.q384", "train.q512", "merge.saehd", "merge.amp", "export.dfm_saehd"]) {
    assert.equal(getCommandDefinition(id), null);
    assert.throws(() => validateCommandParameters(id), { code: "COMMAND_NOT_ALLOWED" });
  }
  const parameters = validateCommandParameters("train.me", { forceModelName: "test-model", resolution: 128, batchSize: 4, cpuOnly: true }, "guided");
  const { launch } = buildCommand(getCommandDefinition("train.me"), {
    parameters, controlFile: "control.jsonl", previewFile: "preview.png", controlAckFile: "control-ack.json",
  });
  assert.equal(launch.executable, path.join(PATHS.repositoryRoot, ".venv", "Scripts", "python.exe"));
  assert.equal(launch.args[0], PATHS.meMain);
  assert.equal(launch.args[1], "web-train");
  const value = flag => launch.args[launch.args.indexOf(flag) + 1];
  assert.equal(value("--model"), path.join(PATHS.workspaceRoot, "model", "test-model"));
  assert.equal(value("--device"), "cpu");
  assert.equal(value("--steps"), "0");
  assert.equal(launch.env.DFL_WEB_CONTROL_ACK_FILE, "control-ack.json");
  assert.ok(!launch.args.includes("--resume"));
  const resumed = buildCommand(getCommandDefinition("train.me"), { parameters: { ...parameters, silentStart: true } }).launch;
  assert.ok(resumed.args.includes("--resume"));
});

test("ME discovery requires a complete PyTorch checkpoint and matching metadata identity", async t => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-discovery-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const modelRoot = path.join(directory, "model");
  await mkdir(modelRoot);
  await writeFile(path.join(modelRoot, "old_SAEHD_data.dat"), "legacy artifact");
  await writeFile(path.join(modelRoot, "old_ME_encoder.npy"), "legacy artifact");
  for (const [name, checkpoint, format, metadataName] of [
    ["good", "saved checkpoint", "me-pytorch", "good"],
    ["empty", "", "me-pytorch", "empty"],
    ["wrong-format", "saved checkpoint", "tensorflow", "wrong-format"],
    ["wrong-name", "saved checkpoint", "me-pytorch", "other"],
  ]) {
    const model = path.join(modelRoot, name);
    await mkdir(model);
    await writeFile(path.join(model, "me.pt"), checkpoint);
    await writeFile(path.join(model, "metadata.json"), JSON.stringify({ format, version: 1, name: metadataName,
      model_class: "ME", iteration: 25, checkpoint: "me.pt", config: { resolution: 128, batch_size: 4 } }));
  }
  const discovered = await discoverModels(directory);
  assert.deepEqual(discovered.models.map(model => model.name), ["good"]);
  assert.equal(discovered.models[0].iteration, 25);
  assert.equal(discovered.models[0].config.resolution, 128);
  assert.equal(discovered.meStats.count, 1);
});

test("ME close acknowledgement must match the request before a stop is marked saved", async t => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-control-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  for (const scenario of ["matching", "stale", "failed-exit", "missing"]) {
    const jobDirectory = path.join(directory, scenario);
    await mkdir(jobDirectory);
    const manager = new JobManager({ metadataWriter: async () => {} });
    const job = {
      id: scenario, commandId: "train.me", category: "training", state: "running", controls: ["close"],
      parameters: { forceModelName: "ack-model" }, runner: { dispose() {}, kill() {} }, locks: [],
      directory: jobDirectory, eventsFile: path.join(jobDirectory, "events.ndjson"),
      controlFile: path.join(jobDirectory, "control.jsonl"), controlAckFile: path.join(jobDirectory, "control-ack.json"),
      metadataFile: path.join(jobDirectory, "metadata.json"), events: [], sequence: 0,
      writeChain: Promise.resolve(), metadataWriteChain: Promise.resolve(),
    };
    manager.jobs.set(job.id, job);
    await manager.control(job.id, "close");
    const request = JSON.parse((await readFile(job.controlFile, "utf8")).trim());
    if (scenario !== "missing") await writeFile(job.controlAckFile, JSON.stringify({ operation: "close", status: "completed",
      requestedAt: scenario === "stale" ? "earlier-request" : request.requestedAt,
      iteration: 2, checkpoint: path.join(PATHS.workspaceRoot, "model", "ack-model", "me.pt") }));
    await manager.handleExit(job, scenario === "failed-exit" ? 1 : 0, null);
    assert.equal(job.stopReason, scenario === "matching" ? "safe-stop" : "safe-stop-unconfirmed");
  }
});

test("ME safe-stop timeout kills the runner and cannot confirm a saved stop", {
  timeout: 3000,
}, async t => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-stop-timeout-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const requestedAt = new Date(Date.now() - 121_000);
  const manager = new JobManager({ now: () => requestedAt, metadataWriter: async () => {} });
  let killCalls = 0;
  let disposeCalls = 0;
  let reportKill;
  const killed = new Promise(resolve => { reportKill = resolve; });
  const job = {
    id: "me-stop-timeout", commandId: "train.me", category: "training", state: "running",
    controls: ["close"], parameters: { forceModelName: "timeout-model" },
    runner: { kill: async () => { killCalls += 1; reportKill(); },
      dispose: () => { disposeCalls += 1; } },
    locks: ["workspace:model", "gpu"], directory,
    eventsFile: path.join(directory, "events.ndjson"),
    controlFile: path.join(directory, "control.jsonl"),
    controlAckFile: path.join(directory, "control-ack.json"),
    metadataFile: path.join(directory, "metadata.json"),
    events: [], sequence: 0, writeChain: Promise.resolve(),
    metadataWriteChain: Promise.resolve(), error: null,
  };
  manager.jobs.set(job.id, job);
  manager.acquireLocks(job.id, job.locks);
  await manager.control(job.id, "close");
  await killed;
  assert.equal(killCalls, 1);
  assert.equal(job.state, "stopping");
  assert.equal(job.stopReason, "safe-stop-timeout");
  const request = JSON.parse((await readFile(job.controlFile, "utf8")).trim());
  assert.equal(request.operation, "close");
  await writeFile(job.controlAckFile, JSON.stringify({
    operation: "close", status: "completed", requestedAt: request.requestedAt,
    iteration: 2, checkpoint: path.join(directory, "me.pt"),
  }));
  await manager.handleExit(job, 0, null);
  assert.equal(job.state, "cancelled");
  assert.equal(job.stopReason, "safe-stop-timeout", "even a matching ACK and exit 0 cannot confirm a timed-out stop");
  assert.equal(disposeCalls, 1);
  assert.equal(manager.locks.has("gpu"), false);
  assert.equal(manager.locks.has("workspace:model"), false);
});
