import assert from "node:assert/strict";
import { mkdtemp, readFile, rename, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createMeSupervisorStateWriter } from "../server/me-supervisor-state.mjs";

test("ME supervisor state retries a transient Windows rename failure", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-state-retry-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const stateFile = path.join(directory, "me-runner-state.json");
  let attempts = 0;
  const write = createMeSupervisorStateWriter(stateFile, {
    renameImpl: async (...args) => {
      attempts += 1;
      if (attempts <= 2) throw Object.assign(new Error("file busy"), {
        code: attempts === 1 ? "EPERM" : "EBUSY",
      });
      await rename(...args);
    },
    waitImpl: async () => {},
  });
  await write({ status: "running", updatedAt: "2026-10-02T14:00:00Z" });
  assert.equal(attempts, 3);
  assert.deepEqual(JSON.parse(await readFile(stateFile, "utf8")), {
    status: "running", updatedAt: "2026-10-02T14:00:00Z",
  });
});

test("ME supervisor state survives a several-second Windows file lock", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-state-lock-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const stateFile = path.join(directory, "me-runner-state.json");
  let attempts = 0;
  let waitedMs = 0;
  const write = createMeSupervisorStateWriter(stateFile, {
    renameImpl: async (...args) => {
      attempts += 1;
      if (attempts <= 7) throw Object.assign(new Error("file busy"), { code: "EPERM" });
      await rename(...args);
    },
    waitImpl: async (ms) => { waitedMs += ms; },
  });
  await write({ status: "running" });
  assert.equal(attempts, 8);
  assert.equal(waitedMs, 3420);
  assert.deepEqual(JSON.parse(await readFile(stateFile, "utf8")), { status: "running" });
});

test("ME supervisor state continues queued writes after one failed rename", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "me-state-recover-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const stateFile = path.join(directory, "me-runner-state.json");
  let attempts = 0;
  const write = createMeSupervisorStateWriter(stateFile, {
    renameImpl: async (...args) => {
      attempts += 1;
      if (attempts === 1) throw Object.assign(new Error("file busy"), { code: "EPERM" });
      await rename(...args);
    },
    retryDelaysMs: [],
  });
  const first = write({ iteration: 1 });
  const state = { iteration: 2 };
  const second = write(state);
  state.iteration = 3;
  await assert.rejects(first, { code: "EPERM" });
  await second;
  assert.equal(attempts, 2);
  assert.deepEqual(JSON.parse(await readFile(stateFile, "utf8")), { iteration: 2 });
});
