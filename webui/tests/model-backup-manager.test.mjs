import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, realpath, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { ModelBackupManager } from "../server/model-backup-manager.mjs";
import { RuntimeServer } from "../server/app-server.mjs";

test("ME backup manager accepts only a project model and a fixed restore command", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "me-backup-manager-"));
  try {
    const model = path.join(root, "model", "人物 A");
    await mkdir(model, { recursive: true });
    const resolvedModel = await realpath(model);
    const calls = [];
    const manager = new ModelBackupManager({ workspaceRoot: root, python: "python.exe", meMain: "me.py",
      executor: async (executable, args, options) => {
        calls.push({ executable, args, options });
        return { stdout: `${JSON.stringify(args[1] === "list-backups" ? { backups: [] }
          : { restoredIteration: 7, rollbackId: "manual/iter-00000008-1234567890123456789" })}\n` };
      } });
    assert.deepEqual(await manager.list("人物 A"), { backups: [] });
    const id = "automatic/iter-00000007-1234567890123456789";
    assert.equal((await manager.restore("人物 A", id)).restoredIteration, 7);
    assert.deepEqual(calls.map(call => call.args), [
      ["me.py", "list-backups", "--model", resolvedModel],
      ["me.py", "restore-backup", "--model", resolvedModel, "--backup-id", id],
    ]);
    assert.ok(calls.every(call => call.executable === "python.exe" && call.options.windowsHide));
    await assert.rejects(manager.list("../outside"), error => error.code === "MODEL_NAME_INVALID");
    await assert.rejects(manager.restore("人物 A", "../../outside"), error => error.code === "MODEL_BACKUP_ID_INVALID");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("model backup HTTP restore requires a session and no active workspace job", async t => {
  class Jobs extends EventEmitter {
    constructor() { super(); this.current = [{ id:"active-training", state:"running" }]; }
    async initialize() {}
    list() { return this.current; }
  }
  const jobs = new Jobs();
  const calls = [];
  const server = new RuntimeServer({ jobManager: jobs,
    trainingEvaluationManager: { initialize: async () => {} },
    modelBackupManager: {
      list: async name => { calls.push(["list", name]); return { backups:[{ id:"automatic/iter-00000007-1234567890123456789" }] }; },
      restore: async (name, id) => { calls.push(["restore", name, id]); return { restoredIteration:7 }; },
    },
  });
  const address = await server.start({ port: 0 });
  t.after(() => server.stop());
  const base = `http://127.0.0.1:${address.port}`;
  const origin = "http://127.0.0.1:4173";
  const health = await fetch(`${base}/api/health`, { headers:{ Origin:origin } });
  const cookie = health.headers.get("set-cookie").split(";")[0];
  const listing = await fetch(`${base}/api/models/%E4%BA%BA%E7%89%A9%20A/backups`, { headers:{ Origin:origin } });
  assert.equal(listing.status, 200);
  assert.deepEqual(calls[0], ["list", "人物 A"]);
  const route = `${base}/api/models/%E4%BA%BA%E7%89%A9%20A/backups/restore`;
  const payload = { method:"POST", headers:{ Origin:origin, "Content-Type":"application/json" },
    body:JSON.stringify({ backupId:"automatic/iter-00000007-1234567890123456789" }) };
  assert.equal((await fetch(route, payload)).status, 403);
  assert.equal((await fetch(route, { ...payload, headers:{ ...payload.headers, Cookie:cookie } })).status, 409);
  assert.equal(calls.length, 1);
  jobs.current = [];
  const restored = await fetch(route, { ...payload, headers:{ ...payload.headers, Cookie:cookie } });
  assert.equal(restored.status, 200);
  assert.equal((await restored.json()).data.restoredIteration, 7);
  assert.deepEqual(calls.at(-1), ["restore", "人物 A", "automatic/iter-00000007-1234567890123456789"]);
});
