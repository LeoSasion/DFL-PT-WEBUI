import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { matchesRuntimeIdentity, ownsManagerRecord } from "../../../webui/scripts/local-manager.mjs";

const root = path.resolve("expected-project", "_internal", "DeepFaceLab");
const payload = (service, dflRoot) => ({ data: { service, runtime: { current: { dflRoot } } } });

test("manager accepts only the new product running from the selected repository", () => {
  assert.equal(matchesRuntimeIdentity(payload("DFL-PT-WEBUI Local Runtime", root), root), true);
  assert.equal(matchesRuntimeIdentity(payload("DeepFaceLabSN Local Runtime", root), root), false);
  assert.equal(matchesRuntimeIdentity(payload("DFL-PT-WEBUI Local Runtime", path.resolve("other-project", "_internal", "DeepFaceLab")), root), false);
  assert.equal(matchesRuntimeIdentity({ ok: true }, root), false);
});

test("a reused PID or another repository's status never grants manager ownership", () => {
  const logs = path.resolve("expected-project", "webui", ".runtime", "logs");
  const now = Date.now();
  const status = { supervisorPid: 42, updatedAt: new Date(now).toISOString(), logFile: path.join(logs, "manager.log") };
  assert.equal(ownsManagerRecord(42, status, logs, now), true);
  assert.equal(ownsManagerRecord(43, status, logs, now), false);
  assert.equal(ownsManagerRecord(42, status, logs, now + 15000), false);
  assert.equal(ownsManagerRecord(42, { ...status, logFile: path.resolve("another-project", "manager.log") }, logs, now), false);
});
