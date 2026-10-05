import assert from "node:assert/strict";
import test from "node:test";
import { getNextWorkflowStep, getUsageReadiness } from "../src/domain/workflow-readiness.js";

const datasets = (srcFrames = 0, dstFrames = 0, srcFaces = 0, dstFaces = 0) => ({ srcFrames: { count: srcFrames }, dstFrames: { count: dstFrames }, srcFaces: { count: srcFaces }, dstFaces: { count: dstFaces } });
const roles = { src: { current: true }, dst: { current: true } };

test("first use waits for authoritative workspace then names the missing material", () => {
  assert.equal(getNextWorkflowStep(null).pending, true);
  assert.equal(getNextWorkflowStep({ materials: {}, datasets: datasets() }).label, "导入 SRC / DST 素材");
  assert.equal(getNextWorkflowStep({ materials: { src: {} }, datasets: datasets() }).label, "导入 DST 素材");
});

test("existing per-side datasets do not require their original videos again", () => {
  const existing = { materials: { dst: {} }, datasets: datasets(0, 0, 5, 0) };
  assert.equal(getNextWorkflowStep(existing).commandId, "dst.extract_frames");
  const next = getNextWorkflowStep({ datasets: datasets(0, 3, 5, 0) });
  assert.equal(next.commandId, "dst.extract_faces");
  assert.equal(getNextWorkflowStep({ datasets: datasets(0, 0, 5, 5) }).target, "roles");
});

test("a missing tool environment stops command recommendations after import", () => {
  const workspace = { materials: { src: {}, dst: {} }, datasets: datasets() };
  const next = getNextWorkflowStep(workspace, 0, false, { toolsReady: false });
  assert.equal(next.target, "settings");
  assert.equal(next.commandId, undefined);
});

test("training setup is deliberate and a known failed GPU check directs to settings", () => {
  const workspace = { datasets: datasets(0, 0, 5, 5), roles };
  const next = getNextWorkflowStep(workspace);
  assert.equal(next.commandId, "train.me");
  assert.equal(next.label, "配置 ME 并检查训练条件");
  const blocked = getNextWorkflowStep(workspace, 0, false, { trainingReady: false });
  assert.equal(blocked.target, "settings");
  assert.equal(blocked.commandId, undefined);
});

test("service, entry points and GPU telemetry remain separate evidence", () => {
  const state = getUsageReadiness({ serviceState: "online", health: { runtime: { pythonAvailable: true, currentAvailable: true, pytorchAvailable: true } }, telemetry: { available: true, gpus: [{ index: 0 }] } });
  assert.equal(state.webReady, true);
  assert.equal(state.toolsReady, true);
  assert.equal(state.toolsVerified, false);
  assert.equal(state.gpuDetected, true);
  assert.equal(state.trainingReady, null);
  const checked = getUsageReadiness({ serviceState: "online", release: { readiness: { toolsReady: true, trainingReady: false } } });
  assert.equal(checked.toolsVerified, true);
  assert.equal(checked.trainingReady, false);
});

test("an active evaluation session and existing exports retain their workflow", () => {
  assert.equal(getNextWorkflowStep({ readiness: { faces: true, me: true } }, 0, true).target, "diagnostics");
  assert.equal(getNextWorkflowStep({ readiness: { faces: true, me: true, merged: true, encoded: true } }, 2).target, "export");
});
