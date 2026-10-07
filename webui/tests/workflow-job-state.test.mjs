import assert from "node:assert/strict";
import test from "node:test";
import { latestWorkflowJob } from "../src/domain/workflow-job-state.js";
const ids = ["encode.quality", "encode.mp4", "encode.master"];
test("the latest alternative task wins across formats, in both failure and recovery directions", () => {
  const old = { id: "old", commandId: "encode.mp4", state: "succeeded", createdAt: "2026-10-07T00:00:00Z" };
  const current = { id: "new", commandId: "encode.quality", state: "failed", createdAt: "2026-10-07T00:01:00Z" };
  assert.equal(latestWorkflowJob([old, current], ids), current);
  assert.equal(latestWorkflowJob([{ ...old, state: "failed" }, { ...current, state: "succeeded" }], ids).state, "succeeded");
});
test("a new running alternative stays current and unrelated tasks do not override it", () => {
  const active = { id: "active", commandId: "encode.master", state: "running", createdAt: "2026-10-07T00:02:00Z" };
  assert.equal(latestWorkflowJob([{ commandId: "merge.preview_me", createdAt: "2026-10-07T00:03:00Z" }, active], ids), active);
  assert.equal(latestWorkflowJob([{ commandId: "train.me" }], ids), null);
  const first = { id: "first", commandId: "encode.mp4" };
  assert.equal(latestWorkflowJob([first, { commandId: "encode.quality" }], ids), first);
});
