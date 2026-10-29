import assert from "node:assert/strict";
import test from "node:test";
import { applyEventToJob, createJobViewStore } from "../src/runtime/useRuntime.js";
import { createLocalizedJobSelector } from "../src/domain/localized-jobs.js";

const job = (id, sequence = 1, values = {}) => ({ id, sequence, createdAt: id, commandId: "train.me", label: "ME 训练", state: "running", ...values });
const metric = (sequence, iteration, jobId = "a") => ({ jobId, sequence, type: "job.metric", payload: { iteration, srcLoss: 0.3, dstLoss: 0.4 } });

test("terminal traffic and sequence-only snapshots preserve visible job references", () => {
  const store = createJobViewStore();
  const initial = store.merge([job("a")]);
  for (let sequence = 2; sequence <= 100; sequence += 1) {
    const output = { jobId: "a", sequence, type: "terminal.output", payload: { data: "line" } };
    assert.equal(applyEventToJob(initial[0], output), initial[0]);
    assert.equal(store.apply(output), initial);
  }
  assert.equal(store.merge([job("a", 100)]), initial);
  assert.equal(store.sequence("a"), 100);
  // A sequence-only snapshot still establishes a newer authoritative watermark.
  assert.equal(store.apply(metric(90, 99)), initial);
  const changed = store.apply(metric(101, 100));
  assert.notEqual(changed, initial);
  assert.equal(changed[0].latestMetric.iteration, 100);
});

test("old HTTP snapshots cannot roll back live metrics or terminal outcomes", () => {
  const store = createJobViewStore();
  store.merge([job("a", 1)]);
  const current = store.apply(metric(20, 200));
  store.apply({ jobId: "a", sequence: 25, type: "terminal.output", payload: { data: "iteration complete" } });
  assert.equal(store.merge([job("a", 19, { latestMetric: { iteration: 190 } })]), current);
  assert.equal(store.merge([job("a", 24, { latestMetric: { iteration: 240 }, controls: [] })]), current);
  assert.equal(store.merge([{ ...current[0], sequence: 30 }]), current);
  assert.equal(store.apply(metric(29, 290)), current);
  const finished = store.apply({ jobId: "a", sequence: 31, type: "job.finished", timestamp: "2026-10-24T12:00:00Z", payload: { state: "cancelled", stopReason: "safe-stop" } });
  assert.equal(finished[0].state, "cancelled");
  assert.equal(store.merge([job("a", 30)]), finished);
});

test("unchanged jobs retain identity when another job changes, with no hidden control updates", () => {
  const store = createJobViewStore();
  const initial = store.merge([job("b"), job("a")]);
  const updated = store.apply(metric(2, 2, "a"));
  assert.equal(updated[0], initial[0]);
  assert.notEqual(updated[1], initial[1]);
  const controls = store.merge([{ ...updated[0], sequence: 3, controls: ["save", "close"] }]);
  assert.notEqual(controls[0], updated[0]);
  assert.deepEqual(controls[0].controls, ["save", "close"]);
  assert.equal(controls[1], updated[1]);
  const health = store.merge([{ ...controls[0], health: { state: "hung", reason: "stale heartbeat" } }]);
  assert.notEqual(health[0], controls[0]);
  assert.equal(health[0].health.state, "hung");
});

test("authoritative deletion keeps jobs created during the request and clears old watermarks", () => {
  const store = createJobViewStore();
  store.merge([job("a", 90)]);
  const knownIds = new Set(["a"]);
  store.merge([job("b", 100)]);
  const retained = store.merge([], knownIds);
  assert.deepEqual(retained.map(item => item.id), ["b"]);
  assert.equal(store.sequence("a"), 0);
  assert.equal(store.merge([job("a", 1)])[1].sequence, 1);
  store.reset();
  assert.equal(store.sequence("b"), 0);
  assert.equal(store.merge([job("b", 1)])[0].sequence, 1);
});

test("prompts, previews, evaluation, progress and state changes publish immediately", () => {
  const store = createJobViewStore();
  store.merge([job("a")]);
  const cases = [
    ["terminal.prompt", { prompt: "continue?" }, "latestPrompt", "continue?"],
    ["job.artifact", { kind: "preview", version: 3 }, "previewVersion", 3],
    ["job.artifact", { kind: "training-evaluation", snapshotId: "eval-3" }, "latestEvaluationSnapshotId", "eval-3"],
    ["job.progress", { percent: 50 }, "latestProgress", { percent: 50 }],
    ["job.state", { state: "stopping", controls: [] }, "state", "stopping"],
  ];
  cases.forEach(([type, payload, field, value], index) => {
    const next = store.apply({ jobId: "a", sequence: index + 2, type, payload });
    assert.deepEqual(next[0][field], value);
  });
});

test("localization reuses unchanged jobs and responds to language, fallback and command updates", () => {
  const select = createLocalizedJobSelector();
  const first = job("a");
  const second = job("b");
  const english = new Map([["train.me", { label: "Train ME" }]]);
  const initial = select([first, second], english);
  assert.equal(select([first, second], new Map(english)), initial);
  const nextFirst = { ...first, controls: ["close"] };
  const changed = select([nextFirst, second], english);
  assert.notEqual(changed[0], initial[0]);
  assert.deepEqual(changed[0].controls, ["close"]);
  assert.equal(changed[1], initial[1]);
  const chinese = select([nextFirst, second], new Map([["train.me", { label: "ME 训练" }]]));
  assert.equal(chinese[0], nextFirst);
  assert.equal(chinese[1], second);
  assert.equal(select([nextFirst, second], new Map()), chinese);
});
