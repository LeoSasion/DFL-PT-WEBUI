import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";
import { runtimeApi } from "../src/runtime/api.js";
import { withRequestDeadline } from "../src/runtime/request-deadline.js";
import { createTerminalEventStore } from "../src/runtime/terminal-event-store.js";
import { activeJobStates } from "../src/domain/job-presentation.js";

const source = (await readFile(new URL("../src/runtime/useRuntime.js", import.meta.url), "utf8"))
  .replace(/^import .*;\r?\n/gm, "")
  .replace(/^export /gm, "");
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const flush = async () => {
  for (let pass = 0; pass < 4; pass += 1) await new Promise(resolve => setImmediate(resolve));
};
const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const job = (id = "a") => ({ id, state: "running", sequence: 0, createdAt: id });
const event = (sequence, jobId = "a") => ({ jobId, sequence, type: "terminal.output", payload: { data: String(sequence) } });

// Execute the production hook with React's relevant state/effect semantics:
// stable refs/setters, dependency comparisons, batched renders and cleanup
// before replacement effects. All API calls, clocks and sockets are local.
function mountRuntime(overrides = {}, { timeoutMs = 30 } = {}) {
  const hooks = [];
  const timers = new Map();
  const sockets = [];
  const api = {
    health: async () => ({ runtime: { current: { workspace: "project-a" } } }),
    commands: async () => [],
    jobs: async () => [job()],
    telemetry: async () => ({ available: true, sample: "current", gpus: [] }),
    events: async () => [],
    ...overrides,
  };
  let index = 0;
  let timerId = 0;
  let mounted = true;
  let scheduled = false;
  let effects = [];
  let value;
  let updatesAfterUnmount = 0;
  const sameDeps = (left, right) => Boolean(left && right && left.length === right.length && left.every((item, i) => Object.is(item, right[i])));
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => { scheduled = false; if (mounted) render(); });
  };
  const memo = (compute, deps) => {
    const slot = index++;
    if (!hooks[slot] || !sameDeps(hooks[slot].deps, deps)) hooks[slot] = { value: compute(), deps };
    return hooks[slot].value;
  };
  const context = vm.createContext({
    createTerminalEventStore,
    activeJobStates,
    AbortController, DOMException,
    useState(initial) {
      const slot = index++;
      if (!hooks[slot]) {
        const record = { value: typeof initial === "function" ? initial() : initial };
        record.set = next => {
          if (!mounted) { updatesAfterUnmount += 1; return; }
          const updated = typeof next === "function" ? next(record.value) : next;
          if (!Object.is(updated, record.value)) { record.value = updated; schedule(); }
        };
        hooks[slot] = record;
      }
      return [hooks[slot].value, hooks[slot].set];
    },
    useRef(initial) { const slot = index++; return hooks[slot] ??= { current: initial }; },
    useMemo: memo,
    useCallback: (callback, deps) => memo(() => callback, deps),
    useEffect(callback, deps) {
      const slot = index++;
      const previous = hooks[slot];
      const record = { deps, callback, cleanup: previous?.cleanup };
      hooks[slot] = record;
      if (!previous || !sameDeps(previous.deps, deps)) effects.push(record);
    },
    runtimeApi: api,
    runtimeWebSocketUrl: (id, after) => `ws://test/${id}?after=${after}`,
    withRequestDeadline: (action, options) => withRequestDeadline(action, { ...options, timeoutMs }),
    window: {
      setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; },
      clearTimeout: id => timers.delete(id),
      setInterval(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay, interval: true }); return id; },
      clearInterval: id => timers.delete(id),
    },
    WebSocket: class {
      static OPEN = 1;
      constructor(url) { this.url = url; this.listeners = {}; this.readyState = 0; sockets.push(this); }
      addEventListener(type, callback) { this.listeners[type] = callback; }
      emit(type, data) { this.listeners[type]?.(type === "message" ? { data: typeof data === "string" ? data : JSON.stringify(data) } : data); }
      close() { this.readyState = 3; this.closed = true; this.emit("close"); }
    },
  });
  vm.runInContext(`${source}\nthis.renderHook = useRuntime;`, context);
  function render() {
    index = 0;
    effects = [];
    value = context.renderHook();
    const changed = effects;
    for (const effect of changed) effect.cleanup?.();
    for (const effect of changed) effect.cleanup = effect.callback();
  }
  render();
  return {
    get value() { return value; },
    get updatesAfterUnmount() { return updatesAfterUnmount; },
    api, sockets, timers,
    runTimer(delay) {
      const entry = [...timers].find(([, timer]) => timer.delay === delay);
      if (!entry) return false;
      const [id, timer] = entry;
      if (!timer.interval) timers.delete(id);
      timer.callback();
      return true;
    },
    replayEffects() {
      const records = hooks.filter(record => record.callback);
      for (const record of records) record.cleanup?.();
      for (const record of records) record.cleanup = record.callback();
    },
    unmount() {
      mounted = false;
      for (const record of hooks) record.cleanup?.();
    },
  };
}

test("runtime read APIs forward cancellation through fetch", async t => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const controller = new AbortController();
  const paths = [];
  globalThis.fetch = async (path, options) => {
    assert.equal(options.signal, controller.signal);
    paths.push(path);
    return { ok: true, json: async () => ({ data: [] }) };
  };
  await runtimeApi.health({ signal: controller.signal });
  await runtimeApi.commands({ signal: controller.signal });
  await runtimeApi.jobs({ signal: controller.signal });
  await runtimeApi.events("a/b", 42, { signal: controller.signal });
  await runtimeApi.telemetry({ signal: controller.signal });
  assert.deepEqual(paths, ["/api/health", "/api/commands", "/api/jobs", "/api/jobs/a%2Fb/events?after=42", "/api/telemetry"]);
});

test("a successful header with a hanging body times out and refresh recovers without late overwrites", async t => {
  const body = deferred();
  const originalFetch = globalThis.fetch;
  let requestSignal;
  globalThis.fetch = async (path, options) => {
    requestSignal = options.signal;
    return { ok: true, json: () => body.promise };
  };
  const harness = mountRuntime({ health: runtimeApi.health });
  t.after(() => { harness.unmount(); globalThis.fetch = originalFetch; });
  await wait(45);
  await flush();
  assert.equal(harness.value.serviceState, "offline");
  assert.equal(harness.value.lastError.code, "RUNTIME_REQUEST_TIMEOUT");
  assert.equal(requestSignal.aborted, true);
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ data: { runtime: { current: { workspace: "recovered" } } } }) });
  await harness.value.refresh({ quiet: true });
  await flush();
  assert.equal(harness.value.serviceState, "online");
  assert.equal(harness.value.health.runtime.current.workspace, "recovered");
  body.resolve({ data: { runtime: { current: { workspace: "late-old" } } } });
  await flush();
  assert.equal(harness.value.health.runtime.current.workspace, "recovered");
});

test("refresh is single-flight, releases an abort-ignoring read, and does not launch late dependent reads", async t => {
  const pending = deferred();
  let healthSignal;
  let jobReads = 0;
  const harness = mountRuntime({
    health: ({ signal }) => { healthSignal = signal; return pending.promise; },
    jobs: async () => { jobReads += 1; return [job()]; },
  });
  t.after(() => harness.unmount());
  const first = harness.value.refresh({ quiet: true });
  assert.equal(harness.value.refresh({ quiet: true }), first);
  await first;
  assert.equal(healthSignal.aborted, true);
  harness.api.health = async () => ({ project: { id: "new" } });
  await harness.value.refresh({ quiet: true });
  await flush();
  assert.equal(harness.value.serviceState, "online");
  assert.equal(jobReads, 1);
  pending.resolve({ project: { id: "old" } });
  await flush();
  assert.equal(jobReads, 1);
  assert.equal(harness.value.health.project.id, "new");
});

test("cleanup and effect replay abort the old lifecycle without hiding new results", async t => {
  const pending = deferred();
  const requests = [];
  const harness = mountRuntime({ health: ({ signal }) => { requests.push(signal); return pending.promise; } });
  t.after(() => harness.unmount());
  await flush();
  harness.api.health = async ({ signal }) => { requests.push(signal); return { project: { id: "remounted" } }; };
  harness.replayEffects();
  await flush();
  assert.equal(requests[0].aborted, true);
  assert.equal(harness.value.health.project.id, "remounted");
  pending.resolve({ project: { id: "old" } });
  await flush();
  assert.equal(harness.value.health.project.id, "remounted");
  harness.unmount();
  await flush();
  assert.equal(harness.timers.size, 0);
  assert.equal(harness.updatesAfterUnmount, 0);
});

test("live sequence 42 advances reconnect cursors, deduplicates replay and survives malformed messages", async t => {
  const afters = [];
  const harness = mountRuntime({ events: async (id, after) => { afters.push(after); return after ? [event(42), event(43)] : [event(10)]; } });
  t.after(() => harness.unmount());
  await flush();
  const firstSocket = harness.sockets[0];
  firstSocket.emit("message", event(42));
  firstSocket.emit("message", event(42));
  firstSocket.emit("message", "invalid json");
  await flush();
  assert.match(harness.value.lastError.message, /无效消息/);
  firstSocket.emit("error");
  assert.equal(harness.runTimer(800), true);
  await flush();
  assert.deepEqual(afters, [0, 42]);
  assert.match(harness.sockets[1].url, /after=43$/);
  assert.deepEqual(Array.from(harness.value.selectedEvents, item => item.sequence), [10, 42, 43]);
  firstSocket.emit("message", event(99));
  await flush();
  assert.deepEqual(Array.from(harness.value.selectedEvents, item => item.sequence), [10, 42, 43]);
});

test("log bursts update terminal events and cursor without replacing visible jobs", async t => {
  const harness = mountRuntime();
  t.after(() => harness.unmount());
  await flush();
  const initialJobs = harness.value.jobs;
  const initialJob = harness.value.selectedJob;
  const socket = harness.sockets[0];
  for (let sequence = 1; sequence <= 50; sequence += 1) {
    socket.emit("message", event(sequence));
    await flush();
    assert.equal(harness.value.jobs, initialJobs);
    assert.equal(harness.value.selectedJob, initialJob);
  }
  assert.equal(harness.value.selectedEvents.length, 50);
  assert.equal(harness.value.selectedJobSequence, 50);
  socket.emit("message", { jobId: "a", sequence: 51, type: "job.metric", payload: { iteration: 10, srcLoss: 0.2, dstLoss: 0.3 } });
  await flush();
  assert.equal(harness.value.selectedJob.latestMetric.iteration, 10);
  const metricJobs = harness.value.jobs;
  harness.api.jobs = async () => [{ ...job(), sequence: 49, latestMetric: { iteration: 9 } }];
  await harness.value.refresh({ quiet: true });
  await flush();
  assert.equal(harness.value.jobs, metricJobs);
  assert.equal(harness.value.selectedJob.latestMetric.iteration, 10);
});

test("switching tasks retains each cursor and rejects the previous task's late backlog", async t => {
  const pending = deferred();
  const calls = [];
  let first = true;
  let oldSignal;
  const harness = mountRuntime({
    jobs: async () => [job("b"), job("a")],
    events: (id, after, { signal }) => {
      calls.push([id, after]);
      if (first) { first = false; oldSignal = signal; return pending.promise; }
      return Promise.resolve([event(id === "a" ? 12 : 7, id)]);
    },
  });
  t.after(() => harness.unmount());
  await flush();
  assert.equal(calls[0][0], "b");
  harness.value.selectJob("a");
  await flush();
  assert.equal(oldSignal.aborted, true);
  pending.resolve([event(99, "b")]);
  await flush();
  assert.equal(harness.sockets.length, 1);
  harness.sockets[0].emit("message", event(42));
  await flush();
  harness.value.selectJob("b");
  await flush();
  harness.value.selectJob("a");
  await flush();
  assert.deepEqual(calls, [["b", 0], ["a", 0], ["b", 0], ["a", 42]]);
  assert.deepEqual(Array.from(harness.value.selectedEvents, item => item.sequence), [12, 42]);
});

test("project changes reset reused job IDs and ignore old project socket and telemetry responses", async t => {
  let project = "project-a";
  const slowTelemetry = deferred();
  const signals = [];
  const calls = [];
  const harness = mountRuntime({
    health: async () => ({ project: { id: project } }),
    telemetry: ({ signal }) => {
      signals.push(signal);
      return project === "project-a" ? slowTelemetry.promise : Promise.resolve({ available: true, project });
    },
    events: async (id, after) => { calls.push([project, after]); return [event(project === "project-a" ? 10 : 3)]; },
  });
  t.after(() => harness.unmount());
  await flush();
  const oldSocket = harness.sockets[0];
  oldSocket.emit("message", event(42));
  await flush();
  project = "project-b";
  await harness.value.refresh({ quiet: true });
  await flush();
  assert.equal(signals[0].aborted, true);
  assert.equal(oldSocket.closed, true);
  oldSocket.emit("message", event(99));
  slowTelemetry.resolve({ available: true, project: "project-a" });
  await flush();
  assert.deepEqual(calls, [["project-a", 0], ["project-b", 0]]);
  assert.deepEqual(Array.from(harness.value.selectedEvents, item => item.sequence), [3]);
  assert.equal(harness.value.telemetry.project, "project-b");
});

test("telemetry does not overlap slow reads and recovers after an abort-ignoring timeout", async t => {
  const pending = deferred();
  const signals = [];
  const harness = mountRuntime({ telemetry: ({ signal }) => { signals.push(signal); return pending.promise; } });
  t.after(() => harness.unmount());
  await flush();
  assert.equal(harness.runTimer(3000), false);
  assert.equal(signals.length, 1);
  await wait(45);
  await flush();
  assert.equal(signals[0].aborted, true);
  assert.equal(harness.value.telemetry.available, false);
  harness.api.telemetry = async () => ({ available: true, sample: "recovered" });
  assert.equal(harness.runTimer(3000), true);
  await flush();
  pending.resolve({ available: true, sample: "late-old" });
  await flush();
  assert.equal(harness.value.telemetry.sample, "recovered");
});

test("a hanging events read times out into live observation and cannot later advance the cursor", async t => {
  const pending = deferred();
  let readSignal;
  const afters = [];
  const harness = mountRuntime({ events: (id, after, { signal }) => { afters.push(after); readSignal = signal; return pending.promise; } });
  t.after(() => harness.unmount());
  await wait(45);
  await flush();
  assert.equal(readSignal.aborted, true);
  assert.equal(harness.sockets.length, 1);
  harness.sockets[0].emit("message", event(42));
  pending.resolve([event(999)]);
  await flush();
  harness.api.events = async (id, after) => { afters.push(after); return []; };
  harness.sockets[0].close();
  harness.runTimer(800);
  await flush();
  assert.deepEqual(afters, [0, 42]);
  assert.deepEqual(Array.from(harness.value.selectedEvents, item => item.sequence), [42]);
});

test("an archived task refreshes the authoritative list without reconnecting or repeating writes", async t => {
  let jobReads = 0;
  const harness = mountRuntime({
    jobs: async () => ++jobReads === 1 ? [job()] : [],
    events: async () => { throw Object.assign(new Error("archived"), { status: 410 }); },
  });
  t.after(() => harness.unmount());
  await flush();
  assert.equal(jobReads, 2);
  assert.equal(harness.value.selectedJobId, null);
  assert.equal(harness.sockets.length, 0);
  assert.equal(harness.runTimer(800), false);
});

test("unmount aborts pending telemetry and backlog without later state writes or reconnects", async () => {
  const pending = deferred();
  const signals = [];
  const read = ({ signal }) => { signals.push(signal); return pending.promise; };
  const harness = mountRuntime({ telemetry: read, events: (id, after, options) => read(options) });
  await flush();
  harness.unmount();
  assert.equal(signals.length, 2);
  assert.equal(signals.every(signal => signal.aborted), true);
  pending.resolve([]);
  await flush();
  assert.equal(harness.updatesAfterUnmount, 0);
  assert.equal(harness.sockets.length, 0);
  assert.equal(harness.timers.size, 0);
});
