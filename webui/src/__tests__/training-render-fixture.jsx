import React, { Profiler } from "react";
import { createRoot } from "react-dom/client";
import { LanguageProvider } from "../i18n.jsx";
import { ProgressFeedbackProvider } from "../components/ProgressFeedback.jsx";
import { runtimeApi } from "../runtime/api.js";
import { applyEventToJob } from "../runtime/useRuntime.js";
import "../styles.css";
import "../components/project-workflow.css";

if (!import.meta.env.DEV) throw new Error("The render fixture requires the isolated development test server");

const projectRoot = "render-fixture-workspace";
let jobs = [{
  id: "training-a", commandId: "train.me", label: "ME 训练", state: "running", sequence: 1,
  createdAt: "2026-10-24T00:00:00Z", startedAt: "2026-10-24T00:00:01Z",
  controls: ["save", "backup", "preview", "close"], parameters: { forceModelName: "fixture-me" },
  latestMetric: { iteration: 100, srcLoss: 0.3, dstLoss: 0.4, targetIterations: 1000 },
  previewVersion: null, health: { state: "healthy" },
}];
if (new URLSearchParams(window.location.search).has("emptyHistory")) jobs[0].latestMetric = null;
const events = new Map();
const sockets = [];
const controls = [];
const nativeQueries = [];
window.__componentCommits = {};
window.__rootCommits = 0;

class FixtureSocket {
  static OPEN = 1;
  constructor(url) {
    this.url = url;
    this.jobId = new URL(url).searchParams.get("jobId");
    this.listeners = {};
    this.readyState = 0;
    sockets.push(this);
    queueMicrotask(() => { if (this.readyState !== 3) { this.readyState = 1; this.listeners.open?.(); } });
  }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  close() { this.readyState = 3; this.listeners.close?.(); }
  send() {}
}
window.WebSocket = FixtureSocket;

const emit = event => {
  jobs = jobs.map(job => job.id === event.jobId
    ? { ...applyEventToJob(job, event), sequence: event.sequence } : job);
  events.set(event.jobId, [...(events.get(event.jobId) ?? []), event]);
  for (const socket of sockets) {
    if (socket.readyState === 1 && socket.jobId === event.jobId) socket.listeners.message?.({ data: JSON.stringify(event) });
  }
};
Object.assign(runtimeApi, {
  health: async () => ({ project: { id: "render-fixture", name: "Render fixture" }, runtime: { current: { workspace: projectRoot }, pythonAvailable: true, currentAvailable: true, pytorchAvailable: true } }),
  commands: async () => [{ id: "train.me", label: "ME 训练", category: "training", controls: ["save", "preview", "close"] }],
  jobs: async () => structuredClone(jobs),
  events: async (id, after) => structuredClone((events.get(id) ?? []).filter(event => event.sequence > after)),
  telemetry: async () => ({ available: true, gpus: [{ index: 0, name: "Fixture GPU", memoryUsedMiB: 1000, memoryTotalMiB: 24000, utilizationPercent: 10, temperatureC: 40 }] }),
  workspace: async () => ({ root: projectRoot, models: [{ name: "fixture-me", type: "ME", ready: true }], datasets: {}, materials: {}, readiness: { me: true }, roles: {} }),
  release: async () => ({ appVersion: "render-fixture", readiness: { toolsReady: true, trainingReady: true } }),
  trainingEvaluationSnapshots: async () => ({ snapshots: [] }),
  watchOperations: () => Object.assign(() => {}, { refresh: async () => {} }),
  control: async (id, operation) => { controls.push({ id, operation }); return structuredClone(jobs.find(job => job.id === id)); },
  resize: async () => ({}),
  alignedAssets: async side => ({ side, total: 0, offset: 0, limit: 60, items: [] }),
  alignedQuarantine: async side => ({ side, total: 0, offset: 0, limit: 60, items: [] }),
  alignedAudit: async side => ({ side, total: 0, offset: 0, limit: 60, items: [], issueCounts: {}, summary: {} }),
  alignedSimilarity: async (side, scope) => {
    const { threshold, offset, compareOffset, limit } = scope;
    const paired = compareOffset !== null; const windowSize = paired ? 250 : 500;
    const windows = (paired ? [offset, compareOffset] : [offset]).map((start, batch) => {
      const count = Math.min(windowSize, Math.max(0, 1000 - start));
      return { batch, offset: start, count, analyzedCount: count, invalidCount: 0, start: count ? start + 1 : null, end: count ? start + count : null };
    });
    nativeQueries.push({ side, threshold, offset, compareOffset, limit, windowSize });
    const selectedCount = windows.reduce((sum, value) => sum + value.count, 0);
    return { side, workspaceKey: projectRoot, fingerprint: "a".repeat(64), threshold, offset, compareOffset, limit,
      mode: paired ? "paired" : "batch", windowSize, total: 1000, selectedCount, analyzedCount: selectedCount,
      invalidCount: 0, groupCount: 0, groups: [], windows, method: "synthetic-render-fixture" };
  },
});

window.__trainingFixture = {
  emit,
  controls,
  nativeQueries,
  get jobs() { return jobs; },
  replaceJobs: next => { jobs = next; },
  setTelemetry: next => { runtimeApi.telemetry = async () => next; },
};

const { App } = await import("../App.jsx");
createRoot(document.getElementById("root")).render(
  <LanguageProvider><ProgressFeedbackProvider>
    <Profiler id="application" onRender={() => { window.__rootCommits += 1; }}><App /></Profiler>
  </ProgressFeedbackProvider></LanguageProvider>,
);
