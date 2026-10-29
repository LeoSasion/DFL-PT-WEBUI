import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createTerminalEventStore } from "./terminal-event-store.js";
import { runtimeApi, runtimeWebSocketUrl } from "./api.js";
import { withRequestDeadline } from "./request-deadline.js";
import { activeJobStates } from "../domain/job-presentation.js";

const JOB_VIEW_EVENTS = new Set(["job.state", "job.metric", "job.progress", "terminal.prompt", "job.artifact", "job.finished"]);

export function mergeJobs(current, incoming) {
  const byId = new Map(current.map((job) => [job.id, job]));
  for (const job of incoming) {
    const existing = byId.get(job.id);
    if (
      existing
      && Number.isFinite(existing.sequence)
      && Number.isFinite(job.sequence)
      && existing.sequence > job.sequence
    ) continue;
    byId.set(job.id, { ...existing, ...job });
  }
  const merged = [...byId.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return JSON.stringify(merged) === JSON.stringify(current) ? current : merged;
}

function preserveEqualSnapshot(current, incoming) {
  return JSON.stringify(current) === JSON.stringify(incoming) ? current : incoming;
}

// A list response is authoritative for jobs known when the request began. Keep
// only newly created jobs that arrived during the request, and preserve newer
// websocket sequences for jobs that are still present in the server snapshot.
export function reconcileJobs(current, incoming, knownIds = new Set(current.map(job => job.id))) {
  const incomingIds = new Set(incoming.map(job => job.id));
  return mergeJobs(current.filter(job => incomingIds.has(job.id) || !knownIds.has(job.id)), incoming);
}

export function selectAvailableJob(jobs, selectedId) {
  return jobs.some(job => job.id === selectedId) ? selectedId
    : jobs.find(job => activeJobStates.has(job.state))?.id
      ?? jobs[0]?.id ?? null;
}

export function applyEventToJob(job, event) {
  if (job.id !== event.jobId) return job;
  if (
    Number.isFinite(job.sequence)
    && Number.isFinite(event.sequence)
    && event.sequence <= job.sequence
  ) return job;
  if (event.type === "job.state") return { ...job, ...event.payload, sequence: event.sequence };
  if (event.type === "job.metric") {
    return { ...job, latestMetric: event.payload, sequence: event.sequence };
  }
  if (event.type === "job.progress") {
    return { ...job, latestProgress: event.payload, sequence: event.sequence };
  }
  if (event.type === "terminal.prompt") {
    return { ...job, latestPrompt: event.payload.prompt, sequence: event.sequence };
  }
  if (event.type === "job.artifact") {
    if (event.payload.kind === "training-evaluation") {
      return {
        ...job,
        latestEvaluationSnapshotId: event.payload.snapshotId,
        sequence: event.sequence,
      };
    }
    return { ...job, previewVersion: event.payload.version, sequence: event.sequence };
  }
  if (event.type === "job.finished") {
    return {
      ...job,
      ...event.payload,
      endedAt: event.timestamp,
      sequence: event.sequence,
    };
  }
  return job;
}

function equalJobView(current, incoming) {
  const { sequence: currentSequence, ...currentView } = current;
  const { sequence: incomingSequence, ...incomingView } = incoming;
  return JSON.stringify(currentView) === JSON.stringify(incomingView);
}

// Snapshot ordering and visible job data have different lifetimes. Retain the
// accepted snapshot watermark even when only its sequence changed, while giving
// React the same job/array references until an actual displayed field changes.
export function createJobViewStore() {
  let jobs = [];
  const versions = new Map();
  return {
    reset() { jobs = []; versions.clear(); },
    sequence(id) { return versions.get(id) ?? 0; },
    merge(incoming, knownIds) {
      const versioned = jobs.map(job => versions.get(job.id) === job.sequence
        ? job : { ...job, sequence: versions.get(job.id) ?? job.sequence });
      const merged = knownIds
        ? reconcileJobs(versioned, incoming, knownIds)
        : mergeJobs(versioned, incoming);
      const previous = new Map(jobs.map(job => [job.id, job]));
      const next = merged.map(job => {
        if (Number.isFinite(job.sequence)) versions.set(job.id, job.sequence);
        const existing = previous.get(job.id);
        return existing && equalJobView(existing, job) ? existing : job;
      });
      const ids = new Set(next.map(job => job.id));
      for (const id of versions.keys()) if (!ids.has(id)) versions.delete(id);
      if (next.length !== jobs.length || next.some((job, index) => job !== jobs[index])) jobs = next;
      return jobs;
    },
    apply(event) {
      const index = jobs.findIndex(job => job.id === event.jobId);
      if (index < 0) return jobs;
      const current = jobs[index];
      if (!JOB_VIEW_EVENTS.has(event.type)) {
        // Keep the original full-stream ordering guard without publishing a
        // sequence-only React update. HTTP snapshots still cannot go backward.
        if (Number.isFinite(event.sequence)) versions.set(current.id, Math.max(versions.get(current.id) ?? 0, event.sequence));
        return jobs;
      }
      const baseline = versions.get(current.id) === current.sequence
        ? current : { ...current, sequence: versions.get(current.id) ?? current.sequence };
      const next = applyEventToJob(baseline, event);
      if (next === baseline) return jobs;
      if (Number.isFinite(next.sequence)) versions.set(current.id, next.sequence);
      if (equalJobView(current, next)) return jobs;
      jobs = jobs.map((job, position) => position === index ? next : job);
      return jobs;
    },
  };
}

export function useRuntime() {
  const [serviceState, setServiceState] = useState("loading");
  const [health, setHealth] = useState(null);
  const [commands, setCommands] = useState([]);
  const [telemetry, setTelemetry] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const terminalStoreRef = useRef(null);
  if (!terminalStoreRef.current) terminalStoreRef.current = createTerminalEventStore();
  const [socketState, setSocketState] = useState("disconnected");
  const [lastError, setLastError] = useState(null);
  const socketRef = useRef(null);
  const jobsRef = useRef(jobs);
  const jobViewsRef = useRef(null);
  if (!jobViewsRef.current) jobViewsRef.current = createJobViewStore();
  const refreshPending = useRef(null);
  const scopeRef = useRef(null);
  const lifecycleRef = useRef(null);
  const scopeControllerRef = useRef(null);
  const eventCursorsRef = useRef(new Map());
  const runtimeScope = health?.runtime?.current?.workspace ?? health?.project?.id;
  const publishJobs = useCallback(next => {
    if (next === jobsRef.current) return;
    jobsRef.current = next;
    setJobs(next);
  }, []);
  const mergeJobViews = useCallback((incoming, knownIds) => {
    publishJobs(jobViewsRef.current.merge(incoming, knownIds));
  }, [publishJobs]);

  useEffect(() => {
    const lifecycle = new AbortController();
    lifecycleRef.current = lifecycle;
    scopeControllerRef.current = new AbortController();
    return () => {
      lifecycle.abort();
      scopeControllerRef.current?.abort();
      refreshPending.current = null;
      terminalStoreRef.current.reset();
    };
  }, []);

  const refresh = useCallback(({ quiet = false } = {}) => {
    const lifecycle = lifecycleRef.current;
    if (!lifecycle || lifecycle.signal.aborted) return Promise.resolve([]);
    if (refreshPending.current) return refreshPending.current;
    const knownIds = new Set(jobsRef.current.map(job => job.id));
    if (!quiet) setServiceState("loading");
    const pending = (async () => { try {
      // Keep state commits outside the bounded action: an abort-ignoring
      // transport may resolve after the deadline or a later mount.
      const { nextHealth, nextCommands, nextJobs } = await withRequestDeadline(async signal => {
        const nextHealth = await runtimeApi.health({ signal });
        if (signal.aborted) throw signal.reason;
        const [nextCommands, nextJobs] = await Promise.all([
          runtimeApi.commands({ signal }),
          runtimeApi.jobs({ signal }),
        ]);
        return { nextHealth, nextCommands, nextJobs };
      }, { signal: lifecycle.signal });
      if (lifecycle.signal.aborted || lifecycleRef.current !== lifecycle) return [];
      const nextScope = nextHealth.runtime?.current?.workspace ?? nextHealth.project?.id;
      const scopeChanged = scopeRef.current !== null && scopeRef.current !== nextScope;
      scopeRef.current = nextScope;
      if (scopeChanged) {
        // Invalidate observers immediately, before React runs their cleanups.
        scopeControllerRef.current?.abort();
        scopeControllerRef.current = new AbortController();
        eventCursorsRef.current.clear();
        jobViewsRef.current.reset();
        terminalStoreRef.current.reset();
        setSelectedJobId(null);
        setTelemetry(null);
      }
      setHealth((current) => preserveEqualSnapshot(current, nextHealth));
      setCommands((current) => preserveEqualSnapshot(current, nextCommands));
      mergeJobViews(nextJobs, knownIds);
      setServiceState("online");
      setLastError(null);
      return nextJobs;
    } catch (error) {
      if (lifecycle.signal.aborted || lifecycleRef.current !== lifecycle) return [];
      setServiceState("offline");
      setSocketState("disconnected");
      setLastError(error);
      return [];
    } })();
    refreshPending.current = pending;
    void pending.finally(() => { if (refreshPending.current === pending) refreshPending.current = null; });
    return pending;
  }, [mergeJobViews]);

  useEffect(() => {
    setSelectedJobId(current => selectAvailableJob(jobs, current));
    const ids = new Set(jobs.map(job => job.id));
    for (const id of eventCursorsRef.current.keys()) {
      if (!ids.has(id)) eventCursorsRef.current.delete(id);
    }
    terminalStoreRef.current.prune(ids);
  }, [jobs]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const delay = serviceState === "online" ? 5000 : 2000;
    const interval = window.setInterval(() => void refresh({ quiet: true }), delay);
    return () => window.clearInterval(interval);
  }, [refresh, serviceState]);

  useEffect(() => {
    if (serviceState !== "online") {
      setTelemetry(null);
      return undefined;
    }
    let cancelled = false;
    const controller = new AbortController();
    const scopeSignal = scopeControllerRef.current.signal;
    const abort = () => controller.abort();
    scopeSignal.addEventListener("abort", abort, { once: true });
    let timer;
    const sample = async () => {
      try {
        const next = await withRequestDeadline(signal => runtimeApi.telemetry({ signal }), { signal: controller.signal });
        if (!cancelled && !controller.signal.aborted) setTelemetry(current => preserveEqualSnapshot(current, next));
      } catch {
        if (!cancelled && !controller.signal.aborted) setTelemetry({ available: false, gpus: [], error: "GPU 遥测暂时不可用" });
      } finally {
        // Schedule after completion so slow samples cannot overlap or arrive
        // out of order. The deadline also releases an unresponsive sample.
        if (!cancelled && !controller.signal.aborted) timer = window.setTimeout(sample, 3000);
      }
    };
    void sample();
    return () => {
      cancelled = true;
      controller.abort();
      scopeSignal.removeEventListener("abort", abort);
      window.clearTimeout(timer);
    };
  }, [serviceState, runtimeScope]);

  const applyEvent = useCallback((event) => {
    if (!event?.jobId || !jobsRef.current.some(job => job.id === event.jobId)) return;
    terminalStoreRef.current.append(event.jobId, [event]);
    publishJobs(jobViewsRef.current.apply(event));
  }, [publishJobs]);

  useEffect(() => {
    if (serviceState !== "online" || !selectedJobId) {
      socketRef.current?.close();
      socketRef.current = null;
      setSocketState("disconnected");
      return undefined;
    }

    let cancelled = false;
    const controller = new AbortController();
    const scopeSignal = scopeControllerRef.current.signal;
    const abort = () => controller.abort();
    scopeSignal.addEventListener("abort", abort, { once: true });
    let reconnectTimer;
    let retry = 0;
    let activeSocket;
    const isCurrent = () => !cancelled && !controller.signal.aborted;
    const advanceCursor = (event) => {
      if (event.jobId !== selectedJobId || !Number.isFinite(event.sequence)) return;
      eventCursorsRef.current.set(selectedJobId, Math.max(eventCursorsRef.current.get(selectedJobId) ?? 0, event.sequence));
    };

    const connect = async () => {
      if (!isCurrent()) return;
      setSocketState(retry ? "reconnecting" : "connecting");
      let after = eventCursorsRef.current.get(selectedJobId) ?? 0;
      try {
        const backlog = await withRequestDeadline(signal => runtimeApi.events(selectedJobId, after, { signal }), { signal: controller.signal });
        if (!isCurrent()) return;
        for (const event of backlog) advanceCursor(event);
        terminalStoreRef.current.append(selectedJobId, backlog);
        after = eventCursorsRef.current.get(selectedJobId) ?? after;
      } catch (error) {
        if (!isCurrent()) return;
        setLastError(error);
        if ([404, 410].includes(error?.status)) {
          setSocketState("disconnected");
          void refresh({ quiet: true });
          return;
        }
      }

      if (!isCurrent()) return;
      const socket = new WebSocket(runtimeWebSocketUrl(selectedJobId, after));
      activeSocket = socket;
      socketRef.current = socket;
      socket.addEventListener("open", () => {
        if (!isCurrent() || activeSocket !== socket) return;
        retry = 0;
        setSocketState("connected");
      });
      socket.addEventListener("message", (message) => {
        if (!isCurrent() || activeSocket !== socket) return;
        try {
          const data = JSON.parse(message.data);
          if (data.type === "snapshot") {
            if (data.payload.id !== selectedJobId) return;
            if (jobsRef.current.some(job => job.id === data.payload.id)) mergeJobViews([data.payload]);
          } else {
            if (data.jobId !== selectedJobId) return;
            advanceCursor(data);
            applyEvent(data);
          }
        } catch {
          setLastError(new Error("终端收到无效消息"));
        }
      });
      socket.addEventListener("close", () => {
        if (!isCurrent() || activeSocket !== socket) return;
        activeSocket = null;
        if (socketRef.current === socket) socketRef.current = null;
        setSocketState("reconnecting");
        const delay = Math.min(800 * (2 ** retry), 5000);
        retry += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      });
      socket.addEventListener("error", () => socket.close());
    };

    void connect();
    return () => {
      cancelled = true;
      controller.abort();
      scopeSignal.removeEventListener("abort", abort);
      window.clearTimeout(reconnectTimer);
      activeSocket?.close();
      if (socketRef.current === activeSocket) socketRef.current = null;
    };
  }, [applyEvent, mergeJobViews, selectedJobId, serviceState, refresh, runtimeScope]);

  const startJob = useCallback(async (commandId, options = {}) => {
    const job = await runtimeApi.start(commandId, options);
    mergeJobViews([job]);
    setSelectedJobId(job.id);
    return job;
  }, [mergeJobViews]);

  const sendInput = useCallback(async (input) => {
    if (!selectedJobId) throw new Error("尚未选择任务");
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "terminal.input", input }));
      return;
    }
    await runtimeApi.input(selectedJobId, input);
  }, [selectedJobId]);

  const resize = useCallback((cols, rows) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "terminal.resize", cols, rows }));
    }
  }, []);

  const control = useCallback(async (operation, jobId = selectedJobId) => {
    if (!jobId) throw new Error("尚未选择任务");
    const job = await runtimeApi.control(jobId, operation);
    mergeJobViews([job]);
    return job;
  }, [mergeJobViews, selectedJobId]);

  const retryJob = useCallback(async (jobId) => {
    if (!jobId) throw new Error("尚未选择任务");
    const job = await runtimeApi.retry(jobId);
    mergeJobViews([job]);
    setSelectedJobId(job.id);
    return job;
  }, [mergeJobViews]);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? null,
    [jobs, selectedJobId],
  );
  return {
    serviceState,
    health,
    commands,
    telemetry,
    jobs,
    selectedJob,
    selectedJobId,
    terminalStore: terminalStoreRef.current,
    // Compatibility reads do not subscribe the application root to terminal data.
    get selectedJobSequence() { return Math.max(jobViewsRef.current.sequence(selectedJobId), eventCursorsRef.current.get(selectedJobId) ?? 0); },
    get selectedEvents() { return terminalStoreRef.current.getSnapshot(selectedJobId).events; },
    socketState,
    lastError,
    refresh,
    preflight: runtimeApi.preflight,
    selectJob: setSelectedJobId,
    startJob,
    sendInput,
    resize,
    control,
    retryJob,
  };
}
