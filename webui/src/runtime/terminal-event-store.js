const MAX_EVENTS = 1400;
export const EMPTY_TERMINAL_SNAPSHOT = Object.freeze({ events: Object.freeze([]), sequence: 0 });

function scheduleFrame(callback) {
  let frame;
  const run = () => { clearTimeout(timer); globalThis.cancelAnimationFrame?.(frame); callback(); };
  const timer = setTimeout(run, 100); // Background tabs must also drain their bounded buffer.
  if (globalThis.requestAnimationFrame) frame = requestAnimationFrame(run);
  else { clearTimeout(timer); const fallback = setTimeout(callback, 0); return () => clearTimeout(fallback); }
  return () => { clearTimeout(timer); cancelAnimationFrame(frame); };
}

export function createTerminalEventStore({ schedule = scheduleFrame } = {}) {
  const snapshots = new Map();
  const pending = new Map();
  const dirty = new Set();
  const listeners = new Map();
  let cancel;
  function materialize(id) {
    const incoming = pending.get(id);
    const current = snapshots.get(id) ?? EMPTY_TERMINAL_SNAPSHOT;
    if (!incoming?.size) return current;
    pending.delete(id);
    const bySequence = new Map(current.events.map(event => [event.sequence, event]));
    for (const event of incoming.values()) if (!bySequence.has(event.sequence)) bySequence.set(event.sequence, event);
    const events = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence).slice(-MAX_EVENTS);
    const sequence = Math.max(current.sequence, ...events.map(event => event.sequence));
    if (events.length === current.events.length && events.every((event, index) => event === current.events[index]) && sequence === current.sequence) return current;
    const snapshot = Object.freeze({ events: Object.freeze(events), sequence });
    snapshots.set(id, snapshot);
    return snapshot;
  }
  function flush() {
    cancel = null;
    const ids = [...dirty]; dirty.clear();
    for (const id of ids) { materialize(id); for (const listener of listeners.get(id) ?? []) listener(); }
  }
  function remove(id) {
    snapshots.delete(id); pending.delete(id); dirty.delete(id);
    for (const listener of listeners.get(id) ?? []) listener();
  }
  return {
    append(id, events) {
      if (!id || !events.length) return;
      const buffered = pending.get(id) ?? new Map();
      for (const event of events) if (event.jobId === id && Number.isFinite(event.sequence) && !buffered.has(event.sequence)) buffered.set(event.sequence, event);
      if (!buffered.size) return;
      if (buffered.size > MAX_EVENTS * 2) {
        for (const sequence of [...buffered.keys()].sort((a, b) => a - b).slice(0, buffered.size - MAX_EVENTS)) buffered.delete(sequence);
      }
      pending.set(id, buffered); dirty.add(id);
      cancel ??= schedule(flush);
    },
    getSnapshot: id => materialize(id),
    subscribe(id, callback) {
      if (!listeners.has(id)) listeners.set(id, new Set());
      listeners.get(id).add(callback);
      return () => { const group = listeners.get(id); group?.delete(callback); if (!group?.size) listeners.delete(id); };
    },
    prune(ids) {
      for (const id of new Set([...snapshots.keys(), ...pending.keys()])) if (!ids.has(id)) remove(id);
    },
    reset() {
      cancel?.(); cancel = null;
      for (const id of new Set([...snapshots.keys(), ...pending.keys()])) remove(id);
    },
  };
}
