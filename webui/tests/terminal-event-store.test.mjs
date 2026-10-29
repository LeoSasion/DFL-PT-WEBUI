import assert from "node:assert/strict";
import test from "node:test";
import { createTerminalEventStore } from "../src/runtime/terminal-event-store.js";

const event = (sequence, jobId = "a") => ({ jobId, sequence, type: "terminal.output", payload: { data: String(sequence) } });
function fixture() {
  const scheduled = new Set();
  const store = createTerminalEventStore({ schedule: callback => { scheduled.add(callback); return () => scheduled.delete(callback); } });
  return { store, flush() { for (const callback of [...scheduled]) { scheduled.delete(callback); callback(); } }, scheduled };
}
test("bursts publish once to the selected terminal, preserve sequence order and deduplicate replay", () => {
  const { store, flush, scheduled } = fixture(); let updates = 0; let other = 0;
  const unsubscribe = store.subscribe("a", () => updates++); store.subscribe("b", () => other++);
  for (let sequence = 1; sequence <= 100; sequence++) store.append("a", [event(sequence)]);
  store.append("a", [event(10), event(43), event(42)]);
  assert.equal(scheduled.size, 1); assert.equal(updates, 0);
  flush(); assert.equal(updates, 1); assert.equal(other, 0);
  const value = store.getSnapshot("a"); assert.equal(value.sequence, 100); assert.equal(value.events.length, 100);
  assert.equal(store.getSnapshot("a"), value, "unchanged snapshots keep their React identity");
  unsubscribe(); store.append("a", [event(101)]); flush(); assert.equal(updates, 1);
});
test("bounded buffers keep the newest events even through a large out-of-order replay", () => {
  const { store, flush } = fixture();
  for (let sequence = 1; sequence <= 6000; sequence++) store.append("a", [event(sequence)]);
  for (let sequence = 4000; sequence >= 1; sequence--) store.append("a", [event(sequence)]);
  flush(); const value = store.getSnapshot("a");
  assert.equal(value.events.length, 1400); assert.equal(value.events[0].sequence, 4601); assert.equal(value.sequence, 6000);
});
test("scope reset cancels pending notifications and job pruning removes stale terminal history", () => {
  const { store, flush, scheduled } = fixture(); let updates = 0;
  store.subscribe("a", () => updates++); store.append("a", [event(1)]);
  store.reset(); assert.equal(scheduled.size, 0); flush(); assert.equal(updates, 1);
  assert.equal(store.getSnapshot("a").events.length, 0);
  store.append("a", [event(2)]); store.append("b", [event(3, "b")]); flush();
  store.prune(new Set(["b"])); assert.equal(store.getSnapshot("a").events.length, 0); assert.equal(store.getSnapshot("b").sequence, 3);
});
