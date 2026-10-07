import test from "node:test";
import assert from "node:assert/strict";
import { bindSceneCuts, normalizeSegments } from "../server/video-tool-manager.mjs";

const digest = "a".repeat(64);
const frame = (index, segmentIndex = 0) => ({ sourceFrameIndex: index, pts: 1000 + index * 3003, timeBase: [1, 90000], segmentIndex });
const detection = { schemaVersion: 2, source: { sha256: digest }, sourceFrameCount: 100, timeBase: [1, 90000],
  cuts: [frame(20), frame(40)], algorithm: "ffmpeg" };
test("drop-only sampling retains cuts between retained frames and resets at segment boundaries", () => {
  const timeline = { source: { sha256: digest }, frames: [frame(10), frame(25), frame(30), frame(60, 1)] };
  const result = bindSceneCuts(timeline, detection);
  assert.deepEqual(result.frames.map(value => [value.shotId, value.cutBefore]), [[0, true], [1, true], [1, false], [2, true]]);
  assert.equal(result.sceneDetection.cutCount, 2);
  assert.equal(bindSceneCuts({ ...timeline, source: { sha256: "b".repeat(64) } }, detection).sceneDetectionApplied, false);
});
test("equal source identity still rejects conflicting cut PTS", () => {
  assert.throws(() => bindSceneCuts({ source: { sha256: digest }, frames: [{ ...frame(20), pts: 123 }] }, detection), /PTS/);
  assert.throws(() => bindSceneCuts({ source: { sha256: digest }, frames: [frame(20)] }, { ...detection, cuts: [frame(40), frame(20)] }), /索引/);
});
test("scene-derived segment preserves its integer bounds and unrounded seconds", () => {
  const start = 3003 / 90000, end = 30030 / 90000;
  const range = { sourceSha256: digest, startIndex: 1, endIndexExclusive: 10, startPts: 4003, endPts: 31030, timeBase: [1, 90000] };
  const [result] = normalizeSegments([{ start, end, sourceFrameRange: range }], 10);
  assert.equal(result.start, start); assert.equal(result.end, end); assert.deepEqual(result.sourceFrameRange, range);
});

test("a verified one-frame scene may be shorter than the manual segment minimum", () => {
  const sourceFrameRange = { sourceSha256: digest, startIndex: 0, endIndexExclusive: 1, startPts: 0, endPts: 40, timeBase: [1, 1000] };
  assert.equal(normalizeSegments([{ start: 0, end: .04, sourceFrameRange }], 1).length, 1);
  assert.throws(() => normalizeSegments([{ start: 0, end: .04 }], 1), /时间范围/);
});
