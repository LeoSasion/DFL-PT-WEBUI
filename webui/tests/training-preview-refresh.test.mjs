import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_PREVIEW_REFRESH_SECONDS,
  PREVIEW_REFRESH_STORAGE_KEY,
  parsePreviewRefreshSeconds,
  planTrainingPreviewDisplay,
  readPreviewRefreshSeconds,
} from "../src/domain/training-preview-refresh.js";

const input = (latestVersion, overrides = {}) => ({
  jobId: "training-one",
  latestVersion,
  manualRefresh: 0,
  intervalSeconds: DEFAULT_PREVIEW_REFRESH_SECONDS,
  active: true,
  ...overrides,
});

test("preview opens immediately, then coalesces frequent Trainer updates into one three-minute display refresh", () => {
  const first = planTrainingPreviewDisplay(null, input(1), 10_000).displayed;
  assert.equal(first.version, 1);
  const waiting = planTrainingPreviewDisplay(first, input(5), 70_000);
  assert.equal(waiting.displayed, first);
  assert.equal(waiting.delayMs, 120_000);
  const stillWaiting = planTrainingPreviewDisplay(first, input(17), 189_999);
  assert.equal(stillWaiting.displayed.version, 1);
  assert.equal(stillWaiting.delayMs, 1);
  const due = planTrainingPreviewDisplay(first, input(18), 190_000);
  assert.equal(due.displayed.version, 18);
  assert.equal(due.delayMs, null);
});

test("manual refresh and first preview bypass the timer while other jobs remain isolated", () => {
  const waitingForFirst = planTrainingPreviewDisplay(null, input(null), 100).displayed;
  assert.equal(waitingForFirst.version, null);
  const first = planTrainingPreviewDisplay(waitingForFirst, input(1), 200).displayed;
  const manual = planTrainingPreviewDisplay(first, input(1, { manualRefresh: 1 }), 300).displayed;
  assert.equal(manual.manualRefresh, 1);
  assert.equal(manual.awaitManualVersion, true);
  const generated = planTrainingPreviewDisplay(manual, input(2, { manualRefresh: 1 }), 400).displayed;
  assert.equal(generated.version, 2);
  assert.equal(generated.awaitManualVersion, false);
  const otherJob = planTrainingPreviewDisplay(generated, input(6, { jobId: "training-two" }), 500).displayed;
  assert.equal(otherJob.jobId, "training-two");
  assert.equal(otherJob.version, 6);
  const finished = planTrainingPreviewDisplay(otherJob, input(7, { jobId: "training-two", active: false }), 600).displayed;
  assert.equal(finished.version, 7);
});

test("refresh preference validates whole seconds and safely falls back to 180", () => {
  assert.equal(parsePreviewRefreshSeconds(" 30 "), 30);
  for (const invalid of ["", "4", "3601", "30.5", "-20", "abc", null, Infinity]) {
    assert.equal(parsePreviewRefreshSeconds(invalid), null);
  }
  assert.equal(readPreviewRefreshSeconds({ getItem: () => "45" }), 45);
  assert.equal(readPreviewRefreshSeconds({ getItem: () => "invalid" }), 180);
  assert.equal(readPreviewRefreshSeconds({ getItem: () => { throw new Error("blocked"); } }), 180);
  assert.equal(PREVIEW_REFRESH_STORAGE_KEY, "dfl-webui-training-preview-refresh-seconds-v1");
});

test("changing the interval reschedules the pending preview without extra requests", () => {
  const first = planTrainingPreviewDisplay(null, input(1), 0).displayed;
  const shorter = planTrainingPreviewDisplay(first, input(2, { intervalSeconds: 30 }), 20_000);
  assert.equal(shorter.delayMs, 10_000);
  const longer = planTrainingPreviewDisplay(first, input(2, { intervalSeconds: 300 }), 20_000);
  assert.equal(longer.delayMs, 280_000);
  const nowDue = planTrainingPreviewDisplay(first, input(2, { intervalSeconds: 10 }), 20_000);
  assert.equal(nowDue.displayed.version, 2);
});
