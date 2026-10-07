import assert from "node:assert/strict";
import test from "node:test";
import {
  buildTrainingPoseRegression,
  snapshotCompatibility,
} from "../src/domain/training-pose-regression.js";

const signature = {
  class: "ME",
  resolution: 64,
  faceType: "wf",
  architecture: "df-d",
  dataFormat: "NCHW",
};

function sample(id, cellId, metrics, side = "dst") {
  return {
    id,
    side,
    cellId,
    yaw: Number(cellId.split("-y")[1]),
    pitch: Number(cellId.slice(1).split("-y")[0]),
    metrics: { reconstruction: metrics },
    variants: ["input", "reconstruction"],
  };
}

function snapshot(overrides = {}) {
  return {
    schemaVersion: 1,
    snapshotId: "iter-00001000-aaaaaaaa",
    modelKey: "quality-smoke-me-123456789abc",
    manifestId: "a".repeat(24),
    iteration: 1000,
    metricSchemaVersion: 1,
    modelSignature: signature,
    createdAt: "2026-08-03T00:00:00.000Z",
    samples: [],
    ...overrides,
  };
}

test("snapshot compatibility exposes the four required comparison gates", () => {
  const baseline = snapshot();
  const current = snapshot({ snapshotId: "iter-00002000-bbbbbbbb", iteration: 2000 });
  const compatible = snapshotCompatibility(baseline, current);
  assert.equal(compatible.comparable, true);
  assert.deepEqual(compatible.checks.map((check) => check.id), [
    "model",
    "manifest",
    "metrics",
    "signature",
  ]);

  const mismatches = [
    ["model", { modelKey: "different-me-123456789abc" }],
    ["manifest", { manifestId: "b".repeat(24) }],
    ["metrics", { metricSchemaVersion: 2 }],
    ["signature", { modelSignature: { ...signature, resolution: 128 } }],
  ];
  for (const [id, override] of mismatches) {
    const result = snapshotCompatibility(baseline, { ...current, ...override });
    assert.equal(result.comparable, false);
    assert.equal(result.checks.find((check) => check.id === id).passed, false);
  }
});

test("pose regression keeps confidence separate and classifies metric direction", () => {
  const baseline = snapshot({
    samples: [
      sample("dst-p0-y0-01", "p0-y0", {
        maskedMse: 0.1,
        eyesMouthMse: 0.12,
        maskDice: 0.8,
        sharpnessRatio: 0.6,
      }),
      sample("dst-p0-y15-01", "p0-y15", {
        maskedMse: 0.08,
        eyesMouthMse: 0.08,
        maskDice: 0.9,
        sharpnessRatio: 1,
      }),
    ],
  });
  const current = snapshot({
    snapshotId: "iter-00002000-bbbbbbbb",
    iteration: 2000,
    samples: [
      sample("dst-p0-y0-01", "p0-y0", {
        maskedMse: 0.08,
        eyesMouthMse: 0.09,
        maskDice: 0.86,
        sharpnessRatio: 0.8,
      }),
      sample("dst-p0-y15-01", "p0-y15", {
        maskedMse: 0.1,
        eyesMouthMse: 0.11,
        maskDice: 0.82,
        sharpnessRatio: 1.2,
      }),
    ],
  });
  const manifest = {
    probes: {
      dst: { yawTicks: [0, 15], pitchTicks: [0] },
    },
  };
  const result = buildTrainingPoseRegression({ baseline, current, manifest });
  const improved = result.cells.find((cell) => cell.id === "p0-y0");
  const regressed = result.cells.find((cell) => cell.id === "p0-y15");

  assert.equal(result.comparable, true);
  assert.equal(improved.status, "improved");
  assert.equal(regressed.status, "regressed");
  assert.equal(improved.confidence, 1 / 3);
  assert.equal(improved.sampleCount, 1);
  assert.equal(improved.metrics.maskDice.status, "improved");
  assert.equal(improved.metrics.sharpnessRatio.status, "improved");
  assert.equal(result.totals.improved, 1);
  assert.equal(result.totals.regressed, 1);
  assert.ok(!Object.hasOwn(result, "score"));
});

test("ME bridge flat reconstruction metrics populate shared pose cells", () => {
  const flatSample = (id, metrics, side = "dst") => ({
    ...sample(id, "p0-y15", {}, side),
    metrics,
    variants: ["input", "reconstruction", "target-mask", "predicted-mask"],
  });
  for (const side of ["src", "dst"]) {
    const baseline = snapshot({
      samples: [1, 2, 3].map((index) => flatSample(`${side}-p0-y15-0${index}`, {
        maskedMse: 0.007,
        eyesMouthMse: 0.02,
        maskDice: 0.8,
        sharpnessRatio: 0.2,
        ...(side === "dst" ? { swap: { maskDice: 0.7, sharpnessRatio: 0.1 } } : {}),
      }, side)),
    });
    const current = snapshot({
      iteration: 2000,
      samples: [1, 2, 3].map((index) => flatSample(`${side}-p0-y15-0${index}`, {
        maskedMse: 0.004,
        eyesMouthMse: 0.01,
        maskDice: 0.9,
        sharpnessRatio: 0.3,
        ...(side === "dst" ? { swap: { maskDice: 0.75, sharpnessRatio: 0.15 } } : {}),
      }, side)),
    });
    const result = buildTrainingPoseRegression({ baseline, current, side });
    const cell = result.cells.find(({ id }) => id === "p0-y15");
    assert.equal(result.sharedSampleCount, 3);
    assert.equal(cell.sampleCount, 3);
    assert.equal(cell.confidence, 1);
    assert.equal(cell.status, "improved");
    for (const metric of Object.values(cell.metrics)) {
      assert.equal(metric.sampleCount, 3);
      assert.equal(metric.status, "improved");
    }
    assert.equal(result.totals.improved, 1);
  }
});

test("explicit reconstruction metrics take precedence over flat snapshot keys", () => {
  const baseline = snapshot({
    samples: [{
      ...sample("dst-p0-y0-01", "p0-y0", { maskedMse: 0.1 }),
      metrics: { maskedMse: 0.2, reconstruction: { maskedMse: 0.1 } },
    }],
  });
  const current = snapshot({
    samples: [{
      ...sample("dst-p0-y0-01", "p0-y0", { maskedMse: 0.08 }),
      metrics: { maskedMse: 0.4, reconstruction: { maskedMse: 0.08 } },
    }],
  });
  const cell = buildTrainingPoseRegression({ baseline, current }).cells
    .find(({ id }) => id === "p0-y0");
  assert.equal(cell.status, "improved");
  assert.equal(cell.selectedMetric.baseline, 0.1);
  assert.equal(cell.selectedMetric.current, 0.08);

  for (const reconstruction of [null, {}, { maskedMse: null }]) {
    const invalidCurrent = {
      ...current,
      samples: [{ ...current.samples[0], metrics: { maskedMse: 0.04, reconstruction } }],
    };
    const invalidCell = buildTrainingPoseRegression({ baseline, current: invalidCurrent }).cells
      .find(({ id }) => id === "p0-y0");
    assert.equal(invalidCell.status, "empty");
    assert.equal(invalidCell.sampleCount, 0);
  }
});

test("swap metrics never fall back to reconstruction or flat snapshot keys", () => {
  const baseline = snapshot({
    samples: [{
      ...sample("dst-p0-y0-01", "p0-y0", { maskDice: 0.5 }),
      metrics: { maskDice: 0.5, reconstruction: { maskDice: 0.5 } },
    }],
  });
  const current = snapshot({
    samples: [{
      ...sample("dst-p0-y0-01", "p0-y0", { maskDice: 0.8 }),
      metrics: { maskDice: 0.8, reconstruction: { maskDice: 0.8 } },
    }],
  });
  const options = { baseline, current, channel: "swap", metricKey: "maskDice" };
  const missing = buildTrainingPoseRegression(options).cells
    .find(({ id }) => id === "p0-y0");
  assert.equal(missing.status, "empty");
  assert.equal(missing.sampleCount, 0);

  const withSwap = (snapshot, maskDice) => ({
    ...snapshot,
    samples: [{ ...snapshot.samples[0], metrics: { ...snapshot.samples[0].metrics, swap: { maskDice } } }],
  });
  const cell = buildTrainingPoseRegression({
    ...options,
    baseline: withSwap(baseline, 0.9),
    current: withSwap(current, 0.8),
  }).cells.find(({ id }) => id === "p0-y0");
  assert.equal(cell.status, "regressed");
  assert.equal(cell.selectedMetric.baseline, 0.9);
  assert.equal(cell.selectedMetric.current, 0.8);
});
