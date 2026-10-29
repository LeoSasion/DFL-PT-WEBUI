import test from "node:test";
import assert from "node:assert/strict";
import { canOfferRuntimeSwitch, cleanupRuntime, cudaActionBlocked, formatRuntimeBytes, normalizeCudaRuntime, projectDirectory, runtimeLabel, sameProject, scopeCudaRuntime } from "../src/cuda-runtime.js";
import { cudaPreview } from "../src/cuda-preview.js";

test("missing host assessment never becomes a simulated download suggestion", () => {
  assert.equal(normalizeCudaRuntime(undefined), null);
  assert.equal(normalizeCudaRuntime(null), null);
  assert.equal(canOfferRuntimeSwitch(null), false);
  assert.deepEqual(normalizeCudaRuntime({ gpuNames: [null, "GTX 1080 Ti", 4] }).gpuNames, ["GTX 1080 Ti"]);
});

test("download is offered only for an explicit supported target and host capability", () => {
  const assessment = cudaPreview("change-required");
  assert.equal(canOfferRuntimeSwitch(assessment), true);
  for (const patch of [{ canSwitch: false }, { canRecover: true }, { targetRuntime: "cu999" }, { status: "downloading" }]) {
    assert.equal(canOfferRuntimeSwitch({ ...assessment, ...patch }), false);
  }
  assert.equal(cudaActionBlocked({ webuiRunning: true }, ""), true);
  assert.equal(cudaActionBlocked({ terminalRunning: true }, ""), true);
  assert.equal(cudaActionBlocked({ environmentStatus: "changing" }, ""), true);
  assert.equal(cudaActionBlocked({}, "prepareCudaRuntime"), true);
  assert.equal(cudaActionBlocked({}, ""), false);
});

test("cleanup and recovery fixtures carry separate transaction-bound decisions", () => {
  const cleanup = cudaPreview("cleanup-choice");
  assert.equal(cleanup.currentRuntime, "cu126");
  assert.equal(cleanup.retainedRuntime, "cu128");
  assert.ok(cleanup.transactionId && cleanup.canDelete && cleanup.retainedBytes > 0);
  assert.equal(cudaPreview("retained").status, "retained");
  const recover = cudaPreview("recover");
  assert.equal(recover.status, "error");
  assert.equal(recover.canRecover, true);
  assert.equal(canOfferRuntimeSwitch(recover), false);
});

test("labels distinguish the enabled package and unknown space honestly", () => {
  assert.equal(runtimeLabel("cu126"), "CUDA 12.6");
  assert.equal(runtimeLabel("cu128"), "CUDA 12.8");
  assert.equal(runtimeLabel("cu126", "宿主标签"), "宿主标签");
  assert.equal(formatRuntimeBytes(2593299994), "2.6 GB");
  assert.equal(formatRuntimeBytes(4537000000), "4.5 GB");
  assert.equal(formatRuntimeBytes(null), "空间待确认");
});

test("assessment and cleanup identities remain bound to the displayed project and record", () => {
  assert.equal(sameProject("C:/Project/", "c:\\project"), true);
  assert.equal(sameProject("C:/Project", "C:/Project-other"), false);
  assert.equal(sameProject("", ""), false);
  assert.equal(projectDirectory({ projectDir: "C:/one", installPath: "C:/two" }), "C:/one");
  const scoped = scopeCudaRuntime(cudaPreview("history"), "C:/one");
  assert.equal(scoped.projectDir, "C:/one");
  assert.equal(scopeCudaRuntime(scoped, "C:/two"), null);
  const deleting = cleanupRuntime(scoped, "44444444444444444444444444444444");
  assert.equal(deleting.deleting, true);
  assert.equal(deleting.canDelete, true);
  assert.equal(deleting.retainedBytes, 4537000000);
  assert.equal(cleanupRuntime(scoped, "unknown"), null);
  assert.equal(cleanupRuntime({ ...scoped, canRecover: true }, deleting.transactionId), null);
});
