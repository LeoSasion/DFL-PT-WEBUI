import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { prepareFeedback } from "../../release/feedback.mjs";
import { inspectInstallation } from "../../release/installation.mjs";
import { inspectToolResources } from "../server/system-readiness.mjs";

test("feedback uses allowlisted diagnostics without private paths, project identity, raw errors or credentials", () => {
  const feedback = prepareFeedback({
    release: { appVersion: "0.1.2-preview", launcherVersion: "0.1.3-preview", installationSource: "official-source", source: { revision: "a".repeat(40), tree: "b".repeat(40), archiveSha256: "c".repeat(64), status: "content-verified" }, secret: "sk-private" },
    failureStep: "C:\\Users\\Private\\key.txt",
    diagnostics: { product: { node: "24.19.0", platform: "win32" }, workspace: { projectId: "private-person", modelCount: 2 }, telemetry: { error: "sk-private", gpus: [{ name: "private" }] }, jobs: [{ commandId: "src.extract_faces", status: "failed", args: ["sk-private"], output: "C:\\Users\\Private" }] },
  });
  const serialized = JSON.stringify(feedback);
  assert.doesNotMatch(serialized, /Private|sk-private|private-person/);
  assert.equal(feedback.failureStep, "unknown");
  assert.match(feedback.preview, /0.1.2-preview/);
  assert.match(feedback.preview, /src.extract_faces/);
  assert.equal(new URL(feedback.issueUrl).searchParams.get("body"), null);
  assert.equal(feedback.diagnostics.runtime.pythonAvailable, null);
  assert.equal(feedback.diagnostics.tasks[0].exitCode, null);
});

test("resource presence and GPU training evidence remain separate", async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), "pt-resources-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const missing = await inspectToolResources(root);
  assert.equal(missing.toolsReady, false);
  assert.equal(missing.trainingReady, null);
  for (const file of [".venv/Scripts/python.exe", "_internal/ffmpeg/ffmpeg.exe", "_internal/ffmpeg/ffprobe.exe", "_internal/DeepFaceLab/main.py", "_internal/DeepFaceLab/me.py", "_internal/DeepFaceLab/facelib/S3FD.npy", "_internal/DeepFaceLab/facelib/2DFAN.npy"]) {
    await mkdir(path.dirname(path.join(root, file)), { recursive: true });
    await writeFile(path.join(root, file), "resource fixture");
  }
  const present = await inspectToolResources(root);
  assert.equal(present.toolsReady, true);
  assert.equal(present.toolsStatus, "resource-presence-checked");
  assert.equal(present.trainingReady, null);
});

test("installation verifies source bytes and marks changed or missing source unverified", async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), "pt-source-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "release"));
  await mkdir(path.join(root, "webui", "src"), { recursive: true });
  const source = path.join(root, "webui", "src", "App.jsx");
  await writeFile(source, "approved");
  await mkdir(path.join(root, "_internal", "DeepFaceLab"), { recursive: true });
  await writeFile(path.join(root, "_internal", "DeepFaceLab", "LICENSE"), "license\r\ntext\r\n");
  await writeFile(path.join(root, "release", "version.json"), JSON.stringify({ version: "0.1.2-preview", launcherVersion: "0.1.3-preview" }));
  await writeFile(path.join(root, "release", "source-files.json"), JSON.stringify({ product: "DFL-PT-WEBUI", files: [{ path: "webui/src/App.jsx", sha256: createHash("sha256").update("approved").digest("hex") }, { path: "_internal/DeepFaceLab/LICENSE", sha256: createHash("sha256").update("license\ntext\n").digest("hex") }] }));
  assert.equal((await inspectInstallation(root)).source.verified, true);
  await writeFile(source, "changed");
  assert.equal((await inspectInstallation(root)).source.status, "modified-or-incomplete");
  await writeFile(path.join(root, "release", "source-files.json"), JSON.stringify({ product: "DFL-PT-WEBUI", files: [{ path: "../private", sha256: "a".repeat(64) }] }));
  assert.equal((await inspectInstallation(root)).source.verified, false);
});
