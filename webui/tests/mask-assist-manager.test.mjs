import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { EventEmitter } from "node:events";
import { lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { PATHS } from "../server/paths.mjs";
import { MASK_ASSIST_PROFILE, createMaskAssistDraft, publishMaskAssistCopies, readMaskAssistDraft, resolveMaskAssistFile, runPython } from "../server/mask-assist-manager.mjs";
import { terminateChildProcessTree } from "../server/process-tree.mjs";

function hash(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

function isRunning(pid) {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error.code === "ESRCH") return false; throw error; }
}

for (const cause of ["termination rejects", "stream never closes"]) {
  test(`mask Python ${cause} reports an unconfirmed stop without a child close event`, { timeout: 3000 }, async () => {
    const isolated = await import(`../server/helper-process.mjs?mask-stop=${encodeURIComponent(cause)}-${Date.now()}`);
    const child = new EventEmitter();
    child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
    child.stdin = new EventEmitter(); child.stdin.end = () => {};
    const controller = new AbortController();
    const work = runPython([], { signal: controller.signal,
      runProcess: (executable, args, options) => isolated.runHelperProcess(executable, args, { ...options,
        spawnProcess: () => child, closeWaitMs: 20,
        terminateProcessTree: async () => { if (cause === "termination rejects") throw new Error("controlled taskkill access denied"); },
      }),
    });
    controller.abort();
    try {
      const result = await Promise.race([work.then(() => null, error => error), new Promise(resolve => setTimeout(() => resolve("not-settled"), 500))]);
      assert.notEqual(result, "not-settled", "A failed stop must settle even without a child close event");
      assert.equal(result.code, "MASK_STOP_UNCONFIRMED");
      assert.equal(isolated.activeHelperProcesses()[0].state, "stop-unconfirmed");
      assert.throws(() => isolated.assertHelperStopsConfirmed(), { code: "HELPER_STOP_UNCONFIRMED" });
      await assert.rejects(isolated.closeHelperProcesses(), { code: "HELPER_STOP_UNCONFIRMED" });
      child.emit("close", 1);
      assert.throws(() => isolated.assertHelperStopsConfirmed(), { code: "HELPER_STOP_UNCONFIRMED" });
    } finally {
      child.emit("close", 1);
      await work.catch(() => {});
    }
  });
}

for (const cause of ["cancel", "timeout", "response overflow"]) {
  test(`mask Python ${cause} waits for the Windows venv launcher and base interpreter to exit`, {
    skip: process.platform !== "win32" || !process.env.DFLSN_ISOLATED_TEST_ROOT,
    timeout: 15_000,
  }, async (t) => {
    assert.ok(path.resolve(PATHS.workspaceRoot).startsWith(path.resolve(process.env.DFLSN_ISOLATED_TEST_ROOT) + path.sep));
    const controller = new AbortController();
    let pids, aliveAtStart;
    const script = ["import os,json,time", "print(json.dumps({'progress':{'completed':os.getpid(),'total':os.getppid()}}),flush=True)",
      "time.sleep(0.2)", ...(cause === "response overflow" ? ["print('x'*4096,flush=True)"] : []), "time.sleep(60)"].join("\n");
    try {
      await assert.rejects(runPython(["-u", "-c", script], {
        signal: controller.signal, timeoutMs: cause === "timeout" ? 2000 : 10_000,
        maxResponseBytes: cause === "response overflow" ? 512 : 1024 * 1024,
        onProgress: ({ current, total }) => {
          pids = { interpreter: current, launcher: total };
          aliveAtStart = isRunning(current) && isRunning(total);
          t.diagnostic(JSON.stringify({ cause, ...pids, aliveAtStart }));
          if (cause === "cancel") controller.abort();
        },
      }), (error) => error.code === ({ cancel: "ABORT_ERR", timeout: "MASK_INFERENCE_TIMEOUT", "response overflow": "MASK_RESPONSE_TOO_LARGE" })[cause]);
      assert.ok(pids, "the stdlib fixture must actually start before termination");
      assert.notEqual(pids.launcher, process.pid, "the parent must be the Windows venv launcher, not the test runner");
      assert.notEqual(pids.interpreter, pids.launcher);
      assert.equal(aliveAtStart, true, "both observed processes must exist before termination");
      assert.equal(isRunning(pids.launcher), false, "runner must not settle while its venv launcher remains alive");
      assert.equal(isRunning(pids.interpreter), false, "runner must not settle while python_base continues running");
    } finally {
      controller.abort();
      if (pids) t.diagnostic(JSON.stringify({ cause, launcherRunningAtSettlement: isRunning(pids.launcher), interpreterRunningAtSettlement: isRunning(pids.interpreter) }));
      // Failure cleanup is limited to the exact fixture PIDs observed in its protocol.
      for (const pid of pids ? [pids.interpreter, pids.launcher] : []) {
        if (pid !== process.pid && isRunning(pid)) await terminateChildProcessTree({ pid });
      }
    }
  });
}

test("native mask selection rejects invalid names before inference", async () => {
  await assert.rejects(createMaskAssistDraft("dst", { names: ["../face.jpg"] }), (error) => error.code === "MASK_NAMES_INVALID");
  await assert.rejects(createMaskAssistDraft("dst", { names: ["face.jpg", "FACE.jpg"] }), (error) => error.code === "MASK_NAMES_INVALID");
});

test("mask draft keeps worker-owned staging when its child cannot be stopped", { skip: !process.env.DFLSN_ISOLATED_TEST_ROOT, timeout: 3000 }, async () => {
  assert.ok(path.resolve(PATHS.workspaceRoot).startsWith(path.resolve(process.env.DFLSN_ISOLATED_TEST_ROOT) + path.sep));
  const isolated = await import(`../server/helper-process.mjs?mask-staging-stop=${Date.now()}`);
  const child = new EventEmitter();
  child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
  child.stdin = new EventEmitter(); child.stdin.end = () => {};
  const controller = new AbortController();
  const name = `stop-fixture-${Date.now()}.jpg`, original = path.join(PATHS.workspaceRoot, "data_dst", "aligned", name);
  let staging, ready;
  const started = new Promise(resolve => { ready = resolve; });
  await writeFile(original, "controlled unchanged original", { flag: "wx" });
  const work = createMaskAssistDraft("dst", { names: [name], signal: controller.signal,
    runProcess: async (executable, args, options) => {
      staging = args[args.indexOf("--output") + 1];
      await mkdir(staging);
      await writeFile(path.join(staging, "partial.txt"), "worker-owned partial output");
      const pending = isolated.runHelperProcess(executable, args, { ...options, spawnProcess: () => child,
        terminateProcessTree: async () => { throw new Error("controlled taskkill access denied"); },
      });
      ready(); return pending;
    },
  });
  try {
    await started; controller.abort();
    await assert.rejects(work, error => error.code === "MASK_STOP_UNCONFIRMED" && error.details.retainedStaging === staging);
    assert.equal(await readFile(path.join(staging, "partial.txt"), "utf8"), "worker-owned partial output");
    assert.equal(await readFile(original, "utf8"), "controlled unchanged original");
    child.emit("close", 1);
    assert.equal((await lstat(staging)).isDirectory(), true);
    assert.throws(() => isolated.assertHelperStopsConfirmed(), { code: "HELPER_STOP_UNCONFIRMED" });
  } finally {
    child.emit("close", 1); await work.catch(() => {});
    await rm(original, { force: true });
    if (staging) await rm(staging, { recursive: true, force: true });
  }
});

test("reviewed mask publication builds an independent full dataset and rejects stale inputs", { skip: !process.env.DFLSN_ISOLATED_TEST_ROOT }, async () => {
  assert.ok(path.resolve(PATHS.workspaceRoot).startsWith(path.resolve(process.env.DFLSN_ISOLATED_TEST_ROOT) + path.sep));
  const aligned = path.join(PATHS.workspaceRoot, "data_dst", "aligned");
  const saved = await mkdtemp(path.join(PATHS.workspaceRoot, ".mask-test-original-"));
  const original = path.join(saved, "aligned");
  await rename(aligned, original);
  const id = "mask-123456789012345678901234";
  const draft = path.join(PATHS.runtimeRoot, "mask-assist", "dst", id);
  const published = path.join(PATHS.workspaceRoot, "data_dst", "aligned_assisted", id);
  let source, assisted, other;
  const originalFrame = path.join(PATHS.workspaceRoot, "data_dst", "mask-binding-test.png");
  try {
    await mkdir(aligned);
    await mkdir(path.join(draft, "copies"), { recursive: true });
    const fixture = spawnSync(PATHS.python, ["-c", [
      "import sys,cv2,numpy as np", "from pathlib import Path",
      "sys.path.insert(0,sys.argv[1])", "from DFLIMG import DFLJPG",
      "a,d,f=map(Path,sys.argv[2:])", "image=np.full((64,64,3),120,np.uint8)",
      "cv2.imwrite(str(f),image)", "points=np.column_stack((np.linspace(10,54,68),np.linspace(12,52,68))).astype(np.float32)",
      "for name in ('face.jpg','other.jpg'):", " cv2.imwrite(str(a/name),image)",
      " obj=DFLJPG.load(str(a/name));obj.set_face_type('whole_face');obj.set_landmarks(points);obj.set_source_landmarks(points);obj.set_source_filename(f.name);obj.set_image_to_face_mat(np.array([[1,0,0],[0,1,0]],np.float32));obj.save()",
      "obj=DFLJPG.load(str(a/'face.jpg'));obj.set_xseg_mask(np.ones((64,64,1),np.float32));obj.filename=str(d/'face.jpg');obj.save()",
    ].join("\n"), PATHS.currentDflRoot, aligned, path.join(draft, "copies"), originalFrame], { encoding: "utf8", windowsHide: true });
    assert.equal(fixture.status, 0, fixture.stderr);
    source = await readFile(path.join(aligned, "face.jpg"));
    assisted = await readFile(path.join(draft, "copies", "face.jpg"));
    other = await readFile(path.join(aligned, "other.jpg"));
    const report = { schemaVersion: 1, id, side: "dst", status: "ready", profile: MASK_ASSIST_PROFILE, semantics: { iou: null },
      entries: [{ file: "face.jpg", inputSha256: hash(source), copySha256: hash(assisted) }] };
    await writeFile(path.join(draft, "report.json"), JSON.stringify(report));
    await assert.rejects(publishMaskAssistCopies("dst", id), (error) => error.code === "MASK_REVIEW_REQUIRED");
    await writeFile(path.join(aligned, "face.jpg"), "stale");
    await assert.rejects(publishMaskAssistCopies("dst", id, { reviewed: true }), (error) => error.code === "MASK_SOURCE_CHANGED");
    await writeFile(path.join(aligned, "face.jpg"), source);
    const controller = new AbortController();
    await assert.rejects(publishMaskAssistCopies("dst", id, { reviewed: true, signal: controller.signal,
      onProgress: () => controller.abort() }), (error) => error.name === "AbortError");
    assert.deepEqual(await readFile(path.join(aligned, "face.jpg")), source);
    let retainedStaging;
    await assert.rejects(publishMaskAssistCopies("dst", id, { reviewed: true,
      runProcess: async () => { throw Object.assign(new Error("controlled binding helper stop unconfirmed"), { code: "HELPER_STOP_UNCONFIRMED" }); },
    }), error => { retainedStaging = error.details?.retainedStaging; return error.code === "HELPER_STOP_UNCONFIRMED" && Boolean(retainedStaging); });
    assert.equal((await lstat(retainedStaging)).isDirectory(), true, "An unconfirmed helper still owns its staging directory");
    await rm(retainedStaging, { recursive: true, force: true });
    const receipt = await publishMaskAssistCopies("dst", id, { reviewed: true });
    assert.equal(receipt.copiedCount, 2);
    assert.equal(receipt.assistedCount, 1);
    assert.equal(receipt.subset, false);
    assert.equal(receipt.reviewedMergeAvailable, false, "unreviewed faces cannot enter strict reviewed mask merge");
    assert.equal(receipt.entries.find(entry => entry.file === "face.jpg").sourceFrame.file, "mask-binding-test.png");
    assert.deepEqual(await readFile(path.join(aligned, "face.jpg")), source);
    assert.deepEqual(await readFile(path.join(published, "face.jpg")), assisted);
    assert.deepEqual(await readFile(path.join(published, "other.jpg")), other);
    await writeFile(path.join(published, "other.jpg"), "edited-copy");
    assert.deepEqual(await readFile(path.join(aligned, "other.jpg")), other);
    const repeated = await publishMaskAssistCopies("dst", id, { reviewed: true });
    assert.equal(repeated.reused, true);
    assert.equal(repeated.facesetPath, receipt.facesetPath);
    assert.equal((await readMaskAssistDraft("dst", id)).publication.status, "published");
    await writeFile(path.join(draft, "copies", "face.jpg"), "changed");
    await assert.rejects(resolveMaskAssistFile("dst", id, 0, "copy"), (error) => error.code === "MASK_DRAFT_CHANGED");
  } finally {
    await rm(aligned, { recursive: true, force: true });
    await rename(original, aligned);
    await rm(saved, { recursive: true, force: true });
    await rm(draft, { recursive: true, force: true });
    await rm(published, { recursive: true, force: true });
    await rm(originalFrame, { force: true });
  }
});
