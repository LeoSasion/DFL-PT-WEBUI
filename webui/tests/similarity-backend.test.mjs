import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import * as promises from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { mapPythonRuntime, probePythonRuntime } from "./python-runtime.mjs";

const originalFs = { ...promises };
const sourceWebui = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceRepository = path.dirname(sourceWebui);
const python = process.env.DFLSN_TEST_PYTHON || path.join(sourceRepository, ".venv", "Scripts", "python.exe");
let runtime;

test("similarity windows preserve the bounded, read-only Python contract", () => {
  const result = spawnSync(python, ["-B", path.join(sourceWebui, "tests", "similarity-contract.py")], {
    encoding: "utf8", timeout: 30_000,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});

// Load production managers in a private repository. No read or mutation in
// these tests resolves PATHS to the user's workspace or local key settings.
async function fixture(run) {
  const root = await originalFs.mkdtemp(path.join(os.tmpdir(), "dfl-similarity-backend-"));
  const server = path.join(root, "webui", "server");
  const aligned = path.join(root, "workspace", "data_src", "aligned");
  const restores = [];
  const previousBytecode = process.env.PYTHONDONTWRITEBYTECODE;
  process.env.PYTHONDONTWRITEBYTECODE = "1";
  try {
    await originalFs.mkdir(server, { recursive: true });
    await originalFs.mkdir(aligned, { recursive: true });
    for (const name of ["paths.mjs", "environment.mjs", "asset-manager.mjs"]) {
      await originalFs.copyFile(path.join(sourceWebui, "server", name), path.join(server, name));
    }
    await originalFs.cp(path.join(sourceWebui, "python"), path.join(root, "webui", "python"), {
      recursive: true, filter: candidate => path.basename(candidate) !== "__pycache__",
    });
    await originalFs.mkdir(path.join(root, "_internal"), { recursive: true });
    await originalFs.symlink(path.join(sourceRepository, "_internal", "DeepFaceLab"),
      path.join(root, "_internal", "DeepFaceLab"), process.platform === "win32" ? "junction" : "dir");
    runtime ??= probePythonRuntime(python);
    const mappedPython = await mapPythonRuntime(runtime, root);
    const creation = spawnSync(mappedPython, ["-B", "-c", [
      "import cv2, numpy as np, pathlib, sys",
      "root=pathlib.Path(sys.argv[1])",
      "image=np.zeros((32,32,3), dtype=np.uint8)",
      "cv2.rectangle(image,(5,5),(23,25),(40,170,240),-1)",
      "for index in range(4): cv2.imwrite(str(root/f'{index:04d}.png'),image)",
    ].join("\n"), aligned], { encoding: "utf8", timeout: 30_000 });
    assert.equal(creation.status, 0, creation.stderr);
    const manager = await import(pathToFileURL(path.join(server, "asset-manager.mjs")));
    const fault = (method, replacement) => {
      const previous = fs.promises[method];
      fs.promises[method] = (...args) => replacement(previous, ...args);
      syncBuiltinESMExports();
      const restore = () => { fs.promises[method] = previous; syncBuiltinESMExports(); };
      restores.push(restore);
      return restore;
    };
    await run({ root, aligned, manager, fault });
  } finally {
    for (const restore of restores.reverse()) restore();
    if (previousBytecode === undefined) delete process.env.PYTHONDONTWRITEBYTECODE;
    else process.env.PYTHONDONTWRITEBYTECODE = previousBytecode;
    assert.equal(path.dirname(path.resolve(root)).toLowerCase(), path.resolve(os.tmpdir()).toLowerCase());
    assert.match(path.basename(root), /^dfl-similarity-backend-/);
    await originalFs.rm(root, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
  }
}

const reviewOf = result => ({ side: result.side, workspaceKey: result.workspaceKey,
  fingerprint: result.fingerprint, threshold: result.threshold,
  offset: result.offset, compareOffset: result.compareOffset, limit: result.limit });

test("similarity cache includes both windows, exact threshold and current inventory", async () => fixture(async ({ aligned, manager }) => {
  const first = await manager.buildAlignedSimilarityGroups("src", { limit: 2, threshold: 0.8601 });
  assert.equal(first.mode, "batch");
  assert.equal(first.cached, false);
  assert.equal((await manager.buildAlignedSimilarityGroups("src", { limit: 2, threshold: 0.8601 })).cached, true);
  const finer = await manager.buildAlignedSimilarityGroups("src", { limit: 2, threshold: 0.8602 });
  assert.equal(finer.cached, false);
  assert.equal(finer.threshold, 0.8602);
  const tail = await manager.buildAlignedSimilarityGroups("src", { limit: 2, offset: 2 });
  assert.deepEqual(tail.groups[0].members.map(member => member.name).sort(), ["0002.png", "0003.png"]);
  const paired = await manager.buildAlignedSimilarityGroups("src", { limit: 4, compareOffset: 2 });
  assert.equal(paired.mode, "paired");
  assert.deepEqual(paired.windows.map(window => window.offset), [0, 2]);
  assert.equal(paired.groups[0].crossBatch, true);
  await originalFs.writeFile(path.join(aligned, "0001.png"), "changed-external-image");
  const changed = await manager.buildAlignedSimilarityGroups("src", { limit: 2, threshold: 0.8601 });
  assert.equal(changed.cached, false);
  assert.notEqual(changed.fingerprint, first.fingerprint);
  assert.equal(changed.invalidCount, 1);
}));

test("backend rejects malformed windows before decoding", async () => fixture(async ({ manager }) => {
  for (const options of [{ offset: -1 }, { offset: 1.5 }, { compareOffset: "bad" },
    { offset: Number.MAX_SAFE_INTEGER + 1 }, { threshold: NaN }, { limit: false },
    { offset: [0] }, { threshold: {} }]) {
    await assert.rejects(manager.buildAlignedSimilarityGroups("src", options), { code: "SIMILARITY_PARAMETERS_INVALID" });
  }
  await assert.rejects(manager.buildAlignedSimilarityGroups("src", { compareOffset: 249 }),
    { code: "SIMILARITY_WINDOWS_OVERLAP" });
}));

test("inventory changes during analysis invalidate the result and its cache", async () => fixture(async ({ aligned, manager, fault }) => {
  let reads = 0;
  const restore = fault("readdir", async (readdir, directory, ...options) => {
    if (directory === aligned && ++reads === 2) {
      await originalFs.writeFile(path.join(aligned, "0001.png"), "changed-during-analysis");
    }
    return readdir(directory, ...options);
  });
  await assert.rejects(manager.buildAlignedSimilarityGroups("src", { limit: 2 }),
    { code: "SIMILARITY_DATASET_CHANGED", status: 409 });
  restore();
  const current = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  assert.equal(current.cached, false);
  assert.equal(current.invalidCount, 1);
}));

test("reviewed quarantine rejects changed, outside-window and representative selections", async () => fixture(async ({ aligned, manager }) => {
  const result = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  const review = reviewOf(result);
  for (const names of [[result.groups[0].representativeName], ["0003.png"]]) {
    await assert.rejects(manager.quarantineAlignedImages("src", names, { review }),
      { code: "SIMILARITY_SELECTION_INVALID", status: 409 });
  }
  await originalFs.writeFile(path.join(aligned, "0001.png"), "replaced-under-same-name");
  await assert.rejects(manager.quarantineAlignedImages("src", ["0001.png"], { review }),
    { code: "SIMILARITY_DATASET_CHANGED", status: 409 });
  assert.deepEqual((await originalFs.readdir(aligned)).sort(), ["0000.png", "0001.png", "0002.png", "0003.png"]);
}));

test("reviewed candidates share one recovery token and restore without losing the representative", async () => fixture(async ({ aligned, manager }) => {
  const result = await manager.buildAlignedSimilarityGroups("src", { limit: 4, compareOffset: 2 });
  const group = result.groups[0];
  const names = group.members.filter(member => !member.representative).map(member => member.name);
  const original = new Map(await Promise.all(names.map(async name => [name, await originalFs.readFile(path.join(aligned, name))])));
  const quarantined = await manager.quarantineAlignedImages("src", names, { review: reviewOf(result) });
  assert.equal(quarantined.count, 3);
  assert.deepEqual(await originalFs.readdir(aligned), [group.representativeName]);
  for (const name of names) {
    const restored = await manager.restoreAlignedImage("src", quarantined.token, encodeURIComponent(name));
    assert.equal(restored.restored, true);
    assert.deepEqual(await originalFs.readFile(path.join(aligned, name)), original.get(name));
  }
  assert.equal((await originalFs.readdir(aligned)).length, 4);
}));

test("a last-moment change is rejected immediately before quarantine creates a token", async () => fixture(async ({ root, aligned, manager, fault }) => {
  const result = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  let reads = 0;
  fault("readdir", async (readdir, directory, ...options) => {
    if (directory === aligned && ++reads === 4) {
      await originalFs.writeFile(path.join(aligned, "0001.png"), "changed-before-move");
    }
    return readdir(directory, ...options);
  });
  await assert.rejects(manager.quarantineAlignedImages("src", ["0001.png"], { review: reviewOf(result) }),
    { code: "SIMILARITY_DATASET_CHANGED", status: 409 });
  assert.equal((await originalFs.readdir(aligned)).length, 4);
  await assert.rejects(originalFs.stat(path.join(root, "workspace", ".webui", "quarantine")), { code: "ENOENT" });
}));

test("generic quarantine stays compatible and rolls back failed batch moves", async () => fixture(async ({ aligned, manager, fault }) => {
  fault("rename", (rename, source, target) => {
    if (source === path.join(aligned, "0002.png")) throw Object.assign(new Error("injected move failure"), { code: "EACCES" });
    return rename(source, target);
  });
  await assert.rejects(manager.quarantineAlignedImages("src", ["0001.png", "0002.png"]), { code: "EACCES" });
  assert.deepEqual((await originalFs.readdir(aligned)).sort(), ["0000.png", "0001.png", "0002.png", "0003.png"]);
}));

test("rollback conflicts preserve external replacement and return the recovery token", async () => fixture(async ({ root, aligned, manager, fault }) => {
  const original = await originalFs.readFile(path.join(aligned, "0001.png"));
  fault("rename", async (rename, source, target) => {
    if (source === path.join(aligned, "0002.png")) {
      await originalFs.writeFile(path.join(aligned, "0001.png"), "external-new-file");
      throw Object.assign(new Error("injected move failure"), { code: "EACCES" });
    }
    return rename(source, target);
  });
  let failure;
  await assert.rejects(manager.quarantineAlignedImages("src", ["0001.png", "0002.png"]), error => {
    failure = error;
    return error.code === "QUARANTINE_RECOVERY_REQUIRED" && error.details.rollbackErrors[0].code === "RESTORE_CONFLICT";
  });
  assert.equal(await originalFs.readFile(path.join(aligned, "0001.png"), "utf8"), "external-new-file");
  assert.deepEqual(await originalFs.readFile(path.join(root, "workspace", ".webui", "quarantine", "src",
    failure.details.token, "0001.png")), original);
}));

test("a replacement during token creation is rejected before any reviewed image moves", async () => fixture(async ({ root, aligned, manager, fault }) => {
  const analysis = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  const candidate = analysis.groups[0].members.find(member => !member.representative).name;
  let replaced = false;
  fault("mkdir", async (mkdir, target, ...args) => {
    const value = await mkdir(target, ...args);
    if (!replaced && /^\d{14}-[a-f0-9]{10}$/.test(path.basename(target))) {
      replaced = true;
      await originalFs.writeFile(path.join(aligned, candidate), "new-unreviewed-replacement");
    }
    return value;
  });
  await assert.rejects(manager.quarantineAlignedImages("src", [candidate], { review: reviewOf(analysis) }),
    { code: "SIMILARITY_DATASET_CHANGED", status: 409 });
  assert.equal(await originalFs.readFile(path.join(aligned, candidate), "utf8"), "new-unreviewed-replacement");
  assert.equal((await originalFs.readdir(aligned)).length, 4);
  const quarantine = path.join(root, "workspace", ".webui", "quarantine", "src");
  for (const token of await originalFs.readdir(quarantine)) assert.deepEqual(await originalFs.readdir(path.join(quarantine, token)), []);
}));

test("a later candidate replacement is detected after earlier moves and rolled back safely", async () => fixture(async ({ root, aligned, manager, fault }) => {
  const analysis = await manager.buildAlignedSimilarityGroups("src", { limit: 4 });
  const names = analysis.groups[0].members.filter(member => !member.representative).slice(0, 2).map(member => member.name);
  const firstOriginal = await originalFs.readFile(path.join(aligned, names[0]));
  let replaced = false;
  fault("rename", async (rename, source, target) => {
    const value = await rename(source, target);
    if (!replaced && source === path.join(aligned, names[0])) {
      replaced = true;
      const replacement = path.join(root, "replacement-candidate.png");
      await originalFs.writeFile(replacement, "new-second-candidate");
      await originalFs.rename(replacement, path.join(aligned, names[1]));
    }
    return value;
  });
  await assert.rejects(manager.quarantineAlignedImages("src", names, { review: reviewOf(analysis) }),
    { code: "SIMILARITY_DATASET_CHANGED", status: 409 });
  assert.deepEqual(await originalFs.readFile(path.join(aligned, names[0])), firstOriginal);
  assert.equal(await originalFs.readFile(path.join(aligned, names[1]), "utf8"), "new-second-candidate");
  assert.equal((await originalFs.readdir(aligned)).length, 4);
}));

test("post-move replacement is reported with a recovery token without restoring unverified bytes", async () => fixture(async ({ root, aligned, manager, fault }) => {
  const analysis = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  const candidate = analysis.groups[0].members.find(member => !member.representative).name;
  fault("rename", async (rename, source, target) => {
    const value = await rename(source, target);
    if (source === path.join(aligned, candidate)) {
      await originalFs.writeFile(source, "external-same-name-replacement");
      await originalFs.writeFile(target, "external-quarantine-replacement");
    }
    return value;
  });
  let failure;
  await assert.rejects(manager.quarantineAlignedImages("src", [candidate], { review: reviewOf(analysis) }), error => {
    failure = error;
    return error.code === "QUARANTINE_RECOVERY_REQUIRED" && error.details.rollbackErrors[0].code === "QUARANTINE_IMAGE_CHANGED";
  });
  assert.equal(await originalFs.readFile(path.join(aligned, candidate), "utf8"), "external-same-name-replacement");
  assert.equal(await originalFs.readFile(path.join(root, "workspace", ".webui", "quarantine", "src",
    failure.details.token, candidate), "utf8"), "external-quarantine-replacement");
}));

test("a SRC review cannot quarantine an identical hard-linked DST inventory", async () => fixture(async ({ root, aligned, manager }) => {
  const dst = path.join(root, "workspace", "data_dst", "aligned");
  await originalFs.mkdir(dst, { recursive: true });
  for (const name of await originalFs.readdir(aligned)) await originalFs.link(path.join(aligned, name), path.join(dst, name));
  const analysis = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  assert.equal((await manager.roleDatasetSnapshot("dst")).fingerprint, analysis.fingerprint);
  const candidate = analysis.groups[0].members.find(member => !member.representative).name;
  await assert.rejects(manager.quarantineAlignedImages("dst", [candidate], { review: reviewOf(analysis) }),
    { code: "SIMILARITY_REVIEW_INVALID", status: 409 });
  assert.equal((await originalFs.readdir(aligned)).length, 4);
  assert.equal((await originalFs.readdir(dst)).length, 4);
}));

test("similarity review requires the originating workspace and complete dataset scope", async () => fixture(async ({ root, aligned, manager }) => {
  const analysis = await manager.buildAlignedSimilarityGroups("src", { limit: 2 });
  assert.equal(analysis.workspaceKey, path.join(root, "workspace"));
  const candidate = analysis.groups[0].members.find(member => !member.representative).name;
  for (const change of [{ workspaceKey: path.join(root, "other-workspace") }, { workspaceKey: undefined }, { side: undefined }]) {
    await assert.rejects(manager.quarantineAlignedImages("src", [candidate], { review: { ...reviewOf(analysis), ...change } }),
      { code: "SIMILARITY_REVIEW_INVALID", status: 409 });
  }
  assert.equal((await originalFs.readdir(aligned)).length, 4);
}));
