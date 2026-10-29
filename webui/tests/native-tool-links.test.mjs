import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { nativeToolDestination } from "../src/domain/native-tool-links.js";
import { PATHS } from "../server/paths.mjs";
import { buildDflEnvironment } from "../server/environment.mjs";

test("native tool links accept only explicit sides and fixed aligned/similarity destinations", () => {
  for (const side of ["src", "dst"]) {
    assert.deepEqual(nativeToolDestination(`#/aligned/${side}`), { nav: side });
    assert.deepEqual(nativeToolDestination(`#/similarity/${side}`), { nav: "tools", focus: { tab: "audit", mode: "similarity", side } });
  }
  for (const hash of ["", "#/aligned/../dst", "#/aligned/other", "#/similarity/src?command=anything", "javascript:alert(1)", "#/aligned/src/extra"]) {
    assert.equal(nativeToolDestination(hash), null);
  }
});

test("historical profile names share one canonical PyTorch root and executable", () => {
  const current = buildDflEnvironment("current");
  for (const profile of ["legacy", "pytorch"]) assert.deepEqual(buildDflEnvironment(profile), current);
  assert.equal(PATHS.dflRoot, PATHS.currentDflRoot); assert.equal(PATHS.dflRoot, PATHS.legacyDflRoot);
  assert.equal(PATHS.dflMain, PATHS.currentMain); assert.equal(PATHS.dflMain, PATHS.legacyMain);
  assert.throws(() => buildDflEnvironment("tensorflow"));
});

test("traditional tool checks resolve the registered project without starting apps or creating configuration", { skip: process.platform !== "win32" }, async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfl-native-tool-"));
  t.after(async () => { assert.equal(path.dirname(path.resolve(root)), path.resolve(os.tmpdir())); await rm(root, { recursive: true, force: true }); });
  for (const folder of ["legacy-cli", "launcher", "webui/.runtime", ".venv/Scripts"]) await mkdir(path.join(root, folder), { recursive: true });
  await copyFile(path.join(PATHS.repositoryRoot, "legacy-cli/open-native-tool.ps1"), path.join(root, "legacy-cli/open-native-tool.ps1"));
  await copyFile(path.join(PATHS.repositoryRoot, "launcher/resolve-active-project.py"), path.join(root, "launcher/resolve-active-project.py"));
  // Reuse the same venv mapping utility as the isolated suite, without copying packages.
  const { mapPythonRuntime, probePythonRuntime } = await import("./python-runtime.mjs");
  await mapPythonRuntime(probePythonRuntime(process.env.DFLSN_TEST_PYTHON ?? PATHS.python), root);
  const registry = path.join(root, "webui/.runtime/projects.json");
  await writeFile(registry, JSON.stringify({ activeId: "interview", projects: [{ id: "interview", name: "访谈" }] }));
  for (const [view, side] of [["aligned", "src"], ["aligned", "dst"], ["similarity", "src"]]) {
    const result = spawnSync("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-File", path.join(root, "legacy-cli/open-native-tool.ps1"), "-View", view, "-Side", side, "-Check"], { encoding: "utf8", windowsHide: true, timeout: 15_000 });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout); assert.equal(parsed.projectId, "interview");
    assert.equal(parsed.url, `http://127.0.0.1:4173/#/${view}/${side}`);
  }
  await writeFile(registry, "{broken");
  const failed = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-File", path.join(root, "legacy-cli/open-native-tool.ps1"), "-View", "aligned", "-Check"], { encoding: "utf8", windowsHide: true });
  assert.notEqual(failed.status, 0); assert.equal(await readFile(registry, "utf8"), "{broken");
});
