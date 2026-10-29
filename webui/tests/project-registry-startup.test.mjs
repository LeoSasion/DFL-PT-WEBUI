import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { ProjectManager } from "../server/project-manager.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfl-project-startup-"));
  t.after(async () => {
    assert.equal(path.dirname(path.resolve(root)), path.resolve(os.tmpdir()));
    await rm(root, { recursive: true, force: true });
  });
  for (const folder of ["server", "shared", ".runtime"]) await mkdir(path.join(root, "webui", folder), { recursive: true });
  for (const name of ["server/paths.mjs", "server/atomic-json.mjs", "shared/project-identity.mjs"]) {
    await copyFile(path.join(repository, "webui", name), path.join(root, "webui", name));
  }
  return { root, registry: path.join(root, "webui/.runtime/projects.json"),
    paths: pathToFileURL(path.join(root, "webui/server/paths.mjs")).href };
}

test("startup and project manager reject the same corrupt or unregistered identities without writing", async t => {
  const { root, registry, paths } = await fixture(t);
  const manager = new ProjectManager({ registryFile: registry });
  const invalid = ["{broken", { activeId: "missing", projects: [{ id: "default" }] },
    { activeId: "../outside", projects: [{ id: "default" }] },
    { activeId: "default", projects: [{ id: "default" }, { id: "default" }] },
    { activeId: "default", projects: [{ id: "default" }, null] }];
  const python = process.env.DFLSN_TEST_PYTHON ?? path.join(repository, ".venv/Scripts/python.exe");
  for (const [index, value] of invalid.entries()) {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    await writeFile(registry, text);
    await assert.rejects(import(`${paths}?case=${index}`), { code: "PROJECT_REGISTRY_UNREADABLE" });
    await assert.rejects(manager.readRegistry(), { code: "PROJECT_REGISTRY_UNREADABLE" });
    const resolved = spawnSync(python, ["-B", path.join(repository, "launcher/resolve-active-project.py"), "--root", root], { encoding: "utf8" });
    assert.equal(resolved.status, 2, resolved.stderr);
    assert.equal(await readFile(registry, "utf8"), text);
  }
});

test("first launch defaults only for a missing registry; valid BOM registries retain their registered identity", async t => {
  const { registry, paths } = await fixture(t);
  assert.equal((await import(`${paths}?missing`)).PATHS.activeProject.id, "default");
  await writeFile(registry, '\uFEFF' + JSON.stringify({ activeId: "interview", projects: [{ id: "interview", name: "访谈" }] }));
  const current = (await import(`${paths}?valid`)).PATHS;
  assert.equal(current.activeProject.id, "interview");
  assert.equal(path.basename(current.workspaceRoot), "interview");
});

test("BAT propagates failed environment initialization before finding Node or starting a manager", { skip: process.platform !== "win32" }, async t => {
  const { root } = await fixture(t);
  const directory = path.join(root, "中文 entry"); await mkdir(path.join(directory, "_internal"), { recursive: true });
  const entry = path.join(directory, "启动 WebUI.bat");
  await copyFile(path.join(repository, "启动 WebUI.bat"), entry);
  await writeFile(path.join(directory, "_internal/setenv.bat"), "@exit /b 23\r\n");
  const result = spawnSync(process.env.ComSpec ?? "cmd.exe", ["/d", "/s", "/c", `""${entry}" start"`], {
    cwd: os.tmpdir(), encoding: "utf8", windowsHide: true, windowsVerbatimArguments: true, timeout: 10_000,
  });
  assert.equal(result.status, 23, result.stdout + result.stderr);
});

test("a redirected workspaces ancestor is rejected even when the selected project does not exist yet", async t => {
  const { root, registry, paths } = await fixture(t);
  const outside = await mkdtemp(path.join(os.tmpdir(), "dfl-project-outside-"));
  t.after(async () => { assert.equal(path.dirname(path.resolve(outside)), path.resolve(os.tmpdir())); await rm(outside, { recursive: true, force: true }); });
  await symlink(outside, path.join(root, "workspaces"), process.platform === "win32" ? "junction" : "dir");
  await writeFile(registry, JSON.stringify({ activeId: "interview", projects: [{ id: "interview" }] }));
  await assert.rejects(import(`${paths}?redirected`), { code: "PROJECT_WORKSPACE_UNSAFE" });
});
