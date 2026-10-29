import assert from "node:assert/strict";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

test("export inventory omits a pending bundle while preserving resumable model discovery", async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfm-publication-readiness-"));
  assert.equal(path.dirname(root), path.resolve(os.tmpdir()));
  t.after(() => rm(root, { recursive: true, force: true }));
  await cp(path.join(repository, "webui/server"), path.join(root, "webui/server"), { recursive: true });
  await cp(path.join(repository, "webui/shared"), path.join(root, "webui/shared"), { recursive: true });
  await cp(path.join(repository, "release"), path.join(root, "release"), { recursive: true });
  const manager = await import(pathToFileURL(path.join(root, "webui/server/workspace-manager.mjs")));
  const directory = path.join(root, "workspace/model/fixture");
  await mkdir(directory, { recursive: true });
  const checkpoint = "synthetic training checkpoint remains unchanged";
  const metadata = { format: "me-pytorch", version: 1, model_class: "ME", name: "fixture",
    checkpoint: "me.pt", iteration: 100, config: {} };
  await writeFile(path.join(directory, "me.pt"), checkpoint);
  await writeFile(path.join(directory, "metadata.json"), JSON.stringify(metadata));
  const names = ["model.dfm", "model_extreme.dfm"];
  for (const name of names) await writeFile(path.join(directory, name), "standalone graph fixture");
  assert.equal((await manager.inspectExportReadiness()).outputCount, 2);
  for (const name of names) await writeFile(path.join(directory, name + ".publish.json"), "pending or unreadable journal");
  const pending = await manager.inspectExportReadiness();
  assert.equal(pending.outputCount, 0);
  assert.equal(pending.modelCount, 1);
  assert.equal(pending.readyCount, 1);
  assert.equal(pending.models[0].iteration, 100);
  for (const name of names) {
    await rm(path.join(directory, name + ".publish.json"));
    await writeFile(path.join(directory, name + ".publish.lock"), "released OS lock");
  }
  assert.equal((await manager.inspectExportReadiness()).outputCount, 2);
  assert.equal(await readFile(path.join(directory, "me.pt"), "utf8"), checkpoint);
  assert.deepEqual(JSON.parse(await readFile(path.join(directory, "metadata.json"), "utf8")), metadata);
});
