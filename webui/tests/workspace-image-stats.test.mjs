import assert from "node:assert/strict";
import { cp, mkdir, mkdtemp, readFile, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { getWorkflowArtifactState } from "../src/domain/workflow-readiness.js";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const slots = {
  srcFrames: "data_src", dstFrames: "data_dst",
  srcFaces: "data_src/aligned", dstFaces: "data_dst/aligned",
  merged: "data_dst/merged", mergedMask: "data_dst/merged_mask",
};
// Actual encoded 2x2 RGB images. BMP is deliberately outside core.pathex's
// get_image_paths format list, even though it is a valid image container.
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkCVjAwMDAxAAGAAtmAQDbRngkAAAAAElFTkSuQmCC", "base64");
const jpeg = Buffer.from("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDhKKKK+tPmz//Z", "base64");
const tiff = Buffer.from("SUkqAAgAAAAKAAABBAABAAAAAgAAAAEBBAABAAAAAgAAAAIBAwADAAAAhgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAjAAAABUBAwABAAAAAwAAABYBBAABAAAAAgAAABcBBAABAAAADAAAABwBAwABAAAAAQAAAAAAAAAIAAgACAAMUKAMUKAMUKAMUKA=", "base64");
const bmp = Buffer.from("Qk1GAAAAAAAAADYAAAAoAAAAAgAAAAIAAAABABgAAAAAABAAAADEDgAAxA4AAAAAAAAAAAAAoFAMoFAMAACgUAygUAwAAA==", "base64");
const images = { "00001.JPG": jpeg, "00002.jpeg": jpeg, "00003.png": png, "00004.tif": tiff, "00005.TIFF": tiff };

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfl-workspace-image-stats-"));
  assert.equal(path.dirname(root), path.resolve(os.tmpdir()));
  assert.ok(path.basename(root).startsWith("dfl-workspace-image-stats-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await cp(path.join(repository, "webui/server"), path.join(root, "webui/server"), { recursive: true });
  await cp(path.join(repository, "release"), path.join(root, "release"), { recursive: true });
  const workspace = path.join(root, "workspace");
  await Promise.all(Object.values(slots).map(slot => mkdir(path.join(workspace, slot), { recursive: true })));
  const manager = await import(pathToFileURL(path.join(root, "webui/server/workspace-manager.mjs")));
  return { workspace, inspect: manager.inspectWorkspace };
}

test("workspace image statistics ignore merge audits and dataset sidecars across all six slots", async (t) => {
  const { workspace, inspect } = await fixture(t);
  const imageDate = new Date("2026-01-01T00:00:00.000Z");
  const helperDate = new Date("2026-02-01T00:00:00.000Z");
  const audit = JSON.stringify({ schemaVersion: 1, frames: 5, completed: true });
  for (const slot of Object.values(slots)) {
    const directory = path.join(workspace, slot);
    for (const [name, bytes] of Object.entries(images)) {
      const target = path.join(directory, name);
      await writeFile(target, bytes);
      await utimes(target, imageDate, imageDate);
    }
    for (const [name, bytes] of Object.entries({
      "merge.audit.json": audit, "metadata.json": "{}", "sample.npy": Buffer.alloc(32),
      "sequence.media.json": "{}", "notes.txt": "sidecar", "unsupported.bmp": bmp,
      "frame.png.tmp": png, ".hidden.png": png,
    })) {
      const target = path.join(directory, name);
      await writeFile(target, bytes);
      await utimes(target, helperDate, helperDate);
    }
    const nested = path.join(directory, "nested");
    await mkdir(nested);
    await writeFile(path.join(nested, "nested.png"), png);
  }
  const result = await inspect();
  const expectedBytes = Object.values(images).reduce((sum, bytes) => sum + bytes.length, 0);
  for (const key of Object.keys(slots)) {
    assert.deepEqual(result.datasets[key], { count: 5, bytes: expectedBytes, modifiedAt: imageDate.toISOString() }, key);
  }
  assert.equal(getWorkflowArtifactState(result).exportConfiguredReady, true);
  assert.equal(getWorkflowArtifactState(result).partialMerge, false);
  assert.equal(await readFile(path.join(workspace, "data_dst/merged/merge.audit.json"), "utf8"), audit);
});

test("auxiliary-only datasets stay unready without changing ME model discovery", async (t) => {
  const { workspace, inspect } = await fixture(t);
  for (const slot of Object.values(slots)) {
    await writeFile(path.join(workspace, slot, "merge.audit.json"), "{}");
    await writeFile(path.join(workspace, slot, "landmarks.npy"), Buffer.alloc(32));
  }
  const model = path.join(workspace, "model", "fixture-me");
  await mkdir(model, { recursive: true });
  await writeFile(path.join(model, "me.pt"), "nonempty checkpoint discovery fixture");
  await writeFile(path.join(model, "metadata.json"), JSON.stringify({ format: "me-pytorch", version: 1,
    model_class: "ME", name: "fixture-me", checkpoint: "me.pt", iteration: 1, config: {} }));
  await writeFile(path.join(model, "notes.json"), "{}");
  const result = await inspect();
  for (const key of Object.keys(slots)) assert.deepEqual(result.datasets[key], { count: 0, bytes: 0, modifiedAt: null }, key);
  assert.equal(result.readiness.frames, false);
  assert.equal(result.readiness.faces, false);
  assert.equal(result.readiness.merged, false);
  assert.equal(getWorkflowArtifactState(result).exportConfiguredReady, false);
  assert.equal(result.readiness.me, true);
  assert.equal(result.models[0].name, "fixture-me");
  assert.equal(result.models[0].fileCount, 3);
});
