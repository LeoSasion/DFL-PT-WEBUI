import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { PATHS } from "../server/paths.mjs";
import { restoreFrameArchive } from "../server/video-tool-manager.mjs";

test("frame archive recovery moves its source timing manifest with the images", {
  skip: !process.env.DFLSN_ISOLATED_TEST_ROOT,
}, async () => {
  assert.ok(path.resolve(PATHS.workspaceRoot).startsWith(path.resolve(process.env.DFLSN_ISOLATED_TEST_ROOT) + path.sep));
  const frameRoot = path.join(PATHS.workspaceRoot, "data_dst");
  const saved = await mkdtemp(path.join(PATHS.workspaceRoot, ".timeline-test-original-"));
  const original = path.join(saved, "data_dst");
  await rename(frameRoot, original);
  const token = "20261006000000-123456789a";
  const archive = path.join(PATHS.archiveRoot, "frames", "dst", token);
  const brokenToken = "20261006000000-123456789b";
  const brokenArchive = path.join(PATHS.archiveRoot, "frames", "dst", brokenToken);
  let undo;
  try {
    await mkdir(frameRoot, { recursive: true });
    await mkdir(archive, { recursive: true });
    await writeFile(path.join(frameRoot, "current.png"), "current-image");
    await writeFile(path.join(frameRoot, "frames.timeline.json"), JSON.stringify({ pts: [10, 20] }));
    await writeFile(path.join(archive, "original.png"), "original-image");
    await writeFile(path.join(archive, "frames.timeline.json"), JSON.stringify({ pts: [40, 160] }));
    await mkdir(brokenArchive, { recursive: true });
    await writeFile(path.join(brokenArchive, "frames.timeline.json"), JSON.stringify({ pts: [40, 160] }));
    await assert.rejects(restoreFrameArchive("dst", brokenToken), (error) => error.code === "FRAME_ARCHIVE_EMPTY");
    assert.equal(await readFile(path.join(frameRoot, "current.png"), "utf8"), "current-image");
    const result = await restoreFrameArchive("dst", token);
    undo = path.join(PATHS.archiveRoot, "frames", "dst", result.undoToken);
    assert.equal(result.restoredFrameCount, 1);
    assert.equal(await readFile(path.join(frameRoot, "original.png"), "utf8"), "original-image");
    assert.deepEqual(JSON.parse(await readFile(path.join(frameRoot, "frames.timeline.json"), "utf8")), { pts: [40, 160] });
    assert.deepEqual(JSON.parse(await readFile(path.join(undo, "frames.timeline.json"), "utf8")), { pts: [10, 20] });
    await restoreFrameArchive("dst", result.undoToken);
    assert.equal(await readFile(path.join(frameRoot, "current.png"), "utf8"), "current-image");
    assert.deepEqual(JSON.parse(await readFile(path.join(frameRoot, "frames.timeline.json"), "utf8")), { pts: [10, 20] });
  } finally {
    await rm(frameRoot, { recursive: true, force: true });
    await rename(original, frameRoot);
    await rm(saved, { recursive: true, force: true });
    await rm(archive, { recursive: true, force: true });
    await rm(brokenArchive, { recursive: true, force: true });
    if (undo) await rm(undo, { recursive: true, force: true });
  }
});
