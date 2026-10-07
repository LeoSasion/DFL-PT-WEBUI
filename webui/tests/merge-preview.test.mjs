import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile, readdir, unlink, rmdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { buildCommand, getCommandDefinition, validateCommandParameters } from "../server/command-registry.mjs";
import { inspectMergePreview, mergePreviewDirectory, resolveMergePreviewAsset } from "../server/merge-preview-manager.mjs";
import { PATHS } from "../server/paths.mjs";

test("bounded preview command retains final merge parameters and directs both outputs away from final sequence", () => {
  const params = validateCommandParameters("merge.preview_me", { forceModelName: "qa", frameStart: 2, frameCount: 3, mode: "hist-match", maskMode: 11, colorTransfer: "robust-lab" }, "guided");
  const preview = buildCommand(getCommandDefinition("merge.preview_me"), { jobId: "preview-test-qa", parameters: params, launchMode: "guided" }).launch;
  const { frameStart, frameCount, ...final } = params;
  const full = buildCommand(getCommandDefinition("merge.me"), { parameters: final, launchMode: "guided" }).launch;
  assert.equal(preview.env.DFL_WEB_MERGE_CONFIG, full.env.DFL_WEB_MERGE_CONFIG);
  assert.equal(preview.args[preview.args.indexOf("--preview-frame-count") + 1], "3");
  assert.equal(preview.args[preview.args.indexOf("--preview-frame-start") + 1], "2");
  for (const flag of ["--output-dir", "--output-mask-dir"]) {
    const target = preview.args[preview.args.indexOf(flag) + 1];
    assert.ok(target.startsWith(mergePreviewDirectory("preview-test-qa")));
    assert.notEqual(target, full.args[full.args.indexOf(flag) + 1]);
  }
  assert.throws(() => validateCommandParameters("merge.preview_me", { frameCount: 21 }, "guided"), { code: "PARAMETER_OUT_OF_RANGE" });
  assert.throws(() => mergePreviewDirectory("../escape"));
});

test("preview receipt binds source, aligned, model and results; altered input is rejected", { skip: !process.env.DFLSN_ISOLATED_TEST_ROOT }, async t => {
  const jobId = "preview-fixture-qa";
  const directory = mergePreviewDirectory(jobId);
  const alignedDir = path.join(PATHS.workspaceRoot, "data_dst", "preview-aligned");
  const modelDir = path.join(PATHS.workspaceRoot, "model", "preview-model");
  t.after(async () => {
    // Delete only this test's known files, then its empty owned directories;
    // unrelated fixture media and later tests retain their original inventory.
    for (const file of [path.join(PATHS.workspaceRoot, "data_dst", "preview-source.png"), path.join(alignedDir, "face.jpg"), path.join(modelDir, "me.pt"), path.join(directory, "merged", "preview-source.png"), path.join(directory, "merged_mask", "preview-source.png"), path.join(directory, "merged", "merge.audit.json")]) {
      await unlink(file).catch(error => { if (error.code !== "ENOENT") throw error; });
    }
    for (const dir of [alignedDir, modelDir, path.join(directory, "merged"), path.join(directory, "merged_mask"), directory]) {
      await rmdir(dir).catch(error => { if (error.code !== "ENOENT") throw error; });
    }
  });
  await Promise.all([mkdir(path.join(directory, "merged"), { recursive: true }), mkdir(path.join(directory, "merged_mask"), { recursive: true }), mkdir(alignedDir, { recursive: true }), mkdir(modelDir, { recursive: true })]);
  const hash = data => createHash("sha256").update(data).digest("hex");
  const source = Buffer.from("source fixture"), aligned = Buffer.from("aligned fixture"), model = Buffer.from("model fixture"), merged = Buffer.from("merged fixture"), mask = Buffer.from("mask fixture");
  await Promise.all([writeFile(path.join(PATHS.workspaceRoot, "data_dst", "preview-source.png"), source), writeFile(path.join(alignedDir, "face.jpg"), aligned), writeFile(path.join(modelDir, "me.pt"), model), writeFile(path.join(directory, "merged", "preview-source.png"), merged), writeFile(path.join(directory, "merged_mask", "preview-source.png"), mask)]);
  const frameNames = (await readdir(path.join(PATHS.workspaceRoot, "data_dst"), { withFileTypes: true })).filter(entry => entry.isFile() && /\.(png|jpe?g|tiff?)$/i.test(entry.name)).map(entry => entry.name).sort();
  const audit = { kind: "merge-quality-audit", status: "complete", configuration: { mode: "overlay" }, preview: { kind: "bounded-independent-preview", frameStart: frameNames.indexOf("preview-source.png") + 1, frameCount: 1, totalSourceFrames: frameNames.length, sourceFrameInventorySha256: hash(frameNames.join("\n")), modelName: "preview-model", modelSha256: hash(model), alignedPath: "data_dst/preview-aligned" }, frames: [{ file: "preview-source.png", sourceSha256: hash(source), outputSha256: hash(merged), outputMaskSha256: hash(mask), alignments: [{ file: "face.jpg", sha256: hash(aligned), storage: "plain" }] }] };
  await writeFile(path.join(directory, "merged", "merge.audit.json"), JSON.stringify(audit));
  assert.equal((await inspectMergePreview(jobId)).completeCount, 1);
  assert.deepEqual((await resolveMergePreviewAsset(jobId, "merged", "preview-source.png")).bytes, merged);
  await assert.rejects(resolveMergePreviewAsset(jobId, "merged", "not-member.png"), { code: "MERGE_PREVIEW_MEMBER_MISSING" });
  await writeFile(path.join(alignedDir, "face.jpg"), "changed input");
  await assert.rejects(inspectMergePreview(jobId), { code: "MERGE_PREVIEW_SOURCE_CHANGED" });
});
