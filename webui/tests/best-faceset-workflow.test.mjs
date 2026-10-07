import assert from "node:assert/strict";
import test from "node:test";
import { bestFacesetDraftState, bestFacesetHistoryLabel, bestFacesetPreviewKey, bestFacesetReasonText, bestFacesetPublicationPreviewMatches, createBestFacesetDraftWriter, registerBestFacesetDraftWriter, flushBestFacesetDraft } from "../src/domain/best-faceset-workflow.js";
import { UX_EN } from "../src/ux-translations.mjs";

test("restoring a changed source retains the draft choices but never restores identity confirmation", () => {
  const draft = { planId: "plan-a", identityReferences: ["front.jpg"], confirmed: true, offset: 125, category: "review", targetCount: 50, qualityModel: "foreground_tenengrad" };
  const valid = bestFacesetDraftState({ draft, sourceValid: true, identityConfirmationValid: true, sourceFingerprint: "current" });
  assert.equal(valid.confirmed, true); assert.equal(valid.offset, 120);
  for (const flags of [{ sourceValid: false, identityConfirmationValid: true }, { sourceValid: true, identityConfirmationValid: false }]) {
    const restored = bestFacesetDraftState({ draft, ...flags, sourceFingerprint: "current" });
    assert.equal(restored.confirmed, false); assert.deepEqual(restored.identityReferences, ["front.jpg"]);
  }
});

test("draft autosave serializes CAS revisions and saves the newest pending edit without restoring an old response", async () => {
  let resolveFirst; const calls = [], saved = [];
  const writer = createBestFacesetDraftWriter({ revision: 4, save: async value => {
    calls.push(value); if (calls.length === 1) await new Promise(resolve => { resolveFirst = resolve; });
    return { revision: value.expectedRevision + 1 };
  }, onSaved: value => saved.push(value.revision) });
  writer.enqueue({ identityReferences: ["a.jpg"] }); await Promise.resolve();
  writer.enqueue({ identityReferences: ["a.jpg", "b.jpg"] });
  writer.enqueue({ identityReferences: ["a.jpg", "b.jpg", "c.jpg"] });
  resolveFirst(); await writer.flush();
  assert.deepEqual(calls.map(item => item.expectedRevision), [4, 5]);
  assert.deepEqual(calls[1].draft.identityReferences, ["a.jpg", "b.jpg", "c.jpg"]);
  assert.deepEqual(saved, [5, 6]);
  writer.enqueue(calls[1].draft); await writer.flush();
  writer.enqueue({ identityReferences: [] }); await writer.flush();
  assert.equal(calls.length, 3); assert.equal(calls[2].expectedRevision, 6);
});

test("a conflicting draft write pauses queued edits instead of replacing another window's choices", async () => {
  const error = Object.assign(new Error("conflict"), { status: 409 }); let writes = 0, seen;
  const writer = createBestFacesetDraftWriter({ save: async () => { writes++; throw error; }, onError: value => { seen = value; } });
  writer.enqueue({ confirmed: false }); await writer.flush();
  writer.enqueue({ confirmed: true }); await writer.flush();
  assert.equal(writes, 1); assert.equal(writer.paused, true); assert.equal(seen, error);
});

test("equivalent numeric cap edits finish the saving state without another write or consuming a CAS revision", async () => {
  const calls = [], saved = []; let status = "saved";
  const writer = createBestFacesetDraftWriter({ revision: 10, save: async payload => {
    calls.push(payload); return { revision: payload.expectedRevision + 1, sourceValid: true, identityConfirmationValid: false, draft: payload.draft };
  }, onSaved: value => { status = "saved"; saved.push(value); } });
  const submitCap = async value => {
    status = "saving";
    writer.enqueue({ targetCount: Number(value), confirmed: false, identityReferences: [] });
    await writer.flush(); assert.equal(status, "saved");
  };
  await submitCap("2000"); await submitCap("02000"); await submitCap("2e3");
  assert.equal(calls.length, 1); assert.deepEqual(saved.map(value => value.revision), [11, 11, 11]);
  assert.equal(saved[1].unchanged, true); assert.equal(saved[2].unchanged, true);
  assert.equal(saved[2].sourceValid, true); assert.equal(saved[2].identityConfirmationValid, false);
  await submitCap("2001");
  assert.deepEqual(calls.map(value => value.expectedRevision), [10, 11]);
  assert.equal(saved.at(-1).revision, 12);
});

test("returning to the same project and side waits for pending writes before reading the server draft", async () => {
  let release, complete = false;
  const writer = createBestFacesetDraftWriter({ save: async () => { await new Promise(resolve => { release = resolve; }); complete = true; return { revision: 1 }; } });
  registerBestFacesetDraftWriter("project:src", writer);
  writer.enqueue({ identityReferences: ["front.jpg"] }); await Promise.resolve();
  const returning = flushBestFacesetDraft("project:src");
  await flushBestFacesetDraft("other-project:src"); assert.equal(complete, false);
  release(); await returning; assert.equal(complete, true);
});

test("preview shortcuts are scoped away from editable fields and modified key chords", () => {
  const key = (value, options = {}) => bestFacesetPreviewKey({ key: value, target: { closest: () => null }, ...options });
  assert.equal(key("ArrowLeft"), "previous"); assert.equal(key("ArrowRight"), "next");
  assert.equal(key("k"), "keep"); assert.equal(key("X"), "exclude"); assert.equal(key("s"), "defer");
  assert.equal(key("z", { ctrlKey: true }), "undo");
  assert.equal(key("k", { ctrlKey: true }), null);
  assert.equal(key("ArrowRight", { target: { closest: () => ({}) } }), null);
});

test("history labels describe creation time, input and retained counts and revision type", () => {
  assert.equal(bestFacesetHistoryLabel({ createdAt: "2026-10-07", total: 12000, counts: { selected: 4000 }, state: "published", parentPlanId: "parent" },
    { formatDate: value => value }), "2026-10-07 · 人工修订 · 12000 → 4000 · 已发布");
});

test("localized selection reasons preserve numeric quality, signed geometry and rare-group counts", () => {
  const en = (key, values = {}) => (UX_EN[key] ?? key).replace(/\{(\w+)\}/g, (_, name) => values[name] ?? `{${name}}`);
  assert.equal(bestFacesetReasonText("质量 91.0 达保底 5.0", en), "Quality 91.0; required floor 5.0");
  assert.equal(bestFacesetReasonText("头部偏转 -32° / 张嘴 / 单眼闭合（几何估计）", en), "Head rotation -32° / Mouth open / One eye closed (geometry estimate)");
  assert.equal(bestFacesetReasonText("非重复代表；同类候选 2 张（稀缺姿态或眼口状态）", en), "Distinct representative; 2 candidates in this group (rare pose or eye/mouth state)");
  assert.equal(bestFacesetReasonText("unchanged diagnostic path E:/fixture.jpg", en), "unchanged diagnostic path E:/fixture.jpg");
});

test("a publication preview cannot authorize a different plan or an externally edited manual revision", () => {
  const plan = { planId: "plan-a", reviewRevision: 6 }, preview = { clientPlanId: "plan-a", clientReviewRevision: 6 };
  assert.equal(bestFacesetPublicationPreviewMatches(plan, preview), true);
  assert.equal(bestFacesetPublicationPreviewMatches({ ...plan, reviewRevision: 7 }, preview), false);
  assert.equal(bestFacesetPublicationPreviewMatches({ ...plan, planId: "plan-b" }, preview), false);
  assert.equal(bestFacesetPublicationPreviewMatches(plan, null), false);
});
