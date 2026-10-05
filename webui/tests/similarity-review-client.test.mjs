import assert from "node:assert/strict";
import test from "node:test";
import { runtimeApi } from "../src/runtime/api.js";
import { clearSimilarityReview, pendingSimilarityReview, rememberSimilarityReview, reviewedSimilarityNames, similarityReceiptMatches, similarityResultMatches, similarityReviewFor, similarityScopeKey } from "../src/domain/similarity-review.js";

const scope = { threshold: 0.86, offset: 500, compareOffset: null, limit: 500 };
const fingerprint = "a".repeat(64);
const workspaceKey = "test-workspace";
const result = {
  ...scope, side: "src", workspaceKey, fingerprint,
  groups: [{ representativeName: "keep.jpg", members: [
    { name: "keep.jpg", representative: true },
    { name: "candidate.jpg", representative: false },
  ] }],
};

test("review requires the exact side, threshold, windows and budget of the visible analysis", () => {
  assert.deepEqual(similarityReviewFor(result, "src", scope, workspaceKey), { ...scope, fingerprint, side: "src", workspaceKey });
  assert.equal(similarityReviewFor(result, "dst", scope, workspaceKey), null);
  assert.equal(similarityReviewFor(result, "src", scope, "other-workspace"), null);
  assert.equal(similarityReviewFor(result, "src", scope), null);
  for (const changed of [
    { threshold: 0.87 }, { offset: 0 }, { compareOffset: 0 }, { limit: 250 },
  ]) assert.equal(similarityReviewFor(result, "src", { ...scope, ...changed }, workspaceKey), null);
});

test("unverified and missing results cannot authorize quarantine", () => {
  assert.equal(similarityReviewFor(null, "src", scope, workspaceKey), null);
  for (const invalid of [null, "", "a".repeat(63), "g".repeat(64)]) {
    assert.equal(similarityReviewFor({ ...result, fingerprint: invalid }, "src", scope, workspaceKey), null);
  }
});

test("precise scopes never reuse a rounded-threshold or differently ordered paired result", () => {
  assert.notEqual(similarityScopeKey("src", { ...scope, threshold: 0.8601 }), similarityScopeKey("src", { ...scope, threshold: 0.8602 }));
  assert.notEqual(similarityScopeKey("src", { ...scope, offset: 0, compareOffset: 250 }), similarityScopeKey("src", { ...scope, offset: 250, compareOffset: 0 }));
});

test("selection keeps only reviewed non-representatives, deduplicated in selected order", () => {
  assert.deepEqual(reviewedSimilarityNames(result, ["candidate.jpg", "keep.jpg", "outside.jpg", "candidate.jpg"]), ["candidate.jpg"]);
  assert.deepEqual(reviewedSimilarityNames(null, ["candidate.jpg"]), []);
  assert.deepEqual(reviewedSimilarityNames({ groups: [{ members: null }] }, ["candidate.jpg"]), []);
  const contradictory = { groups: [
    ...result.groups,
    { representativeName: "candidate.jpg", members: [{ name: "candidate.jpg", representative: true }] },
  ] };
  assert.deepEqual(reviewedSimilarityNames(contradictory, ["candidate.jpg"]), []);
});

test("incomplete or inconsistent analysis responses cannot enter the review UI", () => {
  const complete = { ...result, mode: "batch", windowSize: 500, total: 503, selectedCount: 3,
    analyzedCount: 3, invalidCount: 0, groupCount: 1,
    windows: [{ batch: 0, offset: 500, count: 3, analyzedCount: 3, invalidCount: 0, start: 501, end: 503 }],
    groups: [{ ...result.groups[0], id: "similar-001", memberCount: 2, minimumScore: 0.96,
      members: result.groups[0].members.map(member => ({ ...member, batch: 0, score: 0.96,
        imageUrl: `/api/assets/src/aligned/${encodeURIComponent(member.name)}` })) }],
  };
  assert.equal(similarityResultMatches(complete, "src", scope, workspaceKey), true);
  for (const bad of [null, {}, result, { ...complete, windows: null }, { ...complete, groups: null },
    { ...complete, analyzedCount: NaN }, { ...complete, selectedCount: 501 },
    { ...complete, groups: [{ ...complete.groups[0], members: null }] },
    { ...complete, groups: [{ ...complete.groups[0], members: complete.groups[0].members.map(member => ({ ...member, imageUrl: "https://unexpected.example/image" })) }] },
  ]) assert.equal(similarityResultMatches(bad, "src", scope, workspaceKey), false);
});

function mockFetch(t, callback) {
  const original = globalThis.fetch;
  t.after(() => { globalThis.fetch = original; });
  globalThis.fetch = async (url, options) => ({
    ok: true, status: 200,
    json: async () => ({ ok: true, data: callback(url, options) }),
  });
}

test("similarity API forwards both window offsets through the fixed operation contract", async t => {
  const paired = { threshold: 0.86, offset: 500, compareOffset: 0, limit: 500 };
  mockFetch(t, (url, options) => {
    assert.equal(url, "/api/operations");
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body), { kind: "similarity", side: "dst", parameters: { refresh: false, ...paired } });
    return { id: "op-similarity", status: "succeeded", result: { total: 503 } };
  });
  assert.deepEqual(await runtimeApi.alignedSimilarity("dst", paired), { total: 503 });
});

test("quarantine sends the exact review snapshot; generic callers remain compatible", async t => {
  const review = { ...scope, fingerprint, side: "src", workspaceKey };
  const bodies = [];
  mockFetch(t, (url, options) => {
    assert.equal(url, "/api/assets/src/aligned/quarantine-batch");
    assert.equal(options.method, "POST");
    bodies.push(JSON.parse(options.body));
    return { count: 1 };
  });
  await runtimeApi.quarantineAlignedBatch("src", ["candidate.jpg"], review);
  await runtimeApi.quarantineAlignedBatch("src", ["other.jpg"]);
  assert.deepEqual(bodies, [{ names: ["candidate.jpg"], review }, { names: ["other.jpg"] }]);
});

test("pending write review survives navigation and is isolated by project and side", () => {
  const action = { workspaceKey, side: "src", names: ["candidate.jpg"] };
  rememberSimilarityReview(action);
  try {
    assert.equal(pendingSimilarityReview(workspaceKey, "src"), action);
    assert.equal(pendingSimilarityReview(workspaceKey, "dst"), null);
    assert.equal(pendingSimilarityReview("other-workspace", "src"), null);
    clearSimilarityReview({ ...action });
    assert.equal(pendingSimilarityReview(workspaceKey, "src"), action);
    assert.throws(() => rememberSimilarityReview({ ...action }), /尚未确认/);
    assert.equal(pendingSimilarityReview(workspaceKey, "src"), action);
  } finally { clearSimilarityReview(action); }
  assert.equal(pendingSimilarityReview(workspaceKey, "src"), null);
});

test("quarantine acknowledgements must account for every reviewed file before clearing a pending write", () => {
  const action = { side: "src", names: ["first.jpg", "second.jpg"] };
  const receipt = { side: "src", names: ["second.jpg", "first.jpg"], count: 2,
    token: "20261005123456-a012345678", recoverable: true };
  assert.equal(similarityReceiptMatches(receipt, action), true);
  for (const bad of [null, {}, { ...receipt, side: "dst" }, { ...receipt, count: 1 },
    { ...receipt, names: ["first.jpg", "first.jpg"] }, { ...receipt, token: null },
    { ...receipt, names: ["first.jpg", "outside.jpg"] }]) {
    assert.equal(similarityReceiptMatches(bad, action), false);
  }
});

test("a lost quarantine response times out after one write without resubmitting", async t => {
  const original = globalThis.fetch;
  t.after(() => { globalThis.fetch = original; });
  let writes = 0;
  globalThis.fetch = () => { writes++; return new Promise(() => {}); };
  await assert.rejects(runtimeApi.quarantineAlignedBatch("src", ["candidate.jpg"], { ...scope, fingerprint }, { requestTimeoutMs: 15 }), error => error.code === "RUNTIME_REQUEST_TIMEOUT");
  assert.equal(writes, 1);
});

test("result checking is a bounded read and never a repeat quarantine", async t => {
  mockFetch(t, (url, options) => {
    assert.equal(url, "/api/assets/src/similarity-review-state");
    assert.equal(options.method, undefined);
    return { side: "src", workspaceKey, names: [], fingerprint };
  });
  assert.deepEqual(await runtimeApi.similarityReviewState("src"), { side: "src", workspaceKey, names: [], fingerprint });
});
