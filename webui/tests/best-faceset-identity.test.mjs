import assert from "node:assert/strict";
import test from "node:test";
import { bestFacesetIdentityContract } from "../src/domain/best-faceset-identity.js";

const selectedForA = { identityReferences: ["person-a/front.jpg", "person-a/profile.jpg"], state: "finalized",
  qualityModel: "foreground_tenengrad", selection: { counts: { selected: 3, review: 1, rejected: 0 } } };

test("changing confirmed person A references to person B blocks both publication stages until the selection is regenerated", () => {
  let draft = new Set(selectedForA.identityReferences), confirmed = true;
  assert.equal(bestFacesetIdentityContract(selectedForA, draft, confirmed).canPublish, true);
  draft = new Set(["person-b/front.jpg"]); confirmed = false;
  assert.equal(bestFacesetIdentityContract(selectedForA, draft, confirmed).dirty, true);
  assert.equal(bestFacesetIdentityContract(selectedForA, draft, confirmed).canPublish, false);
  confirmed = true;
  const stale = bestFacesetIdentityContract(selectedForA, draft, confirmed);
  assert.deepEqual(stale.expectedReferences, ["person-b/front.jpg"]); assert.equal(stale.dirty, true); assert.equal(stale.canPublish, false);
  const regeneratedForB = { ...selectedForA, identityReferences: [...draft] };
  const fresh = bestFacesetIdentityContract(regeneratedForB, draft, confirmed);
  assert.equal(fresh.dirty, false); assert.equal(fresh.canPublish, true);
});

test("revoking identity confirmation invalidates an existing selection even without editing the checked images", () => {
  const revoked = bestFacesetIdentityContract(selectedForA, new Set(selectedForA.identityReferences), false);
  assert.deepEqual(revoked.expectedReferences, []); assert.equal(revoked.dirty, true); assert.equal(revoked.canPublish, false);
});

test("reference order does not change the identity contract, and an unconfirmed review-only selection can remain publishable", () => {
  assert.equal(bestFacesetIdentityContract(selectedForA, new Set([...selectedForA.identityReferences].reverse()), true).dirty, false);
  const reviewOnly = { ...selectedForA, identityReferences: [] };
  assert.equal(bestFacesetIdentityContract(reviewOnly, new Set(["unconfirmed.jpg"]), false).canPublish, true);
  assert.equal(bestFacesetIdentityContract({ ...selectedForA, state: "analyzing", selection: null }, new Set(selectedForA.identityReferences), true).canPublish, false);
});
