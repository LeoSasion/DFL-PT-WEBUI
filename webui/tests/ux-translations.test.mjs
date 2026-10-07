import assert from "node:assert/strict";
import test from "node:test";
import { UX_EN } from "../src/ux-translations.mjs";

test("UX translations preserve filenames, counts, ranges and version placeholders", () => {
  const placeholders = value => [...value.matchAll(/\{([A-Za-z][A-Za-z0-9_]*)\}/g)].map(match => match[1]).sort();
  for (const [chinese, english] of Object.entries(UX_EN)) {
    assert.equal(typeof english, "string", chinese);
    assert.ok(english.length > 0, chinese);
    assert.deepEqual(placeholders(english), placeholders(chinese), chinese);
    assert.equal(/[\u4e00-\u9fff]/u.test(english), false, chinese);
  }
});
