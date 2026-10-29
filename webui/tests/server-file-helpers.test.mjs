import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { writeJsonFileAtomic } from "../server/atomic-json.mjs";
import { sha256File } from "../server/hash-file.mjs";

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dfl-server-files-"));
  t.after(async () => {
    assert.equal(path.dirname(path.resolve(root)), path.resolve(os.tmpdir()));
    await rm(root, { recursive: true, force: true });
  });
  return root;
}

test("atomic JSON concurrent replacements preserve complete UTF-8 records and leave no temporary files", async t => {
  const root = await fixture(t), file = path.join(root, "state.json");
  const values = Array.from({ length: 40 }, (_, revision) => ({ revision, label: "辅助遮罩与源帧", samples: Array(30).fill(revision) }));
  await Promise.all(values.map(value => writeJsonFileAtomic(file, value)));
  const actual = JSON.parse(await readFile(file, "utf8"));
  assert.deepEqual(actual, values[actual.revision]);
  assert.equal(await readFile(file, "utf8"), `${JSON.stringify(actual, null, 2)}\n`);
  assert.deepEqual(await readdir(root), ["state.json"]);
});

test("atomic JSON failed replacement cleans its sibling file and preserves the original target", async t => {
  const root = await fixture(t), target = path.join(root, "occupied");
  await mkdir(target); await writeFile(path.join(target, "receipt.json"), "original");
  await assert.rejects(writeJsonFileAtomic(target, { replacement: true }));
  assert.equal(await readFile(path.join(target, "receipt.json"), "utf8"), "original");
  assert.deepEqual(await readdir(root), ["occupied"]);
  const circular = {}; circular.self = circular;
  await assert.rejects(writeJsonFileAtomic(path.join(root, "invalid.json"), circular), TypeError);
  assert.deepEqual(await readdir(root), ["occupied"]);
});

test("atomic JSON private mode and streaming SHA256 preserve file contents", async t => {
  const root = await fixture(t), file = path.join(root, "private.json");
  await writeJsonFileAtomic(file, { secret: "test-only" }, { mode: 0o600 });
  if (process.platform !== "win32") assert.equal((await stat(file)).mode & 0o777, 0o600);
  for (const size of [0, 1, 2 * 1024 * 1024 + 13]) {
    const bytes = Buffer.alloc(size, 0xa7), target = path.join(root, `bytes-${size}`);
    await writeFile(target, bytes);
    assert.equal(await sha256File(target), createHash("sha256").update(bytes).digest("hex"));
  }
  await assert.rejects(sha256File(path.join(root, "missing.bin")), { code: "ENOENT" });
});
