import assert from "node:assert/strict";
import path from "node:path";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import {
  compareEnglishNaturalNames,
  compareNaturalNames,
  directFileStats,
} from "../server/directory-files.mjs";

const directory = path.resolve("synthetic-directory-no-disk-access");
const entry = (name, file = true) => ({ name, isFile: () => file });
const ioError = code => Object.assign(new Error(code), { code });
const emptyStats = { count: 0, bytes: 0, modifiedAt: null };

test("150,000-file statistics aggregate without argument spreading or unbounded stat work", async () => {
  const size = 150_000;
  const entries = Array.from({ length: size }, (_, index) => entry(`${index}.jpg`));
  const visited = new Uint8Array(size);
  const baseTime = Date.parse("2026-01-01T00:00:00.000Z");
  let active = 0;
  let peak = 0;
  let calls = 0;
  const result = await directFileStats(directory, name => name.endsWith(".jpg"), {
    readDirectory: async () => entries,
    statFile: async target => {
      const index = Number(path.basename(target, ".jpg"));
      assert.equal(path.dirname(target), directory);
      assert.equal(visited[index]++, 0, "each file is read exactly once");
      calls += 1;
      active += 1;
      peak = Math.max(peak, active);
      await Promise.resolve();
      active -= 1;
      return { size: index + 1, mtimeMs: baseTime + index };
    },
  });
  assert.deepEqual(result, {
    count: size,
    bytes: size * (size + 1) / 2,
    modifiedAt: new Date(baseTime + size - 1).toISOString(),
  });
  assert.equal(calls, size);
  assert.equal(peak, 16, "the default worker pool stays bounded");
  assert.equal(active, 0);
});

test("statistics retain direct-file filters and support timestamps before the epoch", async () => {
  const names = [];
  const result = await directFileStats(directory, name => /\.jpe?g$/i.test(name), {
    readDirectory: async (target, options) => {
      assert.equal(target, directory);
      assert.deepEqual(options, { withFileTypes: true });
      return [entry("frame2.JPG"), entry(".hidden.jpg"), entry("metadata.json"),
        entry("nested.jpg", false), entry("symlink.jpg", false), entry("frame10.jpeg")];
    },
    statFile: async target => {
      names.push(path.basename(target));
      return target.endsWith(".JPG") ? { size: 0, mtimeMs: -2000 } : { size: 9, mtimeMs: -1000 };
    },
    concurrency: 2,
  });
  assert.deepEqual(names.sort(), ["frame10.jpeg", "frame2.JPG"]);
  assert.deepEqual(result, { count: 2, bytes: 9, modifiedAt: "1969-12-31T23:59:59.000Z" });
});

test("empty and missing directories are empty but read failures remain errors", async () => {
  let statCalls = 0;
  const statFile = async () => { statCalls += 1; throw new Error("unexpected stat"); };
  assert.deepEqual(await directFileStats(directory, undefined, {
    readDirectory: async () => [], statFile,
  }), emptyStats);
  assert.deepEqual(await directFileStats(directory, undefined, {
    readDirectory: async () => { throw ioError("ENOENT"); }, statFile,
  }), emptyStats);
  for (const code of ["EACCES", "EPERM", "ENOTDIR", "EIO"]) {
    const error = ioError(code);
    await assert.rejects(directFileStats(directory, undefined, {
      readDirectory: async () => { throw error; }, statFile,
    }), candidate => candidate === error);
  }
  assert.equal(statCalls, 0);
});

test("a disappearing or unreadable file rejects the partial inventory and drains its bounded reads", async (t) => {
  for (const code of ["ENOENT", "EACCES", "EIO"]) {
    await t.test(code, async () => {
      const error = ioError(code);
      const gates = [];
      let active = 0;
      let settled = false;
      const pending = directFileStats(directory, undefined, {
        readDirectory: async () => Array.from({ length: 100 }, (_, index) => entry(`${index}.jpg`)),
        statFile: async target => {
          active += 1;
          try {
            await new Promise(resolve => gates.push(resolve));
            if (path.basename(target) === "1.jpg") throw error;
            return { size: 1, mtimeMs: 1 };
          } finally {
            active -= 1;
          }
        },
        concurrency: 3,
      });
      pending.then(() => { settled = true; }, () => { settled = true; });
      const rejected = assert.rejects(pending, candidate => candidate === error);
      await setImmediate();
      assert.equal(gates.length, 3);
      gates[1]();
      await setImmediate();
      assert.equal(settled, false, "in-flight reads settle before the caller receives the error");
      assert.equal(active, 2);
      gates[0]();
      gates[2]();
      await rejected;
      assert.equal(gates.length, 3, "no more files are queued after the failure");
      assert.equal(active, 0);
    });
  }
});

test("each request reads fresh directory contents and file statistics", async () => {
  let version = 1;
  let reads = 0;
  const options = {
    readDirectory: async () => { reads += 1; return [entry(`frame${version}.jpg`)]; },
    statFile: async target => {
      assert.equal(path.basename(target), `frame${version}.jpg`);
      return { size: version, mtimeMs: version * 1000 };
    },
  };
  assert.deepEqual(await directFileStats(directory, undefined, options), {
    count: 1, bytes: 1, modifiedAt: "1970-01-01T00:00:01.000Z",
  });
  version = 2;
  assert.deepEqual(await directFileStats(directory, undefined, options), {
    count: 1, bytes: 2, modifiedAt: "1970-01-01T00:00:02.000Z",
  });
  assert.equal(reads, 2);
});

test("invalid concurrency never starts filesystem work", async () => {
  for (const concurrency of [0, -1, 1.5, Infinity, NaN]) {
    await assert.rejects(directFileStats(directory, undefined, {
      readDirectory: async () => assert.fail("unexpected directory read"), concurrency,
    }), RangeError);
  }
});

test("reused natural collators preserve each locale and stable ordering of equal numeric names", () => {
  const names = ["frame10.jpg", "frame2.jpg", "frame02.jpg", "frame1.jpg", "frame001.jpg",
    "Frame2.JPG", "帧10.png", "帧2.png", "é2.jpg", "e10.jpg", "a-2.jpg", "a_10.jpg"];
  for (const [locale, compare] of [[undefined, compareNaturalNames], ["en", compareEnglishNaturalNames]]) {
    const expected = [...names].sort((left, right) => left.localeCompare(right, locale, { numeric: true }));
    assert.deepEqual([...names].sort(compare), expected);
    assert.deepEqual(["frame10.jpg", "frame2.jpg", "frame02.jpg", "frame1.jpg"].sort(compare),
      ["frame1.jpg", "frame2.jpg", "frame02.jpg", "frame10.jpg"]);
    assert.equal(compare("frame2.jpg", "frame02.jpg"), 0);
  }
});
