import assert from "node:assert/strict";
import path from "node:path";
import os from "node:os";
import test from "node:test";
import { createJsonProgressReader } from "../server/progress-reader.mjs";
import { BestFacesetManager } from "../server/best-faceset-manager.mjs";
import { RestorationManager } from "../server/restoration-manager.mjs";

function bytewise(reader, text) { for (const byte of Buffer.from(text)) reader.write(Buffer.from([byte])); }

test("progress reader preserves UTF-8 across every byte boundary and flushes the final unclosed line once", () => {
  const seen = [], reader = createJsonProgressReader(value => seen.push(value));
  bytewise(reader, 'model diagnostic\nDFL_PROGRESS {"stage":"正在分析人脸🙂","current":1}\r\nDFL_PROGRESS {"current":2}');
  assert.deepEqual(seen, [{ stage: "正在分析人脸🙂", current: 1 }]);
  reader.end(); reader.end(); reader.write('DFL_PROGRESS {"current":3}\n');
  assert.deepEqual(seen, [{ stage: "正在分析人脸🙂", current: 1 }, { current: 2 }]);
});

test("progress reader discards malformed and oversized whole lines, then recovers at the next boundary", () => {
  const seen = [], reader = createJsonProgressReader(value => seen.push(value), { maxLineLength: 60 });
  reader.write('DFL_PROGRESS {broken}\n');
  reader.write("x".repeat(100_000)); reader.write('DFL_PROGRESS {"spoofed":true}\n');
  reader.write('DFL_PROGRESS {"done":true}\n'); reader.end();
  assert.deepEqual(seen, [{ done: true }]);
});

test("progress callback failures remain advisory and do not lose following records", () => {
  const seen = [], reader = createJsonProgressReader(value => {
    if (value.current === 1) throw new Error("controlled observer error");
    seen.push(value);
  });
  reader.write('DFL_PROGRESS {"current":1}\nDFL_PROGRESS {"current":2}'); reader.end();
  assert.deepEqual(seen, [{ current: 2 }]);
});

test("best-faceset and restoration adapters share boundary-safe progress without changing result JSON", async () => {
  const seen = [];
  const runProcess = async (_executable, _args, options) => {
    const prefix = options.label === "源帧修复" ? "RESTORATION_PROGRESS " : "DFL_PROGRESS ";
    const reader = { write: options.onStderr };
    bytewise(reader, `${prefix}{"stage":"完成中文分块","completed":2}`);
    return { code: 0, stdout: '{"result":"unchanged"}', stderr: "" };
  };
  const facesets = new BestFacesetManager({ paths: { python: "mock-python", webuiRoot: "mock-webui", repositoryRoot: "mock-repository" }, runProcess });
  assert.deepEqual(await facesets.helper("status", [], { onProgress: value => seen.push(value) }), { result: "unchanged" });
  const workspaceRoot = path.join(os.tmpdir(), "not-created-progress-fixture", "workspace");
  const restoration = new RestorationManager({ workspaceRoot, runtimeRoot: path.join(workspaceRoot, ".webui"), runProcess });
  assert.deepEqual(await restoration.runPython("not-created-request.json", { onProgress: value => seen.push(value) }), { result: "unchanged" });
  assert.deepEqual(seen, Array(2).fill({ stage: "完成中文分块", completed: 2 }));
});
