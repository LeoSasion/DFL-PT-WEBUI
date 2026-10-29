import { randomUUID } from "node:crypto";
import { rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

const writes = new Map();

// Callers retain their own path and schema checks. The exclusive sibling file
// prevents concurrent writes from sharing a temporary file, even in one tick.
export async function writeJsonFileAtomic(target, value, { mode = 0o666 } = {}) {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  const resolved = path.resolve(target);
  const key = process.platform === "win32" ? resolved.toLocaleLowerCase("en-US") : resolved;
  // Windows can reject overlapping replacement renames with EPERM. Queue only
  // writers to this file; unrelated records remain independent.
  const pending = (writes.get(key) ?? Promise.resolve()).catch(() => {}).then(() => replaceJson(resolved, text, mode));
  writes.set(key, pending);
  try { await pending; }
  finally { if (writes.get(key) === pending) writes.delete(key); }
}

async function replaceJson(target, text, mode) {
  const temporary = `${target}.${randomUUID()}.tmp`;
  let owned = true;
  try {
    try { await writeFile(temporary, text, { encoding: "utf8", mode, flag: "wx" }); }
    catch (error) { if (error.code === "EEXIST") owned = false; throw error; }
    await rename(temporary, target);
  } finally {
    if (owned) await unlink(temporary).catch(error => { if (error.code !== "ENOENT") throw error; });
  }
}
