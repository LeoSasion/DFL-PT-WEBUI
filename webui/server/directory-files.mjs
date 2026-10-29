import { readdir, stat } from "node:fs/promises";
import path from "node:path";

// Keep the original callers' locales; construct each collator only once.
export const compareNaturalNames = new Intl.Collator(undefined, { numeric: true }).compare;
export const compareEnglishNaturalNames = new Intl.Collator("en", { numeric: true }).compare;

export async function directFileStats(target, accept = () => true, {
  readDirectory = readdir,
  statFile = stat,
  concurrency = 16,
} = {}) {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1) {
    throw new RangeError("File-stat concurrency must be a positive integer");
  }
  let entries;
  try {
    entries = await readDirectory(target, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return { count: 0, bytes: 0, modifiedAt: null };
    // An unreadable directory is unknown, not an empty inventory.
    throw error;
  }

  let cursor = 0;
  let count = 0;
  let bytes = 0;
  let newest = -Infinity;
  let failed = false;
  let failure;
  async function consume() {
    try {
      while (!failed && cursor < entries.length) {
        const entry = entries[cursor++];
        if (!entry.isFile() || entry.name.startsWith(".") || !accept(entry.name)) continue;
        const info = await statFile(path.join(target, entry.name));
        count += 1;
        bytes += info.size;
        newest = Math.max(newest, info.mtimeMs);
      }
    } catch (error) {
      if (!failed) failure = error;
      failed = true;
    }
  }

  // Allocate only the worker promises, not one promise/stat result per file.
  // On error, stop scheduling and drain in-flight reads before rejecting.
  await Promise.all(Array.from({ length: Math.min(concurrency, entries.length) }, consume));
  if (failed) throw failure;
  return { count, bytes, modifiedAt: count ? new Date(newest).toISOString() : null };
}
