import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

// Hash large media and weights without allocating a buffer for the whole file.
// Path, type, and before/after identity checks remain with the caller.
export async function sha256File(target) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(target)) hash.update(chunk);
  return hash.digest("hex");
}
