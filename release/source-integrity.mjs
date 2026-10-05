import { createHash } from "node:crypto";

// Git archives store LF while Windows checkouts may use CRLF for scripts.
export function sourceFileHash(file, bytes) {
  // Match Git's text-auto boundary without relying on a filename extension.
  // This includes LICENSE, XML manifests and SVGs; binary bytes stay exact.
  const text = !bytes.subarray(0, 8000).includes(0) && Buffer.from(bytes.toString("utf8")).equals(bytes);
  const content = text ? Buffer.from(bytes.toString("utf8").replaceAll("\r\n", "\n")) : bytes;
  return createHash("sha256").update(content).digest("hex");
}
