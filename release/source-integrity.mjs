import { createHash } from "node:crypto";
import path from "node:path";

// Git archives store LF while Windows checkouts may use CRLF for scripts.
export function sourceFileHash(file, bytes) {
  const text = /\.(?:bat|cmd|ps1|psm1|psd1|py|js|mjs|jsx|json|md|yml|yaml|html|css|sh|cs|toml|txt)$/i.test(path.extname(file));
  const content = text ? Buffer.from(bytes.toString("utf8").replaceAll("\r\n", "\n")) : bytes;
  return createHash("sha256").update(content).digest("hex");
}
