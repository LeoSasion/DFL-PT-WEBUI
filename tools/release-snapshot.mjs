import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { sourceFileHash } from "../release/source-integrity.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const listed = execFileSync("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], { cwd: root }).toString().split("\0").filter(Boolean);
const selected = listed.filter(file => /^(?:webui\/(?:src|shared|server|scripts|public|worker)\/|webui\/(?:package.json|pnpm-lock.yaml|vite.config.mjs)$|release\/|launcher\/|tools\/|_internal\/DeepFaceLab\/|legacy-cli\/)/.test(file)
  && !["release/source-pin.json", "release/source-files.json", "release/installation.json", "tools/prepare-launcher-tests.ps1", "tools/run-launcher-tests.ps1"].includes(file)
  && !/(?:^|\/)(?:tests?|design|\.openai)\//.test(file))
  .sort();
const files = await Promise.all(selected.map(async file => ({ path: file, sha256: sourceFileHash(file, await readFile(path.join(root, file))) })));
const payload = { schemaVersion: 1, product: "DFL-PT-WEBUI", files };
if (process.argv[2] === "write") {
  await writeFile(path.join(root, "release/source-files.json"), `${JSON.stringify(payload, null, 2)}\n`);
} else {
  const actual = JSON.parse(await readFile(path.join(root, "release/source-files.json"), "utf8"));
  if (JSON.stringify(actual) !== JSON.stringify(payload)) throw new Error("source snapshot changed; regenerate before a fixed release");
}
console.log(`source snapshot ${createHash("sha256").update(JSON.stringify(files)).digest("hex")} (${files.length} files)`);
