import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { sourceFileHash } from "./source-integrity.mjs";

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const hashPattern = /^[a-f0-9]{64}$/i;
const commitPattern = /^[a-f0-9]{40}$/i;
const version = value => /^\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?$/i.test(value ?? "") ? value : "unknown";
async function json(root, relative) {
  try { return JSON.parse(await readFile(path.join(root, relative), "utf8")); }
  catch { return null; }
}

// This allowlist never reads user settings, Git's private HEAD, or runtime logs.
export async function inspectInstallation(root = defaultRoot) {
  const [release, record, pin, manifest] = await Promise.all([
    json(root, "release/version.json"), json(root, "release/installation.json"),
    json(root, "release/source-pin.json"), json(root, "release/source-files.json"),
  ]);
  const installed = record?.product === "DFL-PT-WEBUI" ? record : null;
  let verified = false;
  if (manifest?.product === "DFL-PT-WEBUI" && Array.isArray(manifest.files) && manifest.files.length > 0) {
    verified = true;
    for (const entry of manifest.files) {
      if (!entry || !hashPattern.test(entry.sha256 ?? "") || typeof entry.path !== "string"
        || !/^(?:webui\/(?:src|shared|server|scripts|public|worker)\/|webui\/(?:package.json|pnpm-lock.yaml|vite.config.mjs)$|release\/|launcher\/|tools\/|_internal\/DeepFaceLab\/|legacy-cli\/)/.test(entry.path)
        || entry.path.split(/[\\/]/).some(part => part === "..") || entry.path.includes("\\")) { verified = false; break; }
      try {
        const bytes = await readFile(path.join(root, entry.path));
        if (sourceFileHash(entry.path, bytes) !== entry.sha256) { verified = false; break; }
      } catch { verified = false; break; }
    }
  }
  const source = installed ?? (pin?.product === "DFL-PT-WEBUI" ? pin : {});
  return {
    product: "DFL-PT-WEBUI",
    appVersion: version(release?.version),
    launcherVersion: version(installed?.launcherVersion ?? release?.launcherVersion),
    installationSource: ["official-source", "source", "portable"].includes(installed?.installationSource)
      ? installed.installationSource : "local-source",
    source: {
      revision: commitPattern.test(source.sourceCommit ?? "") ? source.sourceCommit : null,
      tree: commitPattern.test(source.sourceTree ?? "") ? source.sourceTree : null,
      archiveSha256: hashPattern.test(source.archiveSha256 ?? "") ? source.archiveSha256 : null,
      contentSha256: manifest ? createHash("sha256").update(JSON.stringify(manifest.files ?? [])).digest("hex") : null,
      verified,
      status: verified ? "content-verified" : manifest ? "modified-or-incomplete" : "unverified",
    },
    portable: { version: "0.1.1-preview", status: "previous-release-not-rebuilt" },
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const index = process.argv.indexOf("--root");
  console.log(JSON.stringify(await inspectInstallation(index >= 0 ? process.argv[index + 1] : undefined)));
}
