import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inspectInstallation } from "./installation.mjs";

const template = JSON.parse(await readFile(new URL("./feedback-template.json", import.meta.url), "utf8"));
const numeric = value => value != null && value !== "" && Number.isFinite(Number(value)) ? Number(value) : null;
const safeWord = value => typeof value === "string" && /^[a-z0-9_. -]{1,96}$/i.test(value) ? value : "unknown";

export function feedbackDiagnostic(snapshot = {}) {
  // Deliberately reconstruct fields: raw errors, names, paths, IDs and free text never cross this boundary.
  return {
    platform: safeWord(snapshot.product?.platform), architecture: safeWord(snapshot.product?.architecture),
    node: safeWord(snapshot.product?.node), windowsRelease: safeWord(snapshot.product?.windowsRelease),
    runtime: Object.fromEntries(["pythonAvailable", "currentAvailable", "workspaceAvailable"].map(key => [key, typeof snapshot.runtime?.[key] === "boolean" ? snapshot.runtime[key] : null])),
    datasetCounts: Object.fromEntries(["srcFrames", "dstFrames", "srcFaces", "dstFaces"].map(key => [key, numeric(snapshot.workspace?.datasets?.[key]?.count)])),
    modelCount: numeric(snapshot.workspace?.modelCount), outputCount: numeric(snapshot.workspace?.outputCount),
    gpuCount: Array.isArray(snapshot.telemetry?.gpus) ? snapshot.telemetry.gpus.length : null,
    freeBytes: numeric(snapshot.storage?.freeBytes),
    tasks: (Array.isArray(snapshot.jobs) ? snapshot.jobs : []).slice(-10).map(job => ({
      command: safeWord(job.commandId), status: safeWord(job.status), exitCode: numeric(job.exitCode),
    })),
  };
}

export function prepareFeedback({ release = {}, diagnostics = {}, failureStep = "unknown" } = {}) {
  const step = Object.hasOwn(template.steps, failureStep) ? failureStep : "unknown";
  const safeRelease = {
    product: template.product,
    appVersion: safeWord(release.appVersion), launcherVersion: safeWord(release.launcherVersion),
    installationSource: ["official-source", "source", "portable", "local-source"].includes(release.installationSource) ? release.installationSource : "unknown",
    source: {
      revision: /^[a-f0-9]{40}$/i.test(release.source?.revision ?? "") ? release.source.revision : "unknown",
      tree: /^[a-f0-9]{40}$/i.test(release.source?.tree ?? "") ? release.source.tree : "unknown",
      archiveSha256: /^[a-f0-9]{64}$/i.test(release.source?.archiveSha256 ?? "") ? release.source.archiveSha256 : "unknown",
      contentSha256: /^[a-f0-9]{64}$/i.test(release.source?.contentSha256 ?? "") ? release.source.contentSha256 : "unknown",
      status: ["content-verified", "modified-or-incomplete", "unverified"].includes(release.source?.status) ? release.source.status : "unverified",
    },
  };
  const safeDiagnostic = feedbackDiagnostic(diagnostics);
  return {
    failureStep: step, release: safeRelease, diagnostics: safeDiagnostic, issueUrl: template.issueUrl,
    preview: `${template.instructions}\n\n产品：${template.product}\n失败步骤：${template.steps[step]}\n应用版本：${safeRelease.appVersion}\n启动器版本：${safeRelease.launcherVersion}\n安装来源：${safeRelease.installationSource}\n\n源码与环境摘要：\n${JSON.stringify({ source: safeRelease.source, ...safeDiagnostic }, null, 2)}\n\n重现步骤：\n1. \n\n预期行为：\n\n实际行为：\n`,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const argument = name => { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] : undefined; };
  const release = await inspectInstallation(argument("--root"));
  console.log(JSON.stringify(prepareFeedback({ release, failureStep: argument("--step") })));
}
