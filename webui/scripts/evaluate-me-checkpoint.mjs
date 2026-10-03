#!/usr/bin/env node
/** Read-only offline ME evaluation using the Web pose manifest contract. */

import { spawn } from "node:child_process";
import { realpath, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildDflEnvironment } from "../server/environment.mjs";
import { PATHS } from "../server/paths.mjs";
import {
  createTrainingModelKey,
  TrainingEvaluationManager,
} from "../server/training-evaluation-manager.mjs";

const HELP = `Usage: node webui/scripts/evaluate-me-checkpoint.mjs \\
  --model MODEL_DIR_OR_ME_PT --src SRC_FACESET --dst DST_FACESET \\
  --name MODEL_NAME --evaluation-root OUTPUT_ROOT \\
  (--manifest EXISTING_MANIFEST | --create-manifest) \\
  [--model-key KEY] [--device cpu|cuda:0] [--timeout-seconds 120] \\
  [--checkpoint-sha256 SHA256]

OUTPUT_ROOT is an isolated directory. Generated manifests and snapshots live in
OUTPUT_ROOT/<model-key>/. SRC and DST may be loose directories, .pak or .zip
facesets, including paths outside the active Web project. --manifest must name
an existing canonical manifest in OUTPUT_ROOT/<model-key>/manifests/.
`;

const VALUE_OPTIONS = new Set([
  "model", "src", "dst", "name", "evaluation-root", "manifest", "model-key",
  "device", "timeout-seconds", "checkpoint-sha256",
]);

export function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help" || token === "-h") return { help: true };
    if (token === "--create-manifest") {
      if (options.createManifest) throw new Error("--create-manifest was repeated");
      options.createManifest = true;
      continue;
    }
    if (!token.startsWith("--") || !VALUE_OPTIONS.has(token.slice(2))) {
      throw new Error(`Unknown option: ${token}`);
    }
    const key = token.slice(2);
    if (key in options || !argv[index + 1] || argv[index + 1].startsWith("--")) {
      throw new Error(`Expected one value for ${token}`);
    }
    options[key] = argv[++index];
  }
  for (const key of ["model", "src", "dst", "name", "evaluation-root"]) {
    if (!options[key]) throw new Error(`--${key} is required`);
  }
  if (Boolean(options.manifest) === Boolean(options.createManifest)) {
    throw new Error("Choose exactly one of --manifest and --create-manifest");
  }
  if (!options.name.trim() || options.name.length > 128) {
    throw new Error("--name must contain 1–128 characters");
  }
  const generatedKey = createTrainingModelKey(options.name, "ME");
  const modelKey = options["model-key"] ?? generatedKey;
  if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(modelKey)) {
    throw new Error("--model-key is invalid");
  }
  if (options.createManifest && modelKey !== generatedKey) {
    throw new Error("--model-key must match the generated key when creating a manifest");
  }
  const timeoutSeconds = Number(options["timeout-seconds"] ?? 120);
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 1 || timeoutSeconds > 3600) {
    throw new Error("--timeout-seconds must be an integer from 1 to 3600");
  }
  if (options["checkpoint-sha256"]
      && !/^[a-f0-9]{64}$/.test(options["checkpoint-sha256"])) {
    throw new Error("--checkpoint-sha256 must be 64 lowercase hex characters");
  }
  const outputRoot = path.resolve(options["evaluation-root"]);
  return {
    model: path.resolve(options.model),
    src: path.resolve(options.src),
    dst: path.resolve(options.dst),
    name: options.name,
    outputRoot,
    evaluationRoot: path.join(outputRoot, modelKey),
    modelKey,
    manifest: options.manifest ? path.resolve(options.manifest) : null,
    createManifest: Boolean(options.createManifest),
    device: options.device ?? "cpu",
    timeoutSeconds,
    checkpointSha256: options["checkpoint-sha256"] ?? null,
  };
}

async function physicalPath(target) {
  // Resolve existing symlink ancestors even when the chosen output does not
  // exist yet. This guard must run before the manager creates any directories.
  const absolute = path.resolve(target);
  try {
    return await realpath(absolute);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
    const parent = path.dirname(absolute);
    if (parent === absolute) throw error;
    return path.join(await physicalPath(parent), path.basename(absolute));
  }
}

function contains(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative));
}

export async function assertIsolatedRoot(options) {
  const root = await physicalPath(options.outputRoot);
  const model = await physicalPath(options.model);
  const modelStat = await stat(model);
  const checkpoint = modelStat.isDirectory() ? path.join(model, "me.pt") : model;
  const checkpointStat = await stat(checkpoint);
  if (!checkpointStat.isFile() || path.extname(checkpoint).toLowerCase() !== ".pt") {
    throw new Error("--model must select an ME .pt checkpoint or a directory containing me.pt");
  }
  const protectedPaths = [
    ["checkpoint", modelStat.isDirectory() ? model : path.dirname(model)],
    ["SRC", await physicalPath(options.src)],
    ["DST", await physicalPath(options.dst)],
  ];
  for (const [label, selected] of protectedPaths) {
    if (contains(root, selected) || contains(selected, root)) {
      throw new Error(`Evaluation root overlaps ${label}; choose an isolated output directory`);
    }
  }
}

function runCaptured(command, args, { timeoutMs = 600_000, limitBytes = 4 * 1024 * 1024 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: PATHS.currentDflRoot,
      env: buildDflEnvironment("current", { PYTHONDONTWRITEBYTECODE: "1" }),
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    let bytes = 0;
    let settled = false;
    const timer = setTimeout(() => {
      child.kill();
      finish(new Error(`Pose probe exceeded ${Math.ceil(timeoutMs / 1000)} seconds`));
    }, timeoutMs);
    function finish(error, output) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(output);
    }
    function collect(chunks) {
      return (chunk) => {
        bytes += chunk.length;
        if (bytes > limitBytes) {
          child.kill();
          finish(new Error("Pose probe output exceeds 4 MiB"));
        } else {
          chunks.push(chunk);
        }
      };
    }
    child.stdout.on("data", collect(stdout));
    child.stderr.on("data", collect(stderr));
    child.once("error", finish);
    child.once("close", (code) => {
      if (code !== 0) {
        finish(new Error(Buffer.concat(stderr).toString("utf8").trim() || `Pose probe exited ${code}`));
        return;
      }
      try {
        finish(null, JSON.parse(Buffer.concat(stdout).toString("utf8")));
      } catch {
        finish(new Error("Pose probe returned invalid JSON"));
      }
    });
  });
}

async function buildOfflinePoseProbe(side, { datasetPath }) {
  const helper = path.join(PATHS.webuiRoot, "python", "dfl_asset_tool.py");
  return runCaptured(PATHS.python, [
    helper, "probe-manifest", "--directory", datasetPath, "--side", side,
  ]);
}

function runEvaluator(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(PATHS.python, args, {
      cwd: PATHS.currentDflRoot,
      env: buildDflEnvironment("current", { PYTHONDONTWRITEBYTECODE: "1" }),
      windowsHide: true,
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("close", (code) => resolve(code ?? 1));
  });
}

export async function run(options) {
  await assertIsolatedRoot(options);
  let manifest = options.manifest;
  if (options.createManifest) {
    const manager = new TrainingEvaluationManager({
      root: options.outputRoot,
      archiveRoot: path.join(options.outputRoot, ".archive"),
      probeBuilder: buildOfflinePoseProbe,
    });
    await manager.initialize();
    const generated = await manager.createOrReuseManifest(options.modelKey, {
      modelName: options.name,
      modelClass: "ME",
      srcFaceset: options.src,
      dstFaceset: options.dst,
    });
    manifest = manager.manifestPath(options.modelKey, generated.manifestId);
    process.stdout.write(`ME pose manifest: ${manifest}\n`);
  }
  const args = [
    path.join(PATHS.webuiRoot, "python", "evaluate_me_checkpoint.py"),
    "--model", options.model,
    "--src", options.src,
    "--dst", options.dst,
    "--name", options.name,
    "--manifest", manifest,
    "--evaluation-root", options.evaluationRoot,
    "--model-key", options.modelKey,
    "--device", options.device,
    "--timeout-seconds", String(options.timeoutSeconds),
  ];
  if (options.checkpointSha256) args.push("--checkpoint-sha256", options.checkpointSha256);
  return runEvaluator(args);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      process.stdout.write(HELP);
    } else {
      process.exitCode = await run(options);
    }
  } catch (error) {
    process.stderr.write(`${error.message}\n${HELP}`);
    process.exitCode = 2;
  }
}
