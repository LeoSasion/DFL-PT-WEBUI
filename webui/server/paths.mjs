import { constants as fsConstants, lstatSync, readFileSync, realpathSync } from "node:fs";
import { access, mkdir, readFile } from "node:fs/promises";
import { writeJsonFileAtomic } from "./atomic-json.mjs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultProjectRegistry, parseProjectRegistry } from "../shared/project-identity.mjs";

const serverDirectory = path.dirname(fileURLToPath(import.meta.url));
const webuiRoot = path.resolve(serverDirectory, "..");
const repositoryRoot = path.resolve(webuiRoot, "..");
const internalRoot = path.join(repositoryRoot, "_internal");
const dflRoot = path.join(internalRoot, "DeepFaceLab");
const dflMain = path.join(dflRoot, "main.py");
const projectRegistryRoot = path.join(webuiRoot, ".runtime");
const projectRegistryFile = path.join(projectRegistryRoot, "projects.json");
const managedWorkspacesRoot = path.join(repositoryRoot, "workspaces");

function readActiveProject() {
  let registry;
  try {
    registry = parseProjectRegistry(readFileSync(projectRegistryFile, "utf8"));
  } catch (cause) {
    if (cause.code !== "ENOENT") {
      const error = new Error("项目清单无法读取；请修复清单后重启，工作区未被选择", { cause });
      error.code = "PROJECT_REGISTRY_UNREADABLE";
      throw error;
    }
    registry = defaultProjectRegistry();
  }
  const record = registry.projects.find(project => project.id === registry.activeId);
  const workspaceRoot = record.id === "default" ? path.join(repositoryRoot, "workspace") : path.join(managedWorkspacesRoot, record.id);
  // Resolve the nearest existing ancestor too: a missing project beneath an
  // externally redirected workspaces directory must not be created there.
  let existing = workspaceRoot;
  const suffix = [];
  let resolved;
  while (true) {
    let entry;
    try {
      entry = lstatSync(existing);
      if (existing === workspaceRoot && entry.isSymbolicLink()) throw new Error("Redirected workspace");
      resolved = path.resolve(realpathSync(existing), ...suffix);
      break;
    } catch (cause) {
      if (cause.code !== "ENOENT" || entry?.isSymbolicLink()) throw Object.assign(new Error("项目工作区无法安全解析", { cause }), { code: "PROJECT_WORKSPACE_UNSAFE" });
      suffix.unshift(path.basename(existing)); existing = path.dirname(existing);
    }
  }
  const relative = path.relative(realpathSync(repositoryRoot).toLowerCase(), resolved.toLowerCase());
  if (relative.startsWith("..") || path.isAbsolute(relative)) throw Object.assign(new Error("项目工作区超出仓库范围"), { code: "PROJECT_WORKSPACE_UNSAFE" });
  return { id: record.id, name: record.name, workspaceRoot };
}

const activeProject = readActiveProject();
const workspaceRoot = activeProject.workspaceRoot;
const runtimeRoot = path.join(workspaceRoot, ".webui");

export const PATHS = Object.freeze({
  serverDirectory,
  webuiRoot,
  repositoryRoot,
  internalRoot,
  projectRegistryRoot,
  projectRegistryFile,
  managedWorkspacesRoot,
  activeProject,
  workspaceRoot,
  runtimeRoot,
  jobsRoot: path.join(runtimeRoot, "jobs"),
  archiveRoot: path.join(runtimeRoot, "archive"),
  trainingEvaluationsRoot: path.join(runtimeRoot, "training-evaluations"),
  trainingEvaluationsArchiveRoot: path.join(runtimeRoot, "archive", "training-evaluations"),
  python: path.join(repositoryRoot, ".venv", "Scripts", "python.exe"),
  ffmpeg: path.join(internalRoot, "ffmpeg", "ffmpeg.exe"),
  ffprobe: path.join(internalRoot, "ffmpeg", "ffprobe.exe"),
  dflRoot,
  dflMain,
  meMain: path.join(internalRoot, "DeepFaceLab", "me.py"),
  taeWeights: path.join(internalRoot, "DeepFaceLab", "pretrained", "taeqi2_1.safetensors"),
  taeMain: path.join(internalRoot, "DeepFaceLab", "tae.py"),
  // Compatibility names for existing integrations; all execution uses dflMain/dflRoot.
  currentMain: dflMain,
  legacyMain: dflMain,
  currentDflRoot: dflRoot,
  legacyDflRoot: dflRoot,
  staticRoot: path.join(webuiRoot, "dist", "client"),
});

function canonicalForComparison(value) {
  return path.resolve(value).replaceAll("/", "\\").toLocaleLowerCase("en-US");
}

export function assertWithin(parent, candidate, label = "路径") {
  const parentPath = canonicalForComparison(parent);
  const candidatePath = canonicalForComparison(candidate);
  if (candidatePath !== parentPath && !candidatePath.startsWith(`${parentPath}\\`)) {
    throw new Error(`${label}超出允许范围`);
  }
  return path.resolve(candidate);
}

export function jobDirectory(jobId) {
  if (!/^[a-z0-9][a-z0-9-]{5,63}$/i.test(jobId)) {
    throw new Error("任务 ID 不合法");
  }
  return assertWithin(PATHS.jobsRoot, path.join(PATHS.jobsRoot, jobId), "任务目录");
}

export async function pathExists(target) {
  try {
    await access(target, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function ensureRuntimeDirectories() {
  await Promise.all([
    mkdir(PATHS.jobsRoot, { recursive: true }),
    mkdir(PATHS.archiveRoot, { recursive: true }),
    mkdir(PATHS.trainingEvaluationsRoot, { recursive: true }),
    mkdir(PATHS.trainingEvaluationsArchiveRoot, { recursive: true }),
    mkdir(path.join(PATHS.internalRoot, "_e", "t"), { recursive: true }),
    mkdir(path.join(PATHS.internalRoot, "_e", "u", "AppData", "Local"), { recursive: true }),
    mkdir(path.join(PATHS.internalRoot, "_e", "u", "AppData", "Roaming"), { recursive: true }),
  ]);
}

export async function writeJsonAtomic(target, value) {
  assertWithin(PATHS.runtimeRoot, target, "运行时文件");
  await writeJsonFileAtomic(target, value);
}

export async function readJson(target) {
  return JSON.parse(await readFile(target, "utf8"));
}
