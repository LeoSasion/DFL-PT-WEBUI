import path from "node:path";
import { PATHS } from "./paths.mjs";

const delimiter = path.delimiter;

export function buildDflEnvironment(profile, additions = {}) {
  if (!["current", "legacy", "pytorch"].includes(profile)) {
    throw new Error(`未知 DFL 运行时：${profile}`);
  }

  const internal = PATHS.internalRoot;
  const inheritedEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !["PATH", "PYTHONHOME", "PYTHONPATH", "QT_QPA_PLATFORM_PLUGIN_PATH"].includes(key.toUpperCase())),
  );
  const pythonRoot = path.join(PATHS.repositoryRoot, ".venv");
  const pythonScripts = path.join(pythonRoot, "Scripts");
  const runtimePaths = [
    pythonScripts,
    path.join(internal, "node", "bin"),
    path.join(internal, "ffmpeg"),
    Object.entries(process.env).find(([key]) => key.toUpperCase() === "PATH")?.[1] ?? "",
  ];

  return {
    ...inheritedEnvironment,
    PYTHONHOME: "",
    PYTHONPATH: "",
    PYTHONEXECUTABLE: PATHS.python,
    PYTHONWEXECUTABLE: path.join(pythonScripts, "pythonw.exe"),
    PYTHON_EXECUTABLE: PATHS.python,
    PYTHONW_EXECUTABLE: path.join(pythonScripts, "pythonw.exe"),
    PYTHON_BIN_PATH: PATHS.python,
    PYTHON_LIB_PATH: path.join(pythonRoot, "Lib", "site-packages"),
    QT_QPA_PLATFORM_PLUGIN_PATH: path.join(pythonRoot, "Lib", "site-packages", "PySide6", "plugins"),
    FFMPEG_PATH: path.join(internal, "ffmpeg"),
    WORKSPACE: PATHS.workspaceRoot,
    DFL_WORKSPACE: PATHS.workspaceRoot,
    DFL_ACTIVE_PROJECT_WORKSPACE: PATHS.workspaceRoot,
    DFL_ACTIVE_PROJECT_ID: PATHS.activeProject.id,
    DFL_ROOT: profile === "legacy" ? PATHS.legacyDflRoot : PATHS.currentDflRoot,
    PYTHONIOENCODING: "utf-8",
    PATH: runtimePaths.filter(Boolean).join(delimiter),
    ...additions,
  };
}

export function describeEnvironment(profile) {
  const env = buildDflEnvironment(profile);
  return {
    profile,
    python: PATHS.python,
    dflRoot: env.DFL_ROOT,
    workspace: env.WORKSPACE,
  };
}
