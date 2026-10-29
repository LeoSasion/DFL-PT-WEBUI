export const CUDA_MUTATIONS = new Set([
  "prepareCudaRuntime", "keepCudaRuntime", "deleteCudaRuntime", "recoverCudaRuntime",
]);

export const CUDA_WORKING_STATES = new Set(["downloading", "switching", "verifying", "recovering", "deleting"]);

export function runtimeLabel(runtime, supplied) {
  if (supplied) return supplied;
  if (runtime === "cu126") return "CUDA 12.6";
  if (runtime === "cu128") return "CUDA 12.8";
  return "当前运行环境";
}

export function formatRuntimeBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "空间待确认";
  if (bytes < 1000000) return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(bytes / 1000) + " KB";
  if (bytes < 1000000000) return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(bytes / 1000000) + " MB";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(bytes / 1000000000) + " GB";
}

export function normalizeCudaRuntime(value) {
  if (!value || typeof value !== "object") return null;
  return {
    ...value,
    gpuNames: Array.isArray(value.gpuNames) ? value.gpuNames.filter((name) => typeof name === "string") : [],
    retainedRuntimes: Array.isArray(value.retainedRuntimes) ? value.retainedRuntimes.filter((row) => row && typeof row.transactionId === "string"
      && ["cu126", "cu128"].includes(row.runtime)) : [],
  };
}

export function projectDirectory(state) { return state.projectDir || state.installPath || ""; }

export function sameProject(left, right) {
  const key = (value) => typeof value === "string" ? value.replaceAll("/", "\\").replace(/\\+$/, "").toLowerCase() : "";
  return Boolean(key(left) && key(left) === key(right));
}

export function scopeCudaRuntime(value, projectDir) {
  const runtime = normalizeCudaRuntime(value);
  if (!runtime || !projectDir || (runtime.projectDir && !sameProject(runtime.projectDir, projectDir))) return null;
  return { ...runtime, projectDir };
}

export function cleanupRuntime(runtime, transactionId = runtime?.transactionId) {
  if (!runtime || !transactionId || runtime.canRecover) return null;
  const row = runtime.retainedRuntimes?.find((item) => item.transactionId === transactionId);
  if (!row && transactionId !== runtime.transactionId) return null;
  return { ...runtime, transactionId, retainedRuntime: row?.runtime || runtime.retainedRuntime,
    retainedBytes: row?.bytes ?? runtime.retainedBytes,
    canDelete: row ? Boolean(row.canDelete) : Boolean(runtime.canDelete),
    deleting: Boolean(row?.deleting || (transactionId === runtime.transactionId && runtime.deleting)) };
}

export function cudaActionBlocked(state, busy) {
  return Boolean(busy || state.environmentStatus === "changing" || state.webuiRunning || state.terminalRunning);
}

export function canOfferRuntimeSwitch(runtime) {
  return Boolean(runtime && !runtime.canRecover && runtime.canSwitch
    && ["cu126", "cu128"].includes(runtime.targetRuntime)
    && ["change-required", "error"].includes(runtime.status));
}
