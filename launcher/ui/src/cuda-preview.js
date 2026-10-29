const downloadBytes = 2593299994;
const retainedBytes = 4537000000;

export function cudaPreview(status = "compatible") {
  const base = {
    status, currentRuntime: "cu128", currentLabel: "CUDA 12.8", targetRuntime: null,
    targetLabel: null, gpuNames: ["NVIDIA RTX PRO 6000 Blackwell"],
    message: "当前运行环境与显卡匹配。", downloadBytes: 0, retainedBytes: 0,
    transactionId: null, retainedRuntime: null, downloadRequired: true,
    canSwitch: false, canDelete: false, canRecover: false, error: "", retainedRuntimes: [],
  };
  if (["change-required", "downloading", "error"].includes(status)) {
    Object.assign(base, { targetRuntime: "cu126", targetLabel: "CUDA 12.6", gpuNames: ["NVIDIA GeForce GTX 1080 Ti"],
      message: "检测到 GTX 10 系列显卡，需要匹配的 CUDA 12.6 运行包。", downloadBytes,
      canSwitch: status !== "downloading" });
  }
  if (status === "downloading") Object.assign(base, { downloadedBytes: 1050000000, message: "正在下载并校验匹配的运行包。" });
  if (status === "error") Object.assign(base, { error: "下载校验未通过，原运行环境未改动。请重新下载。" });
  if (["cleanup-choice", "retained"].includes(status)) {
    Object.assign(base, { currentRuntime: "cu126", currentLabel: "CUDA 12.6", targetRuntime: "cu126", targetLabel: "CUDA 12.6",
      gpuNames: ["NVIDIA GeForce GTX 1080 Ti"], message: status === "retained" ? "原 CUDA 12.8 运行包已保留，可以稍后清理。" : "CUDA 12.6 已通过验证并启用。原 CUDA 12.8 运行包等待你的清理选择。",
      retainedBytes, transactionId: "11111111111111111111111111111111", retainedRuntime: "cu128", canDelete: true,
      retainedRuntimes: [{ transactionId: "11111111111111111111111111111111", runtime: "cu128", label: "CUDA 12.8", bytes: retainedBytes, kept: status === "retained", deleting: false, canDelete: true, canReuse: true }] });
  }
  if (status === "ambiguous") Object.assign(base, { gpuNames: ["NVIDIA GeForce GTX 1080 Ti", "NVIDIA RTX PRO 6000 Blackwell"],
    message: "检测到两种需要不同运行包的显卡。请明确选择本次要使用的显卡；切换不会同时启用两种环境。", canSwitch: true });
  if (status === "unavailable") Object.assign(base, { gpuNames: [], message: "暂时无法读取显卡信息，请检查驱动后重新检测。" });
  if (status === "recover") Object.assign(base, { status: "error", gpuNames: [], canRecover: true,
    transactionId: "22222222222222222222222222222222", message: "上次切换中断，请先恢复原运行环境。", error: "恢复完成前不能继续切换或清理旧版。" });
  if (status === "history") Object.assign(base, { status: "compatible", message: "当前运行环境与显卡匹配，本项目还保留了旧运行包。",
    retainedRuntimes: [
      { transactionId: "33333333333333333333333333333333", runtime: "cu126", bytes: 3500000000, kept: true, deleting: false, canDelete: true, canReuse: true },
      { transactionId: "44444444444444444444444444444444", runtime: "cu128", bytes: retainedBytes, kept: false, deleting: true, canDelete: true, canReuse: false },
    ] });
  if (status === "cleanup-warning") Object.assign(base, { status: "compatible", canRetryCleanup: true,
    transactionId: "55555555555555555555555555555555", message: "当前运行环境可用，还有上次准备的残留需要清理。",
    cleanupWarning: "部分下载与临时残留暂时无法清理，可以稍后重试。" });
  return base;
}
