import { IconAlertTriangle, IconArrowRight, IconCheck, IconDownload, IconLoader2, IconRefresh } from "@tabler/icons-react";
import { canOfferRuntimeSwitch, CUDA_WORKING_STATES, cudaActionBlocked, formatRuntimeBytes, runtimeLabel } from "../cuda-runtime.js";

export function CudaRuntimeNotice({ state, busyAction, checking, error, onAction, onReviewCleanup, onManageRetained }) {
  const runtime = state.cudaRuntime;
  if (!runtime) return null;
  const working = CUDA_WORKING_STATES.has(runtime.status) || ["prepareCudaRuntime", "recoverCudaRuntime"].includes(busyAction);
  const retained = !runtime.canRecover && ["cleanup-choice", "retained"].includes(runtime.status) && Boolean(runtime.transactionId);
  const history = runtime.retainedRuntimes;
  const visible = working || retained || history.length > 0 || runtime.canRecover || runtime.canRetryCleanup || ["change-required", "ambiguous", "error", "unavailable"].includes(runtime.status);
  if (!visible) return null;
  const blocked = working || cudaActionBlocked(state, busyAction);
  const current = runtimeLabel(runtime.currentRuntime, runtime.currentLabel);
  const target = runtimeLabel(runtime.targetRuntime, runtime.targetLabel);
  const remainingBytes = runtime.downloadRemainingBytes ?? runtime.downloadBytes;
  const reusable = history.some((row) => row.runtime === runtime.targetRuntime && row.canReuse && !row.deleting);
  const recovered = Boolean(runtime.canRecover);
  const Icon = working ? IconLoader2 : retained || runtime.status === "compatible" ? IconCheck : IconAlertTriangle;
  const title = working ? runtime.canRetryCleanup && !recovered ? "正在清理运行包残留" : "正在准备匹配的运行环境"
    : recovered ? "运行环境切换尚未完成"
      : retained ? runtime.status === "retained" ? "旧版运行环境已保留" : "新运行环境已就绪"
        : runtime.status === "unavailable" ? "暂时无法检查显卡环境"
          : runtime.status === "ambiguous" ? "检测到需要不同运行环境的显卡" : runtime.canRetryCleanup ? "运行包残留待清理" : runtime.status === "compatible" ? "保留的运行环境" : "显卡环境发生变化";
  const measured = !runtime.canRetryCleanup && !recovered && Number.isFinite(runtime.downloadedBytes) && remainingBytes > 0;
  const progress = measured ? Math.max(0, Math.min(runtime.downloadedBytes / remainingBytes * 100, 100)) : null;
  return (
    <section className={"cuda-runtime-notice" + (retained || runtime.status === "compatible" ? " is-success" : "") + (working ? " is-working" : "")}
      aria-label="CUDA 运行环境" aria-busy={working || undefined}>
      <Icon size={21} className={working ? "spin" : ""} aria-hidden="true" />
      <div className="cuda-runtime-notice__copy">
        <h2>{title}</h2>
        <p role="status">{working && !CUDA_WORKING_STATES.has(runtime.status) ? runtime.canRetryCleanup ? "正在清理本项目已确认的下载与临时残留，请保持启动器开启。" : "正在准备并验证所需运行环境，请保持启动器开启。" : runtime.message || (runtime.status === "compatible" ? "当前运行环境与显卡匹配，可以查看本项目保留的旧包。" : retained
          ? "新版本已通过验证并启用，原运行包暂时保留，等待你的清理选择。"
          : "当前运行包与检测到的显卡不匹配，可以下载对应版本后继续使用。")}</p>
        {runtime.gpuNames.length > 0 ? <p className="cuda-runtime-notice__devices">{runtime.gpuNames.join(" · ")}</p> : null}
        {runtime.targetRuntime && !retained ? <p className="cuda-runtime-notice__versions"><span>当前 {current}</span><IconArrowRight size={13} aria-hidden="true" /><strong>建议 {target}</strong>{runtime.downloadRequired !== false && remainingBytes > 0 ? <span>还需下载约 {formatRuntimeBytes(remainingBytes)}</span> : null}</p> : null}
        {retained ? <p className="cuda-runtime-notice__versions"><span>正在使用 {current}</span><span>保留 {runtimeLabel(runtime.retainedRuntime)} · {formatRuntimeBytes(runtime.retainedBytes)}</span></p> : null}
        {working ? <div className="cuda-runtime-progress"><progress aria-label="运行环境准备进度" max="100" value={progress ?? undefined} /><span>{progress === null ? "下载、校验和切换期间请保持启动器开启" : `${formatRuntimeBytes(runtime.downloadedBytes)} / ${formatRuntimeBytes(remainingBytes)}`}</span></div> : null}
        {error || runtime.error ? <p className="cuda-runtime-notice__error" role="alert">{error || runtime.error}</p> : null}
        {runtime.cleanupWarning ? <p className="cuda-runtime-cleanup-warning" role="status">{runtime.cleanupWarning}</p> : null}
        {!working && (state.webuiRunning || state.terminalRunning) ? <p className="cuda-runtime-notice__hint">请先保存并结束所有任务、终端和 WebUI，再切换或清理运行环境。</p> : null}
        {!working && !retained && runtime.status !== "unavailable" ? <p className="cuda-runtime-notice__hint">{recovered ? "请先恢复原运行环境，模型、素材和训练进度保留。" : runtime.status === "compatible" ? "当前运行环境继续使用。你可以查看本项目保留的旧包，单独决定是否清理。" : "模型、素材和训练进度保留。下载验证通过后启用新版，再由你决定是否删除旧版。"}</p> : null}
        {!working && runtime.targetRuntime === "cu126" && !retained ? <p className="cuda-runtime-notice__hint">GTX 10 系列训练请选择 FP32；启动器不会自动更改模型的训练精度。</p> : null}
        {!working && !retained && !recovered && runtime.downloadRequired === false && !reusable ? <p className="cuda-runtime-notice__hint">已有下载缓存，切换前会再次校验。</p> : null}
      </div>
      <div className="cuda-runtime-notice__actions">
        {recovered ? <button className="runtime-button is-primary" disabled={blocked} onClick={() => onAction("recoverCudaRuntime", { transactionId: runtime.transactionId, expectedProjectDir: runtime.projectDir })}><IconRefresh size={16} />恢复原运行环境</button>
          : canOfferRuntimeSwitch(runtime) && !working ? <button className="runtime-button is-primary" disabled={blocked} onClick={() => onAction("prepareCudaRuntime", { targetRuntime: runtime.targetRuntime, expectedProjectDir: runtime.projectDir })}><IconDownload size={16} />{runtime.downloadRequired === false ? reusable ? "切换到已保留的 " : "准备并切换到 " : "下载并切换到 "}{target}</button>
            : null}
        {!recovered && runtime.canRetryCleanup && runtime.transactionId ? <button className="runtime-button" disabled={blocked} onClick={() => onAction("recoverCudaRuntime", { transactionId: runtime.transactionId, expectedProjectDir: runtime.projectDir })}><IconRefresh size={16} />重试清理</button> : null}
        {runtime.status === "ambiguous" && runtime.canSwitch && !working && !recovered ? <>
          <button className="runtime-button" disabled={blocked} onClick={() => onAction("prepareCudaRuntime", { targetRuntime: "cu126", expectedProjectDir: runtime.projectDir })}>使用 GTX 10 · CUDA 12.6</button>
          <button className="runtime-button" disabled={blocked} onClick={() => onAction("prepareCudaRuntime", { targetRuntime: "cu128", expectedProjectDir: runtime.projectDir })}>使用 Blackwell · CUDA 12.8</button>
        </> : null}
        {retained && !working ? <button data-dialog-return-focus className="runtime-button" disabled={blocked} onClick={() => onReviewCleanup(runtime)}>{runtime.status === "retained" ? "清理保留的旧版" : "选择是否保留旧版"}</button> : null}
        {!working && !recovered && history.length > 0 && (!retained || history.some((row) => row.transactionId !== runtime.transactionId)) ? <button data-dialog-return-focus className="runtime-button" disabled={blocked} onClick={() => onManageRetained(runtime)}>管理保留的运行包（{history.length}）</button> : null}
        {!working ? <button className="runtime-button is-quiet" disabled={Boolean(busyAction) || checking} onClick={() => onAction("checkCudaRuntime")}><IconRefresh size={15} className={checking ? "spin" : ""} />{checking ? "检测中" : "重新检测"}</button> : null}
      </div>
    </section>
  );
}
