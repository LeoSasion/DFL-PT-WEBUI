import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InstallView } from "./components/InstallView.jsx";
import { ReadyView } from "./components/ReadyView.jsx";
import { TitleBar } from "./components/TitleBar.jsx";
import { Dialog } from "./components/Dialog.jsx";
import { CudaRuntimeNotice } from "./components/CudaRuntimeNotice.jsx";
import { CUDA_MUTATIONS, CUDA_WORKING_STATES, cleanupRuntime, cudaActionBlocked, formatRuntimeBytes, projectDirectory, runtimeLabel, sameProject, scopeCudaRuntime } from "./cuda-runtime.js";
import { launcherBridge, previewState } from "./bridge.js";

function normalizeState(input) {
  const fallback = previewState();
  if (!input || typeof input !== "object") return fallback;
  return {
    ...fallback,
    ...input,
    steps: Array.isArray(input.steps) ? input.steps : fallback.steps,
    runtimeItems: Array.isArray(input.runtimeItems) ? input.runtimeItems : fallback.runtimeItems,
    cudaRuntime: scopeCudaRuntime(input.cudaRuntime, projectDirectory(input)),
  };
}

function normalizeLog(entry) {
  if (typeof entry === "string") return { text: entry };
  return { ...entry, text: entry.text || entry.line || entry.message || "" };
}

export function App() {
  const initial = useMemo(() => previewState(), []);
  const [state, setState] = useState(() => normalizeState(initial));
  const [logs, setLogs] = useState(() => initial.logs || []);
  const [busyAction, setBusyAction] = useState("");
  const [failedStep, setFailedStep] = useState("unknown");
  const [feedback, setFeedback] = useState(null);
  const [copied, setCopied] = useState(false);
  const [upgradeReview, setUpgradeReview] = useState(false);
  const [cleanupReview, setCleanupReview] = useState(null);
  const [runtimeManager, setRuntimeManager] = useState(null);
  const [actionError, setActionError] = useState("");
  const [cudaError, setCudaError] = useState("");
  const [checkingCuda, setCheckingCuda] = useState(false);
  const stateRef = useRef(state);
  const busyRef = useRef("");
  const cudaRevision = useRef(0);
  const checkInFlight = useRef(0);
  const checkSequence = useRef(0);
  const projectEpoch = useRef(0);
  const cleanupSeen = useRef(new Set());
  const feedbackTrigger = useRef(null);
  stateRef.current = state;
  const mounted = useRef(true);

  const commitState = useCallback((update) => {
    const current = stateRef.current;
    const next = typeof update === "function" ? update(current) : update;
    if (!sameProject(projectDirectory(current), projectDirectory(next))) {
      projectEpoch.current += 1;
      checkSequence.current += 1;
      checkInFlight.current = 0;
      setCheckingCuda(false);
      setCudaError("");
      setActionError("");
    }
    stateRef.current = next;
    setState(next);
  }, []);

  useEffect(() => {
    mounted.current = true;
    const offState = launcherBridge.on("state", (next) => {
      cudaRevision.current += 1;
      commitState((current) => normalizeState({ ...current, ...next }));
    });
    const offProgress = launcherBridge.on("progress", (progress) => {
      commitState((current) => ({
        ...current,
        runtimeItems: current.runtimeItems.map((item) => item.id === progress.id ? { ...item, ...progress } : item),
      }));
    });
    const offLog = launcherBridge.on("log", (entry) => {
      setLogs((current) => [...current.slice(-499), normalizeLog(entry)]);
    });
    const offCuda = launcherBridge.on("cudaRuntime", (patch) => {
      if (!patch || (launcherBridge.isNative && !patch.projectDir)
        || (patch.projectDir && !sameProject(patch.projectDir, projectDirectory(stateRef.current)))) return;
      cudaRevision.current += 1;
      commitState((current) => ({ ...current, cudaRuntime: scopeCudaRuntime({ ...current.cudaRuntime, ...patch }, projectDirectory(current)) }));
    });

    const initialRevision = cudaRevision.current;
    launcherBridge.request("getState").then((result) => {
      if (!mounted.current || !result || cudaRevision.current !== initialRevision) return;
      const next = result.state || result;
      commitState((current) => normalizeState({ ...current, ...next }));
      if (Array.isArray(next.logs)) setLogs(next.logs.map(normalizeLog));
    }).catch((error) => {
      setLogs((current) => [...current, { level: "error", text: error.message }]);
    });

    return () => {
      mounted.current = false;
      offState();
      offProgress();
      offLog();
      offCuda();
    };
  }, [commitState]);

  const refreshCuda = useCallback(async (manual = false) => {
    if (!mounted.current || checkInFlight.current || busyRef.current || document.hidden) return false;
    const current = stateRef.current;
    if (cudaActionBlocked(current, false) || CUDA_WORKING_STATES.has(current.cudaRuntime?.status)) return false;
    if (current.mode !== "ready" && !current.projectReady) return false;
    const project = projectDirectory(current);
    const epoch = projectEpoch.current;
    const revision = cudaRevision.current;
    const requestId = ++checkSequence.current;
    checkInFlight.current = requestId;
    setCheckingCuda(true);
    try {
      const result = await launcherBridge.request("checkCudaRuntime");
      if (!mounted.current || busyRef.current || revision !== cudaRevision.current
        || epoch !== projectEpoch.current || !sameProject(project, projectDirectory(stateRef.current))
        || (result?.projectDir && !sameProject(result.projectDir, project))) return false;
      if (result && Object.hasOwn(result, "cudaRuntime")) {
        if (result.cudaRuntime?.projectDir && !sameProject(result.cudaRuntime.projectDir, project)) return false;
        cudaRevision.current += 1;
        commitState((value) => ({ ...value, cudaRuntime: scopeCudaRuntime(result.cudaRuntime, project) }));
        setCudaError("");
      }
      return true;
    } catch (error) {
      // Keep an actionable previous result; an unavailable first check is honest.
      if (mounted.current && epoch === projectEpoch.current && revision === cudaRevision.current && !stateRef.current.cudaRuntime) {
        commitState((value) => ({ ...value, cudaRuntime: scopeCudaRuntime({ status: "unavailable", message: "环境检查暂时不可用，请重新检测。" }, project) }));
      }
      if (manual === true && mounted.current && epoch === projectEpoch.current && revision === cudaRevision.current) {
        setCudaError(error?.message || "环境检查未完成，请重试。上次检测结果继续保留。");
      }
      return false;
    } finally {
      if (checkInFlight.current === requestId) {
        checkInFlight.current = 0;
        if (mounted.current) setCheckingCuda(false);
      }
    }
  }, [commitState]);

  useEffect(() => {
    const timer = window.setInterval(refreshCuda, 20000);
    const visibility = () => { if (!document.hidden) refreshCuda(); };
    window.addEventListener("focus", refreshCuda);
    document.addEventListener("visibilitychange", visibility);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", refreshCuda); document.removeEventListener("visibilitychange", visibility); };
  }, [refreshCuda]);

  useEffect(() => {
    const runtime = state.cudaRuntime;
    if (runtime?.status !== "cleanup-choice" || runtime.canRecover || !runtime.transactionId || busyAction || feedback || upgradeReview || cleanupReview || runtimeManager) return;
    const key = (state.projectDir || state.installPath) + ":" + runtime.transactionId;
    if (cleanupSeen.current.has(key)) return;
    cleanupSeen.current.add(key);
    setActionError("");
    setCudaError("");
    setCleanupReview({ ...cleanupRuntime(runtime), projectKey: projectDirectory(state) });
  }, [state.cudaRuntime, state.projectDir, state.installPath, busyAction, feedback, upgradeReview, cleanupReview, runtimeManager]);

  useEffect(() => {
    if (cleanupReview && (!sameProject(cleanupReview.projectKey, projectDirectory(state))
      || state.cudaRuntime?.canRecover || (!busyAction && !cleanupRuntime(state.cudaRuntime, cleanupReview.transactionId)))) {
      setCleanupReview(null);
    }
    if (runtimeManager && (!sameProject(runtimeManager.projectKey, projectDirectory(state)) || state.cudaRuntime?.canRecover)) setRuntimeManager(null);
  }, [cleanupReview, runtimeManager, state.projectDir, state.installPath, state.cudaRuntime, busyAction]);

  const runAction = useCallback(async (method, parameters = {}) => {
    if (busyRef.current) return false;
    if (CUDA_MUTATIONS.has(method) && method !== "keepCudaRuntime" && cudaActionBlocked(stateRef.current, false)) return false;
    const project = projectDirectory(stateRef.current);
    const epoch = projectEpoch.current;
    if (CUDA_MUTATIONS.has(method) && !sameProject(parameters.expectedProjectDir, project)) return false;
    if (method === "prepareFeedback") feedbackTrigger.current = document.activeElement;
    busyRef.current = method;
    cudaRevision.current += 1;
    setBusyAction(method);
    setActionError("");
    if (CUDA_MUTATIONS.has(method)) setCudaError("");
    try {
      const result = await launcherBridge.request(method, {
        installPath: stateRef.current.installPath,
        mirror: stateRef.current.mirror,
        ...parameters,
      });
      if (!mounted.current || (CUDA_MUTATIONS.has(method) && (epoch !== projectEpoch.current || !sameProject(project, projectDirectory(stateRef.current))))) return false;
      if (method === "prepareFeedback") { setFeedback(result); setCopied(false); return true; }
      if (result && result.state) {
        commitState((current) => normalizeState({ ...current, ...result.state }));
      } else if (result && typeof result === "object") {
        commitState((current) => normalizeState({ ...current, ...result }));
      }
      return true;
    } catch (error) {
      if (!mounted.current || (CUDA_MUTATIONS.has(method) && epoch !== projectEpoch.current)) return false;
      setActionError(error?.message || "操作未完成，请重试。");
      if (CUDA_MUTATIONS.has(method)) setCudaError(error?.message || "运行环境操作未完成，请重试。");
      setFailedStep(method.includes("Update") ? "upgrade" : method.includes("repair") ? "repair" : method.includes("WebUi") ? "start" : "install");
      setLogs((current) => [...current, {
        level: "error",
        time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
        tag: "ERROR",
        text: error && error.message ? error.message : "操作未完成，请查看终端详情。",
      }]);
      return false;
    } finally {
      busyRef.current = "";
      if (mounted.current) setBusyAction("");
    }
  }, [commitState]);

  const onAction = useCallback((method, parameters = {}) => {
    if (method === "checkCudaRuntime") return refreshCuda(true);
    if (method === "reviewUpgrade") { setActionError(""); setUpgradeReview(true); return; }
    return runAction(method, parameters);
  }, [runAction, refreshCuda]);

  const reviewCleanup = useCallback((runtime, transactionId) => {
    const selected = cleanupRuntime(stateRef.current.cudaRuntime, transactionId || runtime.transactionId);
    if (!selected || !sameProject(runtime.projectDir, projectDirectory(stateRef.current))) return;
    setActionError("");
    setRuntimeManager(null);
    setCleanupReview({ ...selected, projectKey: runtime.projectDir });
  }, []);
  const finishCleanup = async (remove) => {
    if (!cleanupReview || !cleanupReview.transactionId) return;
    const live = cleanupRuntime(stateRef.current.cudaRuntime, cleanupReview.transactionId);
    if (!live || (!remove && live.deleting) || (remove && !live.canDelete)) return;
    const succeeded = await runAction(remove ? "deleteCudaRuntime" : "keepCudaRuntime", {
      transactionId: cleanupReview.transactionId,
      expectedProjectDir: cleanupReview.projectKey,
      ...(remove ? { confirmed: true } : {}),
    });
    if (succeeded) setCleanupReview(null);
  };
  const liveCleanup = cleanupRuntime(state.cudaRuntime, cleanupReview?.transactionId);

  const content = state.mode === "install"
    ? (
      <InstallView
        state={state}
        logs={logs}
        busy={Boolean(busyAction) || state.environmentStatus === "changing" || Boolean(state.cudaRuntime?.canRecover) || CUDA_WORKING_STATES.has(state.cudaRuntime?.status)}
        actionError={actionError}
        onAction={onAction}
        onClearLogs={() => setLogs([])}
      />
    )
    : <ReadyView state={state} logs={logs} busyAction={busyAction} onAction={onAction} />;

  return (
    <div className={"launcher-shell mode-" + state.mode}>
      <TitleBar state={state} />
      <div className="launcher-workspace">
        <CudaRuntimeNotice state={state} busyAction={busyAction} checking={checkingCuda}
          error={cudaError}
          onAction={onAction} onReviewCleanup={reviewCleanup} onManageRetained={(runtime) => setRuntimeManager({ projectKey: runtime.projectDir })} />
        {content}
      </div>
      <div className="feedback-entry"><button disabled={Boolean(busyAction)} onClick={() => runAction("prepareFeedback", { step: failedStep })}>准备反馈信息</button><small>预览后复制，自行提交 Issue</small><details><summary>应用 {state.release?.applicationVersion || "未验证"} · 版本来源</summary><p>安装来源：{state.release?.installationSource || "未验证"}</p><p>源码快照：{state.release?.sourceCommit || "未验证"}</p></details></div>
      {upgradeReview && <Dialog title="升级到启动器固定的应用版本" busy={Boolean(busyAction)} onClose={() => setUpgradeReview(false)} actions={<>
        <button data-autofocus disabled={Boolean(busyAction)} onClick={() => setUpgradeReview(false)}>取消</button>
        <button disabled={Boolean(busyAction) || state.webuiRunning || state.terminalRunning} onClick={async () => { if (await runAction("applyUpdate")) setUpgradeReview(false); }}>备份并升级</button>
      </>}>
        <p>目标应用：{state.upgradeTarget?.applicationVersion || "未验证"}</p><p className="launcher-source-hash">公开提交：{state.upgradeTarget?.sourceCommit || "未固定"}<br />源码 ZIP SHA-256：{state.upgradeTarget?.archiveSha256 || "未固定"}</p>
        <p>从新版启动器内的来源记录下载固定源码并校验 SHA-256。升级前备份源码、当前界面构建和依赖；失败时自动回退。素材、模型、配置和运行环境保留在本地。</p>
        <p>请先保存并结束所有任务、传统终端和 WebUI。Git 工作副本请使用已审查的公开固定提交升级。</p>
        {actionError ? <p className="dialog-error" role="alert">{actionError}</p> : null}
      </Dialog>}
      {feedback && <Dialog title="反馈信息预览" returnFocus={feedbackTrigger.current} onClose={() => setFeedback(null)} description="仅准备本地摘要。请检查后复制；不会自动上传日志、媒体、模型或 API Key。" actions={<>
        <button data-autofocus onClick={() => setFeedback(null)}>关闭</button>
        <button onClick={async () => { try { await navigator.clipboard.writeText(feedback.preview || ""); setCopied(true); } catch { setCopied(false); } }}>{copied ? "已复制" : "复制反馈摘要"}</button>
        <button onClick={() => runAction("openExternal", { url: feedback.issueUrl })}>打开 Issue 页面</button>
      </>}>
        <textarea className="resize-none" readOnly aria-label="反馈摘要" value={feedback.preview || ""} rows={13} />
      </Dialog>}
      {runtimeManager && !feedback && !upgradeReview && !cleanupReview ? <Dialog key="runtime-manager" title="本项目保留的 CUDA 运行包"
        onClose={() => setRuntimeManager(null)} description="这里只管理本项目保留的旧运行包。当前使用的环境、模型和素材会保留。"
        actions={<button data-autofocus onClick={() => setRuntimeManager(null)}>关闭</button>}>
        <ul className="runtime-retained-list">{state.cudaRuntime?.retainedRuntimes.map((row) => <li key={row.transactionId}>
          <div><strong>{runtimeLabel(row.runtime)}</strong><p>{formatRuntimeBytes(row.bytes)} · {row.deleting ? "清理尚未完成，只能继续删除" : row.kept ? "已保留" : "等待保留或清理"}</p></div>
          <button disabled={Boolean(busyAction)} onClick={() => reviewCleanup(state.cudaRuntime, row.transactionId)}>{row.deleting ? "继续清理" : "查看清理选项"}</button>
        </li>)}</ul>
        {!state.cudaRuntime?.retainedRuntimes.length ? <p>没有需要管理的旧运行包。</p> : null}
      </Dialog> : null}
      {cleanupReview && !feedback && !upgradeReview ? <Dialog key={"runtime-cleanup:" + cleanupReview.transactionId} title={liveCleanup?.deleting ? "继续删除未完成的旧运行包？" : "是否删除原 CUDA 运行包？"}
        busy={Boolean(busyAction)} onClose={() => setCleanupReview(null)} className="runtime-cleanup-dialog"
        description={liveCleanup?.deleting ? "上次清理已经开始，旧运行包可能不完整，不能再次标记为保留。确认后继续清理剩余内容。" : `${runtimeLabel(cleanupReview.currentRuntime, cleanupReview.currentLabel)} 是当前使用的版本。你可以保留原版本，以便以后更换显卡时再次使用。`}
        actions={<>
          {liveCleanup?.deleting ? <button data-autofocus className="runtime-button" disabled={Boolean(busyAction)} onClick={() => setCleanupReview(null)}>稍后处理</button>
            : <button data-autofocus className="runtime-button" disabled={Boolean(busyAction)} onClick={() => finishCleanup(false)}>{busyAction === "keepCudaRuntime" ? "正在保存选择…" : "保留旧版"}</button>}
          <button className="runtime-button is-danger" disabled={cudaActionBlocked(state, busyAction) || !liveCleanup?.canDelete} onClick={() => finishCleanup(true)}>{busyAction === "deleteCudaRuntime" ? "正在删除旧版…" : liveCleanup?.deleting ? "继续删除旧版" : "删除旧版"}</button>
        </>}>
        <dl className="runtime-cleanup-facts"><div><dt>准备清理</dt><dd>{runtimeLabel(cleanupReview.retainedRuntime)}</dd></div><div><dt>{liveCleanup?.deleting ? "原运行包大小" : "可释放空间"}</dt><dd>{formatRuntimeBytes(cleanupReview.retainedBytes)}</dd></div></dl>
        <p>只清理本项目中选中的旧版 CUDA 运行包，模型、素材、训练进度和当前正在使用的版本都会保留。不会卸载显卡驱动或系统 CUDA。删除后，再次使用原版本需要重新下载。</p>
        {state.webuiRunning || state.terminalRunning ? <p className="dialog-error">请先保存并结束所有任务、终端和 WebUI，再删除旧版。</p> : null}
        {!liveCleanup?.canDelete && !busyAction ? <p className="dialog-error">当前环境暂不满足清理条件，请重新检测；旧运行包保持原状。</p> : null}
        {actionError ? <p className="dialog-error" role="alert">{actionError}</p> : null}
      </Dialog> : null}
    </div>
  );
}
