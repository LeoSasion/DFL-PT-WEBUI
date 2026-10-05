import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InstallView } from "./components/InstallView.jsx";
import { ReadyView } from "./components/ReadyView.jsx";
import { TitleBar } from "./components/TitleBar.jsx";
import { launcherBridge, previewState } from "./bridge.js";

function normalizeState(input) {
  const fallback = previewState();
  if (!input || typeof input !== "object") return fallback;
  return {
    ...fallback,
    ...input,
    steps: Array.isArray(input.steps) ? input.steps : fallback.steps,
    runtimeItems: Array.isArray(input.runtimeItems) ? input.runtimeItems : fallback.runtimeItems,
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
  const modalRef = useRef(null);
  useEffect(() => {
    if (!feedback && !upgradeReview) return undefined;
    const previous = document.activeElement;
    const dialog = modalRef.current;
    const controls = () => [...(dialog?.querySelectorAll('button:not(:disabled), textarea, a[href], input, select') ?? [])];
    (controls()[0] ?? dialog)?.focus();
    const keyboard = event => {
      if (event.key === "Escape") { event.preventDefault(); setFeedback(null); setUpgradeReview(false); return; }
      if (event.key !== "Tab") return;
      const items = controls();
      if (!items.length) { event.preventDefault(); dialog?.focus(); return; }
      const first = items[0]; const last = items[items.length - 1];
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", keyboard);
    return () => { document.removeEventListener("keydown", keyboard); previous?.focus?.(); };
  }, [Boolean(feedback), upgradeReview]);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const offState = launcherBridge.on("state", (next) => {
      setState((current) => normalizeState({ ...current, ...next }));
    });
    const offProgress = launcherBridge.on("progress", (progress) => {
      setState((current) => ({
        ...current,
        runtimeItems: current.runtimeItems.map((item) => item.id === progress.id ? { ...item, ...progress } : item),
      }));
    });
    const offLog = launcherBridge.on("log", (entry) => {
      setLogs((current) => [...current.slice(-499), normalizeLog(entry)]);
    });

    launcherBridge.request("getState").then((result) => {
      if (!mounted.current || !result) return;
      const next = result.state || result;
      setState((current) => normalizeState({ ...current, ...next }));
      if (Array.isArray(next.logs)) setLogs(next.logs.map(normalizeLog));
    }).catch((error) => {
      setLogs((current) => [...current, { level: "error", text: error.message }]);
    });

    return () => {
      mounted.current = false;
      offState();
      offProgress();
      offLog();
    };
  }, []);
  const runAction = useCallback(async (method, parameters = {}) => {
    if (busyAction) return false;
    setBusyAction(method);
    try {
      const result = await launcherBridge.request(method, {
        installPath: state.installPath,
        mirror: state.mirror,
        ...parameters,
      });
      if (method === "prepareFeedback") { setFeedback(result); setCopied(false); return true; }
      if (result && result.state) {
        setState((current) => normalizeState({ ...current, ...result.state }));
      } else if (result && typeof result === "object") {
        setState((current) => normalizeState({ ...current, ...result }));
      }
      return true;
    } catch (error) {
      setFailedStep(method.includes("Update") ? "upgrade" : method.includes("repair") ? "repair" : method.includes("WebUi") ? "start" : "install");
      setLogs((current) => [...current, {
        level: "error",
        time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
        tag: "ERROR",
        text: error && error.message ? error.message : "操作未完成，请查看终端详情。",
      }]);
      return false;
    } finally {
      if (mounted.current) setBusyAction("");
    }
  }, [busyAction, state.installPath, state.mirror]);

  const onAction = useCallback((method, parameters = {}) => {
    if (method === "reviewUpgrade") { setUpgradeReview(true); return; }
    return runAction(method, parameters);
  }, [runAction]);

  const content = state.mode === "install"
    ? (
      <InstallView
        state={state}
        logs={logs}
        busy={Boolean(busyAction)}
        onAction={onAction}
        onClearLogs={() => setLogs([])}
      />
    )
    : <ReadyView state={state} logs={logs} busyAction={busyAction} onAction={onAction} />;

  return (
    <div className={"launcher-shell mode-" + state.mode}>
      <TitleBar state={state} />
      {content}
      <div className="feedback-entry"><button disabled={Boolean(busyAction)} onClick={() => runAction("prepareFeedback", { step: failedStep })}>准备反馈信息</button><small>预览后复制，自行提交 Issue</small><details><summary>应用 {state.release?.applicationVersion || "未验证"} · 版本来源</summary><p>安装来源：{state.release?.installationSource || "未验证"}</p><p>源码快照：{state.release?.sourceCommit || "未验证"}</p></details></div>
      {upgradeReview && <div className="launcher-modal-backdrop"><section ref={modalRef} tabIndex={-1} className="launcher-modal" role="dialog" aria-modal="true" aria-labelledby="upgrade-title">
        <h2 id="upgrade-title">升级到启动器固定的应用版本</h2>
        <p>目标应用：{state.upgradeTarget?.applicationVersion || "未验证"}</p><p className="launcher-source-hash">公开提交：{state.upgradeTarget?.sourceCommit || "未固定"}<br />源码 ZIP SHA-256：{state.upgradeTarget?.archiveSha256 || "未固定"}</p>
        <p>从新版启动器内的来源记录下载固定源码并校验 SHA-256。升级前备份源码、当前界面构建和依赖；失败时自动回退。素材、模型、配置和运行环境保留在本地。</p>
        <p>请先保存并结束所有任务、传统终端和 WebUI。Git 工作副本请使用已审查的公开固定提交升级。</p>
        <button disabled={Boolean(busyAction) || state.webuiRunning || state.terminalRunning} onClick={async () => { setUpgradeReview(false); await runAction("applyUpdate"); }}>备份并升级</button>
        <button onClick={() => setUpgradeReview(false)}>取消</button>
      </section></div>}
      {feedback && <div className="launcher-modal-backdrop"><section ref={modalRef} tabIndex={-1} className="launcher-modal" role="dialog" aria-modal="true" aria-labelledby="feedback-title">
        <h2 id="feedback-title">反馈信息预览</h2><p>仅准备本地摘要。请检查后复制；不会自动上传日志、媒体、模型或 API Key。</p>
        <textarea readOnly aria-label="反馈摘要" value={feedback.preview || ""} rows={13} />
        <button onClick={async () => { try { await navigator.clipboard.writeText(feedback.preview || ""); setCopied(true); } catch { setCopied(false); } }}>{copied ? "已复制" : "复制反馈摘要"}</button>
        <button onClick={() => runAction("openExternal", { url: feedback.issueUrl })}>打开 Issue 页面</button>
        <button onClick={() => setFeedback(null)}>关闭</button>
      </section></div>}
    </div>
  );
}
