import { useEffect, useRef, useState } from "react";
import { IconArrowRight, IconCheck, IconChevronLeft, IconChevronRight, IconPlayerStop, IconRefresh } from "@tabler/icons-react";
import { runtimeApi } from "../runtime/api.js";
import { useI18n } from "../i18n.jsx";
import { LoadingProgress } from "./ProgressFeedback.jsx";
import { BestFacesetPanel } from "./BestFacesetPanel.jsx";
import "./quality-pipeline.css";

const PAGE_SIZE = 500;
const ACTIVE = new Set(["queued", "running"]);
const STATUS_LABELS = { queued: "等待处理", running: "正在处理", completed: "已完成", cancelled: "已取消", failed: "处理失败", interrupted: "已中断", ready: "待复核", published: "已保存副本" };
const MODEL_LABELS = { "swinir-psnr": "SwinIR-L PSNR", "realesrgan-x4plus": "Real-ESRGAN x4plus", "gfpgan-v1.4": "GFPGAN 1.4" };

export function QualityPipelinePanel({ side, projectKey, refreshVersion, onError, onNotice, onOpenCommand, onUseTrainingInput, initialPlanId, initialPlanRequest, initialMode, initialDraftId }) {
  const { t } = useI18n();
  const [mode, setMode] = useState("best");
  const [bestBusy, setBestBusy] = useState(false);
  const [inventory, setInventory] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [offset, setOffset] = useState(0);
  const [modelId, setModelId] = useState("swinir-psnr");
  const [tasks, setTasks] = useState([]);
  const [task, setTask] = useState(null);
  const [drafts, setDrafts] = useState([]);
  const [draft, setDraft] = useState(null);
  const [receipt, setReceipt] = useState(null);
  const [busy, setBusy] = useState(null);
  const [progress, setProgress] = useState(null);
  const [reviewed, setReviewed] = useState(false);
  const [revision, setRevision] = useState(0);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [loadError, setLoadError] = useState(null);
  const version = useRef(0);
  const draftRequest = useRef(0);
  const operation = useRef(null);
  useEffect(() => {
    if (["best", "masks", "restoration"].includes(initialMode)) setMode(initialMode);
  }, [initialMode, initialPlanRequest]);
  useEffect(() => {
    if (!initialDraftId || initialMode !== "masks" || mode !== "masks") return undefined;
    let cancelled = false;
    void runtimeApi.maskAssistDraft(side, initialDraftId).then(value => { if (!cancelled) { setDraft(value); setReviewed(false); setReceipt(null); } }).catch(error => { if (!cancelled) onError(error); });
    return () => { cancelled = true; };
  }, [side, initialDraftId, initialPlanRequest, initialMode, mode, onError]);

  useEffect(() => { draftRequest.current++; setOffset(0); setTask(null); setTasks([]); setDraft(null); setDrafts([]); setReceipt(null); setPreviewIndex(0); }, [side]);
  useEffect(() => { draftRequest.current++; setOffset(0); setSelected(new Set()); setInventory(null); setPreviewIndex(0); }, [mode]);
  useEffect(() => {
    if (mode === "best") return undefined;
    const request = ++version.current;
    setInventory(null);
    setLoadError(null);
    setSelected(new Set());
    const pending = mode === "restoration"
      ? Promise.all([runtimeApi.restorationInputs(side, { offset, limit: PAGE_SIZE }), runtimeApi.restorationTasks()])
      : Promise.all([runtimeApi.alignedAssets(side, { offset, limit: PAGE_SIZE }), runtimeApi.maskAssistDrafts(side)]);
    pending.then(([next, history]) => {
      if (version.current !== request) return;
      setInventory(next);
      if (mode === "restoration") setTasks(history.filter(item => item.side === side));
      else setDrafts(history);
    }).catch(error => { if (version.current === request) { setLoadError(error.message); onError(error); } });
    return () => { version.current++; };
  }, [side, mode, offset, revision, refreshVersion, onError]);

  useEffect(() => {
    if (!task || !ACTIVE.has(task.status)) return undefined;
    let disposed = false;
    let timer;
    const poll = async () => {
      try {
        const next = await runtimeApi.restorationTask(task.taskId);
        if (disposed) return;
        setTask(next);
        if (!ACTIVE.has(next.status)) setRevision(value => value + 1);
      } catch (error) { if (!disposed) onError(error); }
      if (!disposed) timer = setTimeout(poll, 1500);
    };
    timer = setTimeout(poll, 1000);
    return () => { disposed = true; clearTimeout(timer); };
  }, [task?.taskId, task?.status, onError]);

  const items = inventory?.inputs ?? inventory?.items ?? [];
  const total = inventory?.total ?? 0;
  const active = Boolean(bestBusy || busy || task && ACTIVE.has(task.status));
  const modelAvailable = inventory?.models?.some(item => item.id === modelId && item.available && item.sourceFramesSupported) ?? false;
  const statusLabel = status => t(STATUS_LABELS[status] ?? "状态未知");
  const modelLabel = id => MODEL_LABELS[id] ?? t("未知模型");
  const toggle = name => setSelected(current => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });
  const run = async () => {
    if (!selected.size || mode === "restoration" && !modelAvailable) return;
    draftRequest.current++;
    setBusy(mode); setProgress(null); setReviewed(false); setReceipt(null);
    try {
      if (mode === "restoration") {
        setPreviewIndex(0);
        setTask(await runtimeApi.createRestoration({ side, names: [...selected], fingerprint: inventory.fingerprint, offset, limit: PAGE_SIZE, modelId }));
        onNotice(t("修复已开始，完成后可以重新提取副本人脸。"));
      } else {
        const next = await runtimeApi.createMaskAssist(side, [...selected], { onProgress: value => { operation.current = value.id; setProgress(value); } });
        setDraft(next); setPreviewIndex(0); setRevision(value => value + 1);
        onNotice(t("辅助遮罩已生成，请查看原图、遮罩与边界。"));
      }
    } catch (error) { onError(error); }
    finally { setBusy(null); operation.current = null; }
  };
  const publish = async () => {
    setBusy("publish"); setProgress(null);
    try {
      const next = await runtimeApi.publishMaskAssist(side, draft.id, { onProgress: value => { operation.current = value.id; setProgress(value); } });
      setReceipt(next); setDraft(current => ({ ...current, publication: next })); setRevision(value => value + 1);
      onNotice(t("完整遮罩数据集副本已保存，原 aligned 保留。"));
    } catch (error) { onError(error); }
    finally { setBusy(null); operation.current = null; }
  };
  const cancel = async () => {
    try {
      if (operation.current) await runtimeApi.cancelOperation(operation.current);
      else if (task && ACTIVE.has(task.status)) setTask(await runtimeApi.cancelRestoration(task.taskId));
    } catch (error) { onError(error); }
  };
  const loadDraft = async id => {
    const request = ++draftRequest.current;
    try {
      const next = await runtimeApi.maskAssistDraft(side, id);
      if (request !== draftRequest.current) return;
      setDraft(next); setPreviewIndex(0); setReviewed(false); setReceipt(next.publication ?? null);
    } catch (error) { if (request === draftRequest.current) onError(error); }
  };
  const entry = draft?.entries?.[previewIndex];
  const output = task?.outputs?.[previewIndex];
  const maskUrl = kind => `/api/mask-assist/${side}/drafts/${draft.id}/files/${previewIndex}/${kind}`;
  return <section className="quality-pipeline-panel">
    <header><div><h3>{t("质量方案")}</h3><p>{t("选择合适的人脸训练、修复源帧或准备遮罩；分析依据可在各方案的高级信息中查看。")}</p></div>
      <button type="button" className="button secondary" onClick={() => onOpenCommand(`${side}.extract_faces`)}>{t("按推荐方案提取")}<IconArrowRight size={15}/></button>
    </header>
    <div className="advanced-mode-switch" role="tablist" aria-label={t("质量处理环节")}>
      <button type="button" role="tab" aria-selected={mode === "best"} className={mode === "best" ? "is-active" : ""} disabled={active} onClick={() => setMode("best")}>{t("最佳训练人脸")}</button>
      <button type="button" role="tab" aria-selected={mode === "restoration"} className={mode === "restoration" ? "is-active" : ""} disabled={active} onClick={() => setMode("restoration")}>{t("源帧修复")}</button>
      <button type="button" role="tab" aria-selected={mode === "masks"} className={mode === "masks" ? "is-active" : ""} disabled={active} onClick={() => setMode("masks")}>{t("辅助遮罩")}</button>
    </div>
    {mode === "best" ? <BestFacesetPanel side={side} projectKey={projectKey} initialPlanId={initialPlanId} initialPlanRequest={initialPlanRequest} refreshVersion={refreshVersion} onError={onError} onNotice={onNotice} onOpenCommand={onOpenCommand} onUseTrainingInput={onUseTrainingInput} onBusyChange={setBestBusy}/> : <>
    <p className="quality-pipeline-note">{mode === "restoration"
      ? t("修复输出保持原帧尺寸，保存为新 PNG 副本。完成后重新提取，生成独立 aligned 数据集；原帧与原人脸保留。")
      : t("BiSeNet 生成面部遮罩草稿，保留眼、口和耳部类别。确认后保存完整 aligned 副本；人工标注与通用 XSeg 保留。")}</p>
    <div className="quality-pipeline-controls">
      {mode === "restoration" && <label>{t("修复模型")}<select value={modelId} disabled={active || !inventory} onChange={event => setModelId(event.target.value)}>{["swinir-psnr", "realesrgan-x4plus"].map(id => {
        const available = inventory?.models?.some(item => item.id === id && item.available && item.sourceFramesSupported) ?? false;
        return <option key={id} value={id} disabled={!available}>{modelLabel(id)}{id === "swinir-psnr" ? ` · ${t("推荐")}` : ""}{!available ? ` · ${t("资源尚未准备")}` : ""}</option>;
      })}</select></label>}
      <span>{t("本轮最多 500 张")} · {t("范围 {start}–{end} / {total}", { start: items.length ? offset + 1 : 0, end: offset + items.length, total })} · {t("已选择 {count} 张", { count: selected.size })}</span>
      <button type="button" className="button secondary" disabled={active || !items.length} onClick={() => setSelected(selected.size === items.length ? new Set() : new Set(items.map(item => item.name)))}>{t("选择本批全部")}</button>
      <button type="button" className="button primary" disabled={active || !selected.size || mode === "restoration" && !modelAvailable} onClick={() => void run()}>{mode === "restoration" ? t("生成修复副本") : t("生成遮罩草稿")}</button>
      {active && <button type="button" className="button secondary" onClick={() => void cancel()}><IconPlayerStop size={15}/>{t("安全取消")}</button>}
    </div>
    {loadError ? <div role="alert"><p>{t("素材读取失败，请重试。")} {loadError}</p><button type="button" className="button secondary" onClick={() => setRevision(value => value + 1)}><IconRefresh size={15}/>{t("重试")}</button></div> : !inventory ? <LoadingProgress compact label={t("读取本批素材…")}/> : !items.length ? <p role="status">{mode === "restoration" ? t("先提取视频帧，再进行源帧修复。") : t("先提取 aligned 人脸，再生成辅助遮罩。")}</p> :
      <div className="quality-pipeline-files">{items.map(item => <label key={item.name}><input type="checkbox" checked={selected.has(item.name)} disabled={active} onChange={() => toggle(item.name)}/><span title={item.name}>{item.name}</span></label>)}</div>}
    <div className="quality-pipeline-controls"><button type="button" className="button secondary" disabled={active || offset === 0} onClick={() => setOffset(value => Math.max(0, value - PAGE_SIZE))}><IconChevronLeft size={15}/>{t("上一批")}</button><button type="button" className="button secondary" disabled={active || offset + items.length >= total} onClick={() => setOffset(value => value + PAGE_SIZE)}>{t("下一批")}<IconChevronRight size={15}/></button><button type="button" className="button secondary" disabled={active} onClick={() => setRevision(value => value + 1)}><IconRefresh size={15}/>{t("刷新")}</button></div>
    {busy && <LoadingProgress compact label={t(progress?.stage ?? "正在处理…")} current={progress?.current} total={progress?.total} detail={t(progress?.detail)}/>}
    {mode === "restoration" && <div className="quality-pipeline-results">
      {tasks.length > 0 && <label>{t("修复记录")}<select value={task?.taskId ?? ""} disabled={active} onChange={event => { setTask(tasks.find(item => item.taskId === event.target.value) ?? null); setPreviewIndex(0); }}><option value="">{t("选择记录")}</option>{tasks.map(item => <option key={item.taskId} value={item.taskId}>{modelLabel(item.modelId)} · {t("已选择 {count} 张", { count: item.selectedCount })} · {statusLabel(item.status)}</option>)}</select></label>}
      {task && <p role="status">{t("修复状态")}：{statusLabel(task.status)} · {task.progress?.completed ?? 0}/{task.selectedCount}{task.error && ` · ${task.error}`}</p>}
      {task?.status === "completed" && <><label>{t("查看副本")}<select value={previewIndex} onChange={event => setPreviewIndex(Number(event.target.value))}>{task.outputs.map((item, i) => <option key={item.name} value={i}>{item.sourceName}</option>)}</select></label>{output && <img className="quality-pipeline-preview" src={`/api/restoration/tasks/${task.taskId}/images/${encodeURIComponent(output.name)}`} alt={t("修复输出副本")}/>}<button type="button" className="button primary" onClick={() => onOpenCommand(`${side}.extract_restored`, { restorationTaskId: task.taskId })}>{t("重新提取修复副本")}<IconArrowRight size={15}/></button><p>{t("提取结果保存为独立数据集，可在训练配置中选择；不会覆盖原 aligned。")}</p></>}
    </div>}
    {mode === "masks" && <div className="quality-pipeline-results">
      {drafts.length > 0 && <label>{t("遮罩草稿记录")}<select value={draft?.id ?? ""} disabled={active} onChange={event => { if (event.target.value) void loadDraft(event.target.value); else { draftRequest.current++; setDraft(null); setReceipt(null); } }}><option value="">{t("选择记录")}</option>{drafts.map(item => <option key={item.id} value={item.id}>{item.id} · {t("已选择 {count} 张", { count: item.selectedCount })} · {statusLabel(item.status)}</option>)}</select></label>}
      {draft && <><label>{t("复核图片")}<select value={previewIndex} onChange={event => { setPreviewIndex(Number(event.target.value)); setReviewed(false); }}>{draft.entries.map((item, i) => <option key={item.file ?? item.name} value={i}>{item.file ?? item.name}</option>)}</select></label>{entry && <div className="quality-pipeline-comparison">{["original", "overlay", "mask"].map((kind, i) => <figure key={kind}><img src={maskUrl(kind)} alt={t(["原图", "边界预览", "生成遮罩"][i])}/><figcaption>{t(["原图", "边界预览", "生成遮罩"][i])}</figcaption></figure>)}</div>}<label className="quality-pipeline-reviewed"><input type="checkbox" checked={reviewed} disabled={active} onChange={event => setReviewed(event.target.checked)}/>{t("已复核本批遮罩，保存完整数据集副本")}</label><button type="button" className="button primary" disabled={!reviewed || active} onClick={() => void publish()}><IconCheck size={15}/>{t("保存完整遮罩副本")}</button></>}
      {receipt && <div className="quality-pipeline-receipt"><strong>{t("新数据集已准备")}</strong><code>{receipt.facesetPath}</code><span>{t("原始数据保留，未选择的图片也包含在新副本中。")}</span><button type="button" className="button secondary" onClick={() => onOpenCommand("train.me", { [`${side}Faceset`]: receipt.facesetPath })}>{t("在训练配置中使用副本")}<IconArrowRight size={15}/></button>{side === "dst" && <><button type="button" className="button secondary" onClick={() => onOpenCommand("merge.me", { dstFaceset: receipt.facesetPath })}>{t("在合成配置中使用副本")}<IconArrowRight size={15}/></button><button type="button" className="button secondary" disabled={!receipt.reviewedMergeAvailable} onClick={() => onOpenCommand("merge.me", { dstFaceset: receipt.facesetPath, maskMode: 10 })}>{t("使用复核遮罩合成")}<IconArrowRight size={15}/></button><span>{receipt.reviewedMergeAvailable ? t("全部人脸已绑定原始帧，合成模式 10 会直接使用复核遮罩。") : t(receipt.reviewedMergeReason ?? "此记录未绑定全部原始帧，请重新生成并复核完整批次。")}</span></>}</div>}
    </div>}
    </>}
  </section>;
}
