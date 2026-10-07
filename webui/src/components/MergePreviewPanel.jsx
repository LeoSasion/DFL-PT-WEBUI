import { useEffect, useState } from "react";
import { IconArrowLeft, IconArrowRight, IconPhoto, IconRefresh } from "@tabler/icons-react";
import { useI18n } from "../i18n.jsx";
import { runtimeApi } from "../runtime/api.js";
import { jobPresentation } from "../domain/job-presentation.js";
import "./MergePreviewPanel.css";

export function MergePreviewPanel({ workspace, jobs = [], onOpenCommand, onOpenJob, onError }) {
  const { t } = useI18n();
  const previewJobs = jobs.filter(job => job.commandId === "merge.preview_me").sort((a, b) => (b.createdAt ?? "").localeCompare(a.createdAt ?? ""));
  const [selectedId, setSelectedId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [index, setIndex] = useState(0);
  const [retry, setRetry] = useState(0);
  const selectedJob = previewJobs.find(job => job.id === selectedId) ?? previewJobs[0];
  const presentation = jobPresentation(selectedJob);
  useEffect(() => {
    let cancelled = false;
    setResult(null); setError(null); setIndex(0);
    if (!selectedJob) return undefined;
    if (["queued", "starting", "running", "waiting_input", "stopping"].includes(selectedJob.state)) return undefined;
    void runtimeApi.mergePreview(selectedJob.id).then(value => { if (!cancelled) setResult(value); }).catch(nextError => {
      if (!cancelled) { setError(nextError); if (nextError.code !== "MERGE_PREVIEW_PENDING") onError?.(nextError); }
    });
    return () => { cancelled = true; };
  }, [selectedJob?.id, selectedJob?.state, retry, onError]);
  const item = result?.items[index];
  const finalParameters = result ? {
    ...result.preview.requestedParameters,
    forceModelName: result.preview.modelName,
    dstFaceset: result.preview.alignedPath,
    cpuOnly: result.preview.device?.cpuOnly ?? selectedJob?.parameters?.cpuOnly ?? false,
    gpuIndexes: result.preview.device?.gpuIndexes ?? selectedJob?.parameters?.gpuIndexes ?? "",
    previewJobId: result.jobId,
  } : {};
  const hasModel = Boolean(workspace?.readiness?.me || workspace?.models?.some(model => model.type?.toUpperCase() === "ME" && model.ready !== false));
  return <section className="merge-preview-panel" aria-label={t("独立合成小样")}>
    <header><div><h3><IconPhoto size={18}/>{t("先检查独立小样")}</h3><p>{t("明确选择 1–20 帧，使用本次合成参数生成独立结果与遮罩。检查后可将相同参数填入全片合成。")}</p></div>
      <button type="button" className="button primary" disabled={!hasModel} title={!hasModel ? t("需要先选择可用 ME 模型") : undefined} onClick={() => onOpenCommand("merge.preview_me")}>{t("配置合成小样")}</button>
    </header>
    <p className="merge-preview-policy">{t("小样保存在 .webui/merge-previews，不替换正式 merged 序列，也不能直接用于全片视频导出。")}</p>
    {previewJobs.length ? <div className="merge-preview-controls">
      <label><span>{t("查看小样任务")}</span><select value={selectedJob?.id ?? ""} onChange={event => setSelectedId(event.target.value)}>{previewJobs.map(job => <option key={job.id} value={job.id}>{new Date(job.createdAt).toLocaleString()} · {t("第 {start} 帧起，{count} 帧", { start: job.parameters?.frameStart, count: job.parameters?.frameCount })} · {t(jobPresentation(job).label)}</option>)}</select></label>
      <button type="button" className="button secondary" onClick={() => setRetry(value => value + 1)}><IconRefresh size={15}/>{t("刷新")}</button>
      {onOpenJob ? <button type="button" className="button secondary" onClick={() => onOpenJob(selectedJob.id)}>{t(presentation.label)} · {t("查看日志")}</button> : null}
    </div> : null}
    {error ? <p role="status">{t(error.code === "MERGE_PREVIEW_PENDING" ? "小样尚未生成可读取回执；任务完成后会更新，或查看日志。" : error.message)}</p> : null}
    {result ? <>
      <div className="merge-preview-summary"><strong>{t("小样范围：第 {start}–{end} 帧 / 共 {total} 帧", { start: result.preview.frameStart, end: result.preview.frameStart + result.preview.frameCount - 1, total: result.preview.totalSourceFrames })}</strong><span>{t("完整结果 {count} / {total}", { count: result.completeCount, total: result.items.length })}</span></div>
      <p className="merge-preview-policy">{result.outputLocation} · {t("这是局部样本，不能证明整段视频的合成质量。")}</p>
      {item ? <div className="merge-preview-triptych">{[{ label: "DST 原帧", url: item.sourceUrl }, { label: "小样合成", url: item.mergedUrl }, { label: "小样遮罩", url: item.maskUrl }].map(asset => <figure key={asset.label}><figcaption>{t(asset.label)}</figcaption>{asset.url ? <img src={asset.url} alt={`${t(asset.label)} · ${item.name}`}/> : <div>{t("尚未生成")}</div>}</figure>)}</div> : null}
      <div className="merge-preview-footer"><div className="merge-preview-stepper"><button type="button" aria-label={t("上一张小样")} disabled={index <= 0} onClick={() => setIndex(value => value - 1)}><IconArrowLeft size={16}/></button><span>{item?.name} · {result.items.length ? index + 1 : 0} / {result.items.length}</span><button type="button" aria-label={t("下一张小样")} disabled={index >= result.items.length - 1} onClick={() => setIndex(value => value + 1)}><IconArrowRight size={16}/></button></div>
        <button type="button" className="button primary" disabled={result.status !== "complete" || result.completeCount !== result.items.length} onClick={() => onOpenCommand("merge.me", finalParameters)}>{t("按相同参数配置全片合成")}</button>
      </div>
    </> : !selectedJob ? <p>{t("还没有独立合成小样。已有正式合成结果仍可在合成复核中查看。")}</p> : null}
  </section>;
}
