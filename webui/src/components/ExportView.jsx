import { useI18n } from "../i18n.jsx";
import { OutputGallery } from "./OutputGallery.jsx";
import { CommandRows } from "./OperationsView.jsx";
import { activeJobStates, jobPresentation } from "../domain/job-presentation.js";
import { getWorkflowArtifactState } from "../domain/workflow-readiness.js";
import "./WorkflowGuidance.css";

export function ExportView({ workspace, commands, jobs, onOpenCommand, onError, onNotice, onOpenJob }) {
  const { t } = useI18n();
  const job = jobs.find(job => job.commandId.startsWith("encode.") && activeJobStates.has(job.state))
    ?? jobs.find(job => job.commandId.startsWith("encode."));
  const presentation = jobPresentation(job);
  const artifacts = getWorkflowArtifactState(workspace);
  const canExport = Boolean(workspace && artifacts.exportConfiguredReady && !activeJobStates.has(job?.state));
  const exportBlockReason = !workspace ? t("正在读取合成产物") : artifacts.partialMerge
    ? t("合成帧与遮罩数量不一致，请补齐并检查合成结果") : !artifacts.mergedAvailable
      ? t("需要先生成合成帧与遮罩序列") : activeJobStates.has(job?.state) ? t("导出任务正在运行") : "";
  return <section className="operation-view export-view">
    <header className="operation-header"><div><h2>{t("视频导出")}</h2><p>{t("先检查成片，再按需要重新导出。")}</p></div>
      {job ? <button type="button" className={`button secondary is-${presentation.tone}`} onClick={() => onOpenJob(job.id)}>{t(presentation.label)} · {t("查看任务日志")}</button> : null}
    </header>
    {workspace ? <OutputGallery workspace={workspace} jobs={jobs} onError={onError} onNotice={onNotice} onOpenJob={onOpenJob}/> : <p role="status">{t("正在读取项目成果…")}</p>}
    {workspace ? <section className="workflow-guidance" aria-label={t("视频导出条件")}>
      <strong>{t(artifacts.partialMerge ? "合成序列需要补齐" : artifacts.mergedAvailable ? "已发现合成输入" : "尚未发现合成输入")}</strong>
      <p>{artifacts.countsKnown ? t("合成帧 {frames} 张 · 遮罩 {masks} 张", { frames: artifacts.mergedCount.toLocaleString(), masks: artifacts.maskCount.toLocaleString() }) : t("输入数量将在任务预检中读取。")}</p>
      <p>{t(artifacts.exportConfiguredReady ? "数量检查通过；启动前仍会检查逐帧对应关系、音轨与输出位置。" : artifacts.encodedAvailable ? "已有成片可继续查看；重新导出需要可用的合成序列。" : "导出将合成序列封装为视频，先配置合成并检查帧与遮罩。")}</p>
      {!artifacts.exportConfiguredReady ? <button type="button" className="button primary" onClick={() => onOpenCommand(artifacts.meAvailable ? "merge.me" : "train.me")}>{t(artifacts.meAvailable ? artifacts.partialMerge ? "检查并补齐合成" : "配置合成" : "查看模型与训练配置")}</button> : null}
      {!artifacts.exportConfiguredReady && !artifacts.meAvailable ? <small>{t("未发现可用 ME 模型；可在训练配置中选择或建立模型，再生成合成序列。")}</small> : null}
      {workspace.materials?.dst ? <small>{t("音轨来源：DST 素材；具体音频与输出参数将在预检中确认。")}</small> : <small>{t("未发现 DST 原视频；是否能够无音轨导出，以当前任务预检为准。")}</small>}
    </section> : null}
    <section className="export-recommended"><div><strong>{t("RGB 母版 + MP4 · 推荐方案")}</strong><p>{t("保留逐帧校验的 RGB 母版，同时生成可播放的 MP4 和遮罩视频。母版适合保存，MP4 适合播放和分享。")}</p></div>
      <button type="button" className="button primary" disabled={!canExport} title={exportBlockReason} onClick={() => onOpenCommand("encode.quality")}>{t("导出母版与播放版")}</button>
    </section>
    <details className="export-formats"><summary>{t("其他视频格式")}</summary><CommandRows commands={commands.filter(command => command.category === "encode" && command.id !== "encode.quality")} onOpenCommand={onOpenCommand} disabled={!canExport} disabledReason={exportBlockReason}/></details>
    <details className="export-formats"><summary>{t("模型导出（DeepFaceLive）")}</summary><CommandRows commands={commands.filter(command => command.category === "model")} onOpenCommand={onOpenCommand}/></details>
  </section>;
}
