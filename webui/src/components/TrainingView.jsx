import { useEffect, useRef, useState } from "react";
import {
  IconAdjustmentsHorizontal,
  IconAlertTriangle,
  IconArchive,
  IconCamera,
  IconChartDots3,
  IconCheck,
  IconChevronDown,
  IconChevronUp,
  IconCircle,
  IconDeviceFloppy,
  IconFileAnalytics,
  IconFolderOpen,
  IconMovie,
  IconRefresh,
  IconRoute,
  IconShieldX,
  IconSparkles,
  IconUsersGroup,
} from "@tabler/icons-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LoadingProgress } from "./ProgressFeedback.jsx";
import { pipelineTasks as defaultPipelineTasks } from "../data/dashboard.js";
import { useI18n } from "../i18n.jsx";
import { jobPresentation } from "../domain/job-presentation.js";
import { getTrainingSaveStatus } from "../domain/workflow-readiness.js";
import "./WorkflowGuidance.css";
import {
  MAX_PREVIEW_REFRESH_SECONDS,
  MIN_PREVIEW_REFRESH_SECONDS,
  PREVIEW_REFRESH_STORAGE_KEY,
  parsePreviewRefreshSeconds,
  planTrainingPreviewDisplay,
  readPreviewRefreshSeconds,
} from "../domain/training-preview-refresh.js";

const taskIcons = {
  extract: IconMovie,
  src: IconUsersGroup,
  dst: IconUsersGroup,
  sort: IconAdjustmentsHorizontal,
  xseg: IconSparkles,
  me: IconFileAnalytics,
  diagnose: IconChartDots3,
  merge: IconRoute,
  export: IconFileAnalytics,
};

const activeTrainingStates = new Set(["queued", "starting", "running", "waiting_input"]);
const previewEligibleStates = new Set(["starting", "running", "waiting_input", "stopping"]);

function formatIteration(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return String(value);
}

function formatRate(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K/h`;
  return `${Math.round(value)}/h`;
}

function formatEta(seconds, t) {
  if (!Number.isFinite(seconds)) return t("未设目标");
  if (seconds <= 0) return t("已达目标");
  const minutes = Math.max(1, Math.round(seconds / 60));
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const remainingMinutes = minutes % 60;
  if (days) return t("{days}天 {hours}小时", { days, hours });
  if (hours) return t("{hours}小时 {minutes}分", { hours, minutes: remainingMinutes });
  return t("{minutes}分钟", { minutes: remainingMinutes });
}

export function PipelinePanel({ activeTask, tasks = defaultPipelineTasks, title = "当前流水线", description, onSelectTask }) {
  const { t } = useI18n();
  const [showCompleted, setShowCompleted] = useState(false);
  const hiddenCompletedCount = tasks.filter((task) => task.state === "done" && task.id !== activeTask).length;
  const currentIndex = Math.max(0, tasks.findIndex((task) => task.id === activeTask || task.state === "active"));
  const nextTaskId = tasks.find((task, index) => index > currentIndex && task.state === "waiting")?.id;
  const visibleTasks = tasks.filter((task) => showCompleted || task.state !== "done" || task.id === activeTask);
  return (
    <section className="panel pipeline-panel" aria-labelledby="pipeline-title">
      <div className="panel-heading">
        <h2 id="pipeline-title">{t(title)}</h2>
      </div>
      <div className="pipeline-list">
        {description ? <p className="training-entry-note">{t(description)}</p> : null}
        {hiddenCompletedCount ? (
          <button
            className="pipeline-completed-summary"
            type="button"
            aria-expanded={showCompleted}
            onClick={() => setShowCompleted((current) => !current)}
          >
            <span><IconCheck size={14} stroke={2.6} />{t("{count} 个上游步骤已完成", { count: hiddenCompletedCount })}</span>
            {showCompleted ? <IconChevronUp size={14} /> : <IconChevronDown size={14} />}
          </button>
        ) : null}
        {visibleTasks.map((task) => {
          const Icon = taskIcons[task.id] ?? IconFileAnalytics;
          const selected = activeTask === task.id;
          return (
            <button
              className={`pipeline-row ${selected ? "is-selected" : ""} ${task.id === nextTaskId ? "is-next" : ""} is-${task.tone} is-status-${task.state}`}
              key={task.id}
              type="button"
              onClick={() => onSelectTask(task)}
              title={`${task.index}. ${task.label} · ${task.time}`}
            >
              <Icon size={22} stroke={1.55} className="pipeline-icon" />
              <span className="pipeline-copy">
                <strong>{task.index}. {task.label}</strong>
                <small>{task.time}</small>
              </span>
              {task.state === "done" ? (
                <span className="state-mark done" aria-label={t("已完成")}>
                  <IconCheck size={13} stroke={2.8} />
                </span>
              ) : task.state === "active" ? (
                <span className="state-pulse" aria-label={t("运行中")} />
              ) : task.state === "failed" ? (
                <span className="state-mark failed" aria-label={t("失败")}>
                  <IconAlertTriangle size={13} stroke={2.2} />
                </span>
              ) : task.state === "unconfirmed" ? (
                <span className="state-mark failed" aria-label={t("本次保存未确认")}><IconAlertTriangle size={13} stroke={2.2}/></span>
              ) : task.state === "available" ? (
                <span className="state-mark waiting" aria-label={t("已检测到产物")}><IconFolderOpen size={13} stroke={2}/></span>
              ) : (
                <span className="state-mark waiting" aria-label={task.id === "xseg" ? t("可选") : task.id === "sort" ? t("待复核") : t("等待中")}>
                  <IconCircle size={11} stroke={2} />
                </span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{formatIteration(label)} iters</strong>
      <span>SRC {payload[0]?.value}</span>
      <span>DST {payload[1]?.value}</span>
    </div>
  );
}

function TrainingChart({ iteration, lossHistory, className = "" }) {
  const { language, t } = useI18n();
  const hasHistory = lossHistory.length > 0;
  return (
    <div className={`chart-block ${className} ${hasHistory ? "" : "is-empty"}`.trim()} aria-label={t("训练损失曲线")}>
      <div className="section-label-row">
        <strong>{t("训练损失曲线")}</strong>
        <span>{t("越低越好")}</span>
        <div className="chart-legend" aria-label={t("图例")}>
          <span><i className="legend-dot green" /> SRC</span>
          <span><i className="legend-dot amber" /> DST</span>
        </div>
      </div>
      <div className="chart-canvas">
        {!hasHistory ? (
          <div className="chart-empty">{t("训练输出出现迭代数据后，将在此绘制真实损失曲线")}</div>
        ) : null}
        {hasHistory ? (
          <ResponsiveContainer
            width="100%"
            height="100%"
            initialDimension={{ width: 640, height: 127 }}
          >
            <LineChart data={lossHistory} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#163126" strokeOpacity={0.62} vertical={false} />
            <XAxis
              axisLine={false}
              dataKey="iteration"
              minTickGap={34}
              tick={{ fill: "#738079", fontSize: 11 }}
              tickFormatter={formatIteration}
              tickLine={false}
            />
            <YAxis
              axisLine={false}
              domain={[0, "auto"]}
              tick={{ fill: "#738079", fontSize: 11 }}
              tickFormatter={(value) => Number(value.toPrecision(3))}
              tickLine={false}
              width={44}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#2ce39f", strokeOpacity: 0.35 }} />
            {iteration > 0 ? (
              <ReferenceLine
                x={iteration}
                stroke="#32e4a6"
                strokeDasharray="3 4"
                label={{
                  value: iteration.toLocaleString(language === "zh" ? "zh-CN" : "en-US"),
                  fill: "#b9f7dc",
                  fontSize: 11,
                  position: "insideTop",
                  dy: 4,
                }}
              />
            ) : null}
            <Line type="monotone" dataKey="gLoss" stroke="#2ce39f" strokeWidth={1.55} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="dLoss" stroke="#f3b83f" strokeWidth={1.45} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : null}
      </div>
    </div>
  );
}

function PreviewRefreshMenu({ intervalSeconds, onChange }) {
  const { t } = useI18n();
  const detailsRef = useRef(null);
  const [draft, setDraft] = useState(String(intervalSeconds));
  const [error, setError] = useState(false);

  const applyInterval = (event) => {
    event.preventDefault();
    const seconds = parsePreviewRefreshSeconds(draft);
    if (seconds === null) {
      setError(true);
      return;
    }
    onChange(seconds);
    setError(false);
    detailsRef.current.open = false;
  };

  return (
    <details className="preview-refresh-menu" ref={detailsRef}>
      <summary className="button compact secondary" title={t("设置训练预览自动刷新间隔")}>
        <IconRefresh size={14} stroke={1.9} />{t("自动刷新")}
        <IconChevronDown size={13} stroke={1.9} />
      </summary>
      <form className="preview-refresh-popover" onSubmit={applyInterval}>
        <label htmlFor="training-preview-refresh-seconds">{t("预览刷新间隔（秒）")}</label>
        <input
          id="training-preview-refresh-seconds"
          type="number"
          inputMode="numeric"
          min={MIN_PREVIEW_REFRESH_SECONDS}
          max={MAX_PREVIEW_REFRESH_SECONDS}
          step="1"
          required
          value={draft}
          onChange={(event) => { setDraft(event.target.value); setError(false); }}
        />
        {error ? <small role="alert">{t("请输入 5–3600 之间的整数秒数")}</small> : null}
        <p>{t("仅调整此浏览器显示预览的频率，不改变训练速度。")}</p>
        <button className="button compact primary" type="submit">{t("应用")}</button>
      </form>
    </details>
  );
}

function PreviewGrid({ refreshKey, previewUrl, refreshIntervalSeconds, trainingState, hasSavedModel, modelLoading }) {
  const { t } = useI18n();
  const [retryCount, setRetryCount] = useState(0);
  const [failedRequest, setFailedRequest] = useState(null);
  const trainingHasStarted = previewEligibleStates.has(trainingState);
  const showPreview = Boolean(previewUrl);
  const requestKey = `${previewUrl}:${refreshKey}:${retryCount}`;
  const previewFailed = failedRequest === requestKey;
  const imageUrl = previewUrl && retryCount > 0
    ? `${previewUrl}${previewUrl.includes("?") ? "&" : "?"}preview_retry=${retryCount}`
    : previewUrl;
  const emptyTitle = previewFailed ? t("训练预览暂时无法读取") : trainingHasStarted
    ? t("正在等待首张训练预览")
    : t("当前没有运行中的训练");
  const emptyDetail = previewFailed ? t("可重试读取；模型保存状态请以已保存模型和任务日志为准。") : trainingHasStarted
    ? t("Trainer 生成首张真实预览后会自动显示")
    : t("启动 ME 后，这里会显示 Trainer 生成的真实预览");
  return (
    <div className="preview-block" key={refreshKey}>
      <div className="preview-labels">
        <span>{trainingHasStarted ? t("实时训练预览") : t("上次训练预览")}</span>
        <small>{trainingHasStarted ? t("每 {seconds} 秒显示最新预览", { seconds: refreshIntervalSeconds }) : modelLoading ? t("正在读取模型…") : hasSavedModel ? t("历史结果，可继续训练") : t("历史预览不代表模型已保存")}</small>
      </div>
      {showPreview && !previewFailed ? (
        <div className="preview-assets is-live">
          <img
            key={requestKey}
            className="live-preview"
            src={imageUrl}
            alt={t("ME 最新训练预览")}
            decoding="async"
            onError={() => setFailedRequest(requestKey)}
          />
        </div>
      ) : (
        <div
          className={`preview-empty ${trainingHasStarted ? "is-waiting" : "is-inactive"}`}
          role="status"
          aria-live="polite"
          data-preview-state={trainingHasStarted ? "waiting" : "inactive"}
          data-preview-unavailable={previewFailed || undefined}
        >
          <IconFileAnalytics size={25} stroke={1.45} aria-hidden="true" />
          <strong>{emptyTitle}</strong>
          <span>{emptyDetail}</span>
          {previewFailed ? <button className="button compact secondary" type="button" onClick={() => setRetryCount(count => count + 1)}>
            <IconRefresh size={14} />{t("重试读取预览")}
          </button> : null}
        </div>
      )}
    </div>
  );
}

export function TrainingWorkspace({
  iteration,
  trainingState,
  previewVersion,
  manualPreviewRefresh,
  etaSeconds,
  targetIterations,
  startedAt,
  operationKey,
  onSave,
  onBackup,
  onRefresh,
  onEvaluate,
  onOpenDiagnostics,
  canEvaluate,
  latestEvaluationSnapshotId,
  pendingAction,
  onSafeStop,
  trainingJob,
  savedModel,
  savedModelState,
  onResume,
  onMerge,
}) {
  const { language, t } = useI18n();
  const isRunning = activeTrainingStates.has(trainingState);
  const canControl = ["starting", "running", "waiting_input"].includes(trainingState);
  const presentation = jobPresentation(trainingJob ?? { state: trainingState });
  const stateLabel = t(presentation.label);
  const stateTone = presentation.tone;
  const saveStatus = getTrainingSaveStatus(trainingJob, savedModel);
  const recommendDiagnostics = Boolean(latestEvaluationSnapshotId);
  const [refreshIntervalSeconds, setRefreshIntervalSeconds] = useState(
    readPreviewRefreshSeconds,
  );
  const [displayedPreview, setDisplayedPreview] = useState(null);
  const previewJobId = trainingJob?.id ?? null;
  const previewIsActive = previewEligibleStates.has(trainingState);

  useEffect(() => {
    const input = {
      jobId: previewJobId,
      latestVersion: previewVersion,
      manualRefresh: manualPreviewRefresh,
      intervalSeconds: refreshIntervalSeconds,
      active: previewIsActive,
    };
    const { displayed, delayMs } = planTrainingPreviewDisplay(displayedPreview, input, Date.now());
    if (displayed !== displayedPreview) setDisplayedPreview(displayed);
    if (delayMs === null) return undefined;
    const timer = window.setTimeout(() => {
      setDisplayedPreview((current) => planTrainingPreviewDisplay(current, input, Date.now()).displayed);
    }, delayMs);
    return () => window.clearTimeout(timer);
  }, [displayedPreview, manualPreviewRefresh, previewIsActive, previewJobId, previewVersion, refreshIntervalSeconds]);

  const changeRefreshInterval = (seconds) => {
    setRefreshIntervalSeconds(seconds);
    try { window.localStorage.setItem(PREVIEW_REFRESH_STORAGE_KEY, String(seconds)); } catch { /* Private browsing can reject storage. */ }
  };
  const visiblePreview = displayedPreview?.jobId === previewJobId ? displayedPreview : null;
  const previewUrl = visiblePreview?.version === null || !visiblePreview
    ? null
    : `/api/jobs/${encodeURIComponent(previewJobId)}/preview?v=${visiblePreview.version}&display=${visiblePreview.manualRefresh}`;
  const previewRefresh = `${previewJobId ?? "none"}:${visiblePreview?.version ?? "none"}:${visiblePreview?.manualRefresh ?? 0}`;
  const savedModelLabel = savedModel ? t("已保存模型：{name}", { name: savedModel.name })
    : savedModelState === "loading" ? t("正在读取模型…")
    : savedModelState === "missing" ? t("原模型未找到，请重新选择模型")
    : savedModelState === "ambiguous" ? t("有多个已保存模型，请选择要继续的模型")
    : t("尚未检测到已保存模型");
  return (
    <section className="panel training-panel" aria-labelledby="training-title">
      <div className="training-heading">
        <div>
          <h2 id="training-title">{t("训练任务")} <span>·</span> ME</h2>
          <span className={`status-pill is-${stateTone}`}>
            {stateLabel}
          </span>
        </div>
        <div className="training-heading-actions">
          <PreviewRefreshMenu intervalSeconds={refreshIntervalSeconds} onChange={changeRefreshInterval} />
          <button
            className="button compact secondary"
            type="button"
            onClick={onEvaluate}
            disabled={!canEvaluate || Boolean(pendingAction)}
            title={canEvaluate ? t("使用当前权重生成只读评估快照") : t("运行中的受控 ME 任务才能生成快照")}
          >
            <IconCamera size={14} stroke={1.9}/>{t("评估快照")}
          </button>
          <button
            className={`button compact ${latestEvaluationSnapshotId || recommendDiagnostics ? "primary" : "secondary"}`}
            type="button"
            onClick={onOpenDiagnostics}
          >
            <IconChartDots3 size={14} stroke={1.9}/>{t("质量诊断")}
          </button>
        </div>
      </div>
      <div className="training-preview-stage">
        {isRunning ? (
          <LoadingProgress
            compact
            className="training-run-progress"
            label={trainingState === "running" ? t("ME 正在训练") : stateLabel}
            detail={targetIterations
              ? t("当前 {current} / 估算目标 {target} 次迭代", {
                current: iteration.toLocaleString(language === "zh" ? "zh-CN" : "en-US"),
                target: targetIterations.toLocaleString(language === "zh" ? "zh-CN" : "en-US"),
              })
              : t("持续读取 Trainer 的真实迭代指标")}
            value={targetIterations > 0 ? (iteration / targetIterations) * 100 : undefined}
            current={targetIterations > 0 ? iteration : undefined}
            total={targetIterations > 0 ? targetIterations : undefined}
            etaSeconds={etaSeconds}
            startedAt={startedAt}
            operationKey={operationKey}
            rememberDuration={false}
          />
        ) : null}
        <PreviewGrid
          refreshKey={previewRefresh}
          previewUrl={previewUrl}
          refreshIntervalSeconds={refreshIntervalSeconds}
          trainingState={trainingState}
          hasSavedModel={Boolean(savedModel)}
          modelLoading={savedModelState === "loading"}
        />
      </div>
      {!isRunning && trainingState !== "stopping" ? <div className="training-resume-bar">
        <span>{savedModelLabel}
          {savedModel?.modifiedAt ? <small>{t("保存时间")} · {new Date(savedModel.modifiedAt).toLocaleString(language === "zh" ? "zh-CN" : "en-US")}</small> : null}</span>
        <button className="button primary" type="button" disabled={savedModelState === "loading"} onClick={onResume}>{savedModel ? t("继续训练") : ["missing", "ambiguous"].includes(savedModelState) ? t("选择训练模型") : t("新建训练")}</button>
        {savedModel ? <button className="button secondary" type="button" onClick={onMerge}>{t("进入合成")}</button> : null}
      </div> : <div className="training-actions">
        <button className="button primary" type="button" onClick={onSave} disabled={!canControl || Boolean(pendingAction)}>
          <IconDeviceFloppy size={17} stroke={1.9} />{t("保存")}
        </button>
        <button className="button secondary" type="button" onClick={onBackup} disabled={!canControl || Boolean(pendingAction)}>
          <IconArchive size={17} stroke={1.9} />{t("备份")}
        </button>
        <button className="button secondary" type="button" onClick={onRefresh} disabled={!canControl || Boolean(pendingAction)}>
          <IconRefresh size={17} stroke={1.9} />{t("刷新预览")}
        </button>
        <button className="button danger" type="button" onClick={onSafeStop} disabled={!canControl || Boolean(pendingAction)}>
          <IconShieldX size={17} stroke={1.9} />{t("安全停止")}
        </button>
      </div>}
      <div className={`workflow-save-status is-${saveStatus.saveState}`} role="status">
        <strong>{t(saveStatus.historyLabel)}</strong>
        <p>{t(saveStatus.detail)}</p>
      </div>
    </section>
  );
}

function MetricRow({ label, value, suffix, percent, tone = "green", valueTone = null }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      {typeof percent === "number" ? (
        <span className="metric-bar" aria-hidden="true">
          <span className={`metric-fill ${tone}`} style={{ width: `${percent}%` }} />
        </span>
      ) : null}
      <strong className={valueTone ? `is-${valueTone}` : undefined}>{value}{suffix}</strong>
    </div>
  );
}

export function StatusPanel({
  trainingJob,
  telemetry,
  lossHistory = [],
  onOpenModels,
  historyError,
  onRetryHistory,
}) {
  const { language, t } = useI18n();
  const metric = trainingJob?.latestMetric;
  const gpu = telemetry?.gpus?.[0];
  const memoryPercent = gpu?.memoryTotalMiB
    ? Math.min(100, (gpu.memoryUsedMiB / gpu.memoryTotalMiB) * 100)
    : null;
  const trainingState = trainingJob?.state ?? "idle";
  const trainerHealth = trainingJob?.health;
  const trainerHealthLabel = trainerHealth?.state === "healthy" ? t("正常")
    : trainerHealth?.state === "busy" ? t("正在处理长步骤")
      : trainerHealth?.state === "hung" ? t("疑似停滞") : t("检测中");
  const presentation = jobPresentation(trainingJob);
  const trainingStateTone = presentation.tone;
  return (
    <aside className="panel status-panel" aria-labelledby="status-title">
      <div className="panel-heading">
        <h2 id="status-title">{t("实时状态")}</h2>
        <button
          className="status-model-button"
          type="button"
          onClick={onOpenModels}
          title={t("检查点由 DFL 正式保存流程写入，不展示模拟文件。")}
        >
          <IconFolderOpen size={14} stroke={1.8} />{t("复制模型目录")}
        </button>
      </div>
      <div className="status-metric-section is-training">
        <h3>{t("训练指标")}</h3>
        <div className="metrics-list">
          <MetricRow
            label={t("状态")}
            value={t(presentation.label)}
            suffix=""
            valueTone={trainingStateTone}
          />
          <MetricRow label={t("训练进程")} value={!trainingJob || trainingState === "queued" ? "—" : previewEligibleStates.has(trainingState) ? trainingJob?.pid ?? "—" : t("已结束")} suffix="" />
          {previewEligibleStates.has(trainingState) && trainingJob?.commandId === "train.me" ?
            <MetricRow label={t("训练心跳")} value={trainerHealthLabel} suffix=""
              valueTone={trainerHealth?.state === "hung" ? "amber" : trainerHealth?.state === "healthy" ? "green" : null} /> : null}
          <MetricRow label={t("当前迭代")} value={metric?.iteration?.toLocaleString(language === "zh" ? "zh-CN" : "en-US") ?? "—"} suffix="" />
          <MetricRow label={t("单次迭代")} value={metric?.iterationTime ?? "—"} suffix="" />
          <MetricRow label={t("训练速度")} value={formatRate(metric?.iterationsPerHour)} suffix="" />
          <MetricRow label={t("进度估算")} value={formatEta(metric?.etaSeconds, t)} suffix="" />
          <MetricRow label={t("SRC 损失")} value={typeof metric?.srcLoss === "number" ? metric.srcLoss.toFixed(4) : "—"} suffix="" />
          <MetricRow label={t("DST 损失")} value={typeof metric?.dstLoss === "number" ? metric.dstLoss.toFixed(4) : "—"} suffix="" />
        </div>
        {previewEligibleStates.has(trainingState) && trainerHealth?.state === "hung" ? <p className="training-history-recovery" role="status">
          <IconAlertTriangle size={15}/>{t(trainerHealth.reason || "Trainer 迭代长时间没有进展；请检查日志与显存。")}
        </p> : null}
      </div>
      {historyError ? <div className="training-history-recovery" role="status">
        <span>{t("训练历史暂时无法读取")}</span>
        {onRetryHistory ? <button type="button" className="button compact secondary" onClick={onRetryHistory}>{t("重试")}</button> : null}
      </div> : null}
      <TrainingChart
        className="status-loss-chart"
        iteration={metric?.iteration ?? 0}
        lossHistory={lossHistory}
      />
      <div className="status-metric-section is-gpu">
        <h3>{t("GPU 状态")}</h3>
        <div className="metrics-list">
          <MetricRow
            label={t("GPU 利用率")}
            value={typeof gpu?.utilizationPercent === "number" ? Math.round(gpu.utilizationPercent) : "—"}
            suffix={typeof gpu?.utilizationPercent === "number" ? "%" : ""}
            percent={gpu?.utilizationPercent}
          />
          <MetricRow
            label={t("显存")}
            value={gpu ? `${(gpu.memoryUsedMiB / 1024).toFixed(1)} / ${(gpu.memoryTotalMiB / 1024).toFixed(1)}` : "—"}
            suffix={gpu ? " GB" : ""}
            percent={memoryPercent}
          />
          <MetricRow
            label={t("GPU 温度")}
            value={typeof gpu?.temperatureC === "number" ? Math.round(gpu.temperatureC) : "—"}
            suffix={typeof gpu?.temperatureC === "number" ? "°C" : ""}
            percent={gpu?.temperatureC}
            tone={gpu?.temperatureC >= 80 ? "amber" : "green"}
          />
        </div>
      </div>
    </aside>
  );
}

export function WorkbenchGrid(props) {
  return (
    <div className="workbench-grid">
      <PipelinePanel {...props.pipeline} />
      <TrainingWorkspace {...props.training} />
      <StatusPanel {...props.status} />
    </div>
  );
}
