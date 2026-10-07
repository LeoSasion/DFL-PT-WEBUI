import { preflightRecovery } from "../domain/preflight-recovery.js";
import { commandModelFamily, createTaskConfiguration, displayWorkspacePath, modelNameIssue, searchTaskCommands, selectTaskModel, taskModels, taskOutputLocations } from "../domain/task-configuration.js";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  IconAlertTriangle,
  IconArrowLeft,
  IconArrowRight,
  IconCheck,
  IconChecks,
  IconCode,
  IconLock,
  IconPlayerStop,
  IconPlus,
  IconSettings,
  IconX,
} from "@tabler/icons-react";
import { taskTypes } from "../data/dashboard.js";
import { useI18n } from "../i18n.jsx";
import { LoadingProgress } from "./ProgressFeedback.jsx";
import { METrainingFields } from "./METrainingFields.jsx";
import { METrainingReview } from "./METrainingReview.jsx";
import { TrainingInputPicker } from "./TrainingInputPicker.jsx";
import "./WorkflowGuidance.css";

import { useDialogFocus } from "./Overlays.jsx";

export function NewTaskDialog({
  open,
  taskType,
  workspacePath,
  serviceOnline,
  commands,
  onTaskType,
  onPreflight,
  onResolvePreflight,
  onClose,
  onCreate,
  workspace,
  telemetry,
  initialParameters,
  recommendedCommandId,
  initialStep = "choice",
  onOpenMergeReview,
  projectKey,
  trainingInputs,
  onTrainingInputsChange,
}) {
  const { t } = useI18n();
  const [submitting, setSubmitting] = useState(false);
  const submittingRef = useRef(false);
  const { dialogRef, initialFocusRef } = useDialogFocus(open, () => {
    if (!submittingRef.current) onClose();
  });
  const stepPanelRef = useRef(null);
  const preflightRef = useRef(onPreflight);
  preflightRef.current = onPreflight;
  const [step, setStep] = useState(1);
  const [parameters, setParameters] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [preflight, setPreflight] = useState({ state: "idle", data: null, error: null });
  const [launchError, setLaunchError] = useState(null);
  const [preflightAttempt, setPreflightAttempt] = useState(0);
  const [newModel, setNewModel] = useState(false);
  const [commandSearch, setCommandSearch] = useState("");
  const availableCommands = useMemo(
    () => commands?.length
      ? commands
      : taskTypes.map((task) => ({
        ...task,
        label: t(task.label),
        description: t("本地 DeepFaceLab 固定工作流"),
        parameters: [],
        locks: [],
      })),
    [commands, t],
  );
  const selectedCommand = useMemo(
    () => availableCommands.find((command) => command.id === taskType) ?? availableCommands[0],
    [availableCommands, taskType],
  );
  const parameterSchemas = selectedCommand?.parameters ?? [];
  const importingME = selectedCommand?.id === "model.import_me_tf";
  const configuringME = selectedCommand?.id === "train.me" || importingME;
  const mergingME = ["merge.me", "merge.preview_me"].includes(selectedCommand?.id);
  const models = taskModels(selectedCommand, workspace?.models);
  const selectedModel = models.find(model => model.name === parameters.forceModelName);
  const modelIssue = modelNameIssue(parameters.forceModelName, newModel,
    (workspace?.models ?? []).filter(model => model.type?.toUpperCase() === commandModelFamily(selectedCommand)));
  const matchingCommands = searchTaskCommands(availableCommands, commandSearch);
  const outputLocations = taskOutputLocations(selectedCommand, workspace, parameters);
  const hasModelChoice = parameterSchemas.some(schema => schema.id === "forceModelName");
  const hasDeviceChoice = parameterSchemas.some(schema => schema.id === "gpuIndexes");
  const visibleParameters = parameterSchemas.filter(
    (parameter) => !["forceModelName", "gpuIndexes", "cpuOnly"].includes(parameter.id)
      && (parameter.id !== "silentStart" || (showAdvanced && !newModel))
      && (showAdvanced || !parameter.advanced),
  );

  useEffect(() => {
    setStep(open && initialStep === "parameters" ? 2 : 1);
    if (!open) return;
    setShowAdvanced(false);
    setCommandSearch("");
    setPreflight({ state: "idle", data: null, error: null });
    setLaunchError(null);
  }, [open, initialStep]);

  useEffect(() => {
    if (!open) return;
    const configuration = createTaskConfiguration(selectedCommand, initialParameters, workspace?.models);
    setParameters(configuration.parameters);
    setNewModel(configuration.newModel);
    setPreflight({ state: "idle", data: null, error: null });
    setLaunchError(null);
  }, [selectedCommand?.id, open, initialParameters]);

  useEffect(() => {
    if (!open) return;
    const panel = stepPanelRef.current;
    if (step === 1) initialFocusRef.current?.focus();
    else if (step === 2) (panel?.querySelector("input:not(:disabled), select:not(:disabled)") ?? panel?.querySelector("button:not(:disabled)"))?.focus();
    else panel?.focus();
  }, [open, step]);

  useEffect(() => {
    if (open && launchError) stepPanelRef.current?.focus();
  }, [open, launchError]);

  useEffect(() => {
    if (!open || step !== 3 || !selectedCommand) return undefined;
    let cancelled = false;
    setPreflight({ state: "checking", data: null, error: null });
    // Let the command/configuration reset commit before issuing a request.
    // Closing and immediately reopening must not preflight the old step 3.
    const kickoff = setTimeout(() => {
      if (cancelled) return;
      void preflightRef.current(selectedCommand.id, {
        launchMode: "guided", parameters,
      }).then(data => {
        if (!cancelled) setPreflight({ state: "ready", data, error: null });
      }).catch(error => {
        if (!cancelled) setPreflight({ state: "failed", data: null, error });
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(kickoff);
    };
  }, [open, parameters, selectedCommand?.id, step, preflightAttempt]);

  if (!open) return null;

  const setParameter = (schema, rawValue) => {
    let value = rawValue;
    if (schema.type === "boolean") value = Boolean(rawValue);
    if (schema.type === "select") {
      value = schema.options.find((option) => String(option.value) === String(rawValue))?.value
        ?? rawValue;
    }
    setParameters((current) => ({ ...current, [schema.id]: value }));
    setPreflight({ state: "idle", data: null, error: null });
  };

  const setMEParameters = (update) => {
    setParameters(update);
    setPreflight({ state: "idle", data: null, error: null });
    setLaunchError(null);
  };

  const formatParameter = (schema) => {
    const value = parameters[schema.id];
    if (value === "" || value === undefined || value === null) return t("自动 / 终端询问");
    if (schema.type === "boolean") return value ? t("是") : t("否");
    if (schema.type === "select") {
      return schema.options.find((option) => String(option.value) === String(value))?.label ?? String(value);
    }
    return `${value}${schema.suffix ? ` ${schema.suffix}` : ""}`;
  };

  const recovery = preflight.state === "failed"
    ? preflightRecovery(preflight.error, selectedCommand, t)
    : null;
  const resourceAdvice = preflight.state === "ready" ? preflight.data?.resources : null;

  const resolvePreflight = () => {
    if (!recovery) return;
    if (recovery.target === "parameters") {
      setStep(2);
      return;
    }
    onResolvePreflight?.(recovery.target);
  };

  const launch = async (launchMode) => {
    if (submittingRef.current || !serviceOnline || (launchMode === "guided" && preflight.state !== "ready")) return;
    submittingRef.current = true;
    setSubmitting(true);
    setLaunchError(null);
    try {
      await onCreate({
        launchMode,
        parameters: launchMode === "guided" ? parameters : {},
      });
    } catch (error) {
      setLaunchError(error);
      setPreflight({ state: "failed", data: null, error });
      setStep(3);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  };

  const stepLabels = ["选择任务", "配置参数", "确认执行"].map((label) => t(label));
  const mergeModeDescriptions = {
    overlay: "按遮罩叠加预测人脸，适合先检查轮廓与颜色。",
    "hist-match": "先匹配目标帧亮度分布再叠加，适合光照差异较大的画面。",
    seamless: "融合边缘与目标背景，适合检查轮廓衔接。",
    "seamless-hist-match": "匹配亮度后融合边缘，同时检查颜色与接缝。",
    "raw-rgb": "输出未经融合的预测区域，用于排查模型结果。",
    "raw-predict": "输出原始预测脸，用于检查预测本身。",
    original: "保留原帧，用于与替换结果对照。",
  };
  const mergeLabels = {
    mode: { overlay: "遮罩叠加 · 推荐", "hist-match": "亮度匹配后叠加", seamless: "无缝融合", "seamless-hist-match": "亮度匹配 + 无缝融合", "raw-rgb": "原始 RGB 区域", "raw-predict": "原始预测脸", original: "保留原帧" },
    maskMode: { 1: "目标脸范围", 2: "模型预测脸遮罩", 3: "模型目标脸遮罩", 4: "两侧模型遮罩交集 · 推荐", 5: "两侧模型遮罩并集", 6: "XSeg 预测脸遮罩", 7: "XSeg 目标脸遮罩", 8: "两侧 XSeg 遮罩交集", 9: "模型 + XSeg 组合", 10: "已复核 DST 遮罩 · 原帧绑定", 11: "模型预测 × 已复核 DST 遮罩" },
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !submittingRef.current) onClose();
    }}>
      <section
        className="modal-card task-wizard"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-task-title"
        aria-describedby="new-task-description"
      >
        <header>
          <div>
            <span className="modal-icon"><IconPlus size={20} stroke={2} /></span>
            <div>
              <h2 id="new-task-title">{t("新建任务")}</h2>
              <p id="new-task-description">{t("选择任务、确认素材和保存位置，再开始处理。")}</p>
            </div>
          </div>
          <button className="icon-button quiet" type="button" aria-label={t("关闭")} onClick={onClose} disabled={submitting}>
            <IconX size={19} />
          </button>
        </header>

        <div className="wizard-progress" aria-label={t("任务创建进度")}>
          {stepLabels.map((label, index) => {
            const number = index + 1;
            return (
              <div
                key={label}
                className={`wizard-progress-item ${number === step ? "is-current" : ""} ${number < step ? "is-done" : ""}`}
              >
                <span>{number < step ? <IconCheck size={14} /> : number}</span>
                <strong>{label}</strong>
              </div>
            );
          })}
        </div>
        {submitting ? (
          <LoadingProgress inline compact label={t("正在创建任务并连接终端…")} detail={t("窗口会在任务成功创建后关闭")} operationKey="task-create" />
        ) : preflight.state === "checking" ? (
          <LoadingProgress inline compact label={t("正在执行前置检查…")} detail={t("正在核对素材、运行时与资源锁")} operationKey="task-preflight" />
        ) : null}

        <div className="wizard-body">
          <section className="wizard-main" ref={stepPanelRef} tabIndex={-1} aria-label={stepLabels[step - 1]} aria-busy={submitting} inert={submitting || undefined}>
            {step === 1 && (
              <div className="wizard-step-panel">
                {recommendedCommandId && availableCommands.some(command => command.id === recommendedCommandId) ? <button type="button" className="button secondary" onClick={() => { onTaskType(recommendedCommandId); setCommandSearch(""); }}>
                  {t("当前项目建议")} · {availableCommands.find(command => command.id === recommendedCommandId)?.label}
                </button> : null}
                <label className="wizard-field"><span>{t("搜索任务")}</span><input ref={initialFocusRef} type="search" value={commandSearch} onChange={event => setCommandSearch(event.target.value)} placeholder={t("输入任务名称")} /></label>
                {commandSearch.trim() ? <div className="wizard-search-feedback" role="status">
                  <span>{matchingCommands.length ? t("找到 {count} 个任务", { count: matchingCommands.length }) : t("没有匹配任务，当前选择仍保留。")}</span>
                  <button className="text-button" type="button" onClick={() => { setCommandSearch(""); initialFocusRef.current?.focus(); }}>{t("清除搜索")}</button>
                </div> : null}
                <label className="wizard-field">
                  <span>{t("任务类型")}</span>
                  <select
                    value={selectedCommand?.id ?? ""}
                    onChange={(event) => onTaskType(event.target.value)}
                  >
                    {selectedCommand && !matchingCommands.some(command => command.id === selectedCommand.id) ? <optgroup label={t("当前选择（不在搜索结果中）")}><option value={selectedCommand.id}>{selectedCommand.label}</option></optgroup> : null}
                    {[...new Set(matchingCommands.map(command => command.category ?? "other"))].map(category => <optgroup key={category} label={t(({extract:"素材提取",dataset:"数据集工具",training:"模型训练",sort:"排序清洗",mask:"XSeg 遮罩",merge:"模型应用",encode:"视频封装",model:"模型工具",video:"视频处理",utility:"环境维护",other:"其他任务"})[category] ?? category)}>
                      {matchingCommands.filter(command => (command.category ?? "other") === category).map(command => <option key={command.id} value={command.id}>{command.label}</option>)}
                    </optgroup>)}
                  </select>
                </label>
                <div className="command-profile">
                  <span className="command-profile-icon"><IconSettings size={20} /></span>
                  <div className="preflight-copy">
                    <strong>{selectedCommand?.label}</strong>
                    <p>{selectedCommand?.description}</p>
                    <details className="command-tags"><summary>{t("技术详情")}</summary>
                      <span>{selectedCommand?.id === "train.me" ? "ME PyTorch" : "PyTorch 素材工具"}</span>
                      <span>{selectedCommand?.stage ?? "workflow"}</span>
                      <span>{selectedCommand?.side?.toUpperCase?.() ?? "LOCAL"}</span>
                    </details>
                  </div>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="wizard-step-panel">
                <div className="wizard-section-heading">
                  <div>
                    <strong>{selectedCommand?.label}</strong>
                    <p>{hasModelChoice || hasDeviceChoice ? t("先选择模型和设备，其余参数可按需调整。") : t("检查以下设置，再确认运行。")}</p>
                    <button type="button" className="text-button" onClick={() => { setPreflight({ state: "idle", data: null, error: null }); setStep(1); }}>{t("更换任务")}</button>
                  </div>
                  {!configuringME && parameterSchemas.some((parameter) => parameter.advanced) && (
                    <button
                      className={`text-button ${showAdvanced ? "is-active" : ""}`}
                      type="button"
                      onClick={() => setShowAdvanced((current) => !current)}
                    >
                      {showAdvanced ? t("收起高级参数") : t("显示高级参数")}
                    </button>
                  )}
                </div>
                {hasModelChoice ? <div className={`model-choice${configuringME ? " me-model-choice" : ""}`}>
                  {!importingME ? <label className="wizard-field"><span>{t("使用模型")}</span>
                    <select value={newModel ? "__new__" : parameters.forceModelName ?? ""} onChange={event => {
                      const next = selectTaskModel(selectedCommand, parameters, event.target.value, models);
                      setNewModel(next.newModel);
                      setMEParameters(next.parameters);
                    }}>
                      <option value="">{selectedCommand.id === "train.me" || selectedCommand.category === "merge" || selectedCommand.category === "model" ? t("请选择模型") : t("由终端选择模型")}</option>
                      {models.map(model => <option key={model.name} value={model.name}>{model.name}</option>)}
                      {parameters.forceModelName && !newModel && !models.some(model => model.name === parameters.forceModelName) ? <option value={parameters.forceModelName}>{parameters.forceModelName}</option> : null}
                      {selectedCommand.category === "training" ? <option value="__new__">{t("新建模型")}</option> : null}
                    </select>
                  </label> : null}
                  {newModel ? <label className="wizard-field"><span>{t("新模型名称")}</span><input value={parameters.forceModelName ?? ""} maxLength={64} required aria-invalid={modelIssue && modelIssue !== "empty" ? true : undefined} aria-describedby="new-model-help" onChange={event => setParameters(current => ({...current,forceModelName:event.target.value}))} />
                    <small id="new-model-help" className={modelIssue && modelIssue !== "empty" ? "wizard-field-error" : ""}>
                      {modelIssue === "duplicate" ? t("同名模型已存在；请选择已有模型继续训练，或使用新名称。") : modelIssue === "invalid" ? t("模型名称不能包含路径符号、首尾空格，且不能超过 64 个字符。") : importingME ? t("导入到新的 ME 模型；网络权重保留，训练进度从零开始。") : selectedCommand.id === "train.me" ? t("选择架构与训练配置，新模型会保存到独立目录。") : t("使用独立名称创建模型；首次配置会在终端继续。")}
                    </small>
                  </label>
                    : selectedModel?.modifiedAt ? <p>{t("保存时间")} · {new Date(selectedModel.modifiedAt).toLocaleString()}</p> : null}
                  {!newModel && selectedModel && selectedCommand.category === "training" ? <p>{t("继续所选模型的训练；已有进度和模型设置会保留。")}</p> : null}
                </div> : null}
                {hasDeviceChoice ? <label className="wizard-field device-choice"><span>{t("运行设备")}</span>
                  <select value={parameters.cpuOnly ? "cpu" : parameters.gpuIndexes ?? ""} onChange={event => setMEParameters(current => ({...current,cpuOnly:event.target.value === "cpu",gpuIndexes:event.target.value === "cpu" ? "" : event.target.value}))}>
                    <option value="">{t("自动选择 GPU")}</option>
                    {(telemetry?.gpus ?? []).map(gpu => <option key={gpu.index} value={String(gpu.index)}>{gpu.name ?? `GPU ${gpu.index}`}{Number.isFinite(gpu.memoryTotalMiB) ? ` · ${(gpu.memoryTotalMiB / 1024).toFixed(1)} GB` : ""}</option>)}
                    {parameters.gpuIndexes && !(telemetry?.gpus ?? []).some(gpu => String(gpu.index) === parameters.gpuIndexes) ? <option value={parameters.gpuIndexes}>GPU {parameters.gpuIndexes}</option> : null}
                    <option value="cpu">{t("CPU（较慢）")}</option>
                  </select>
                  <small>{t("自动模式由 DFL 沿用或询问设备；指定 GPU 可避免设备选择问答。")}</small>
                  {!parameters.cpuOnly && (showAdvanced || selectedCommand.id === "train.me") ? <details><summary>{t("选择多张 GPU")}</summary><input aria-label={t("多 GPU 索引")} placeholder="0,1" value={parameters.gpuIndexes ?? ""} onChange={event => setMEParameters(current => ({...current,gpuIndexes:event.target.value}))} /></details> : null}
                </label> : null}
                {selectedCommand.id === "train.me" && projectKey ? <TrainingInputPicker
                  projectKey={projectKey} value={trainingInputs} onChange={onTrainingInputsChange}
                  configuration={parameters} onApply={inputs => setMEParameters(current => ({
                    ...current,
                    srcFaceset: inputs.srcFaceset ?? inputs.trainingDataSrcPath ?? current.srcFaceset,
                    dstFaceset: inputs.dstFaceset ?? inputs.trainingDataDstPath ?? current.dstFaceset,
                  }))}
                /> : null}
                {configuringME ? <METrainingFields
                  schemas={parameterSchemas}
                  parameters={parameters}
                  selectedModel={selectedModel}
                  newModel={newModel}
                  models={models}
                  onChange={setMEParameters}
                  importing={importingME}
                /> : visibleParameters.length ? (
                  <div className="wizard-form-grid">
                    {visibleParameters.map((schema) => (
                      schema.type === "boolean" ? (
                        <label className="wizard-toggle" key={schema.id}>
                          <input
                            type="checkbox"
                            checked={Boolean(parameters[schema.id])}
                            onChange={(event) => setParameter(schema, event.target.checked)}
                          />
                          <span>
                            <strong>{schema.label}</strong>
                            {schema.help && <small>{schema.help}</small>}
                          </span>
                        </label>
                      ) : (
                        <label className="wizard-field" key={schema.id}>
                          <span>{schema.label}</span>
                          {schema.type === "select" ? (
                            <select
                              value={parameters[schema.id] ?? ""}
                              disabled={selectedCommand.id === "train.me" && !newModel && schema.id === "resolution"}
                              onChange={(event) => setParameter(schema, event.target.value)}
                            >
                              {schema.options.map((option) => (
                                <option key={String(option.value)} value={option.value}>{mergingME ? t(mergeLabels[schema.id]?.[option.value] ?? option.label) : option.label}</option>
                              ))}
                            </select>
                          ) : (
                            <div className="wizard-input-wrap">
                              <input
                                type={schema.type}
                                value={parameters[schema.id] ?? ""}
                                disabled={selectedCommand.id === "train.me" && !newModel && schema.id === "batchSize"}
                                min={schema.min}
                                max={schema.max}
                                step={schema.integer ? 1 : undefined}
                                placeholder={schema.placeholder}
                                onChange={(event) => setParameter(schema, event.target.value)}
                              />
                              {schema.suffix && <span>{schema.suffix}</span>}
                            </div>
                          )}
                          {schema.help && <small>{schema.help}</small>}
                        </label>
                      )
                    ))}
                  </div>
                ) : !hasModelChoice && !hasDeviceChoice ? (
                  <div className="wizard-empty">
                    <IconChecks size={24} />
                    <strong>{t("此任务无需额外参数")}</strong>
                    <p>{t("继续后会检查素材、模型与资源锁。")}</p>
                  </div>
                ) : null}
                {mergingME ? <section className="wizard-merge-guidance" aria-label={t("当前合成用途")}>
                  <strong>{t("当前合成设置")}</strong>
                  <p>{t(mergeModeDescriptions[parameters.mode] ?? "按当前模式生成合成帧。")}</p>
                  {selectedCommand.id === "merge.preview_me" ? <p>{t("本次仅处理第 {start}–{end} 帧（最多 20 张），输出到独立小样目录。正式合成序列保持独立；检查小样后再明确启动全片合成。", { start: parameters.frameStart, end: Number(parameters.frameStart) + Number(parameters.frameCount) - 1 })}</p> : null}
                  {selectedCommand.id === "merge.me" && parameters.previewJobId ? <div className="wizard-merge-preview-binding" role="status">
                    <strong>{t("已采用小样参数")}</strong><p>{t("启动前会再次检查此小样的素材、模型和合成参数。若调整参数，可重新生成小样；也可清除绑定后按普通合成预检执行。")}</p>
                    <button type="button" className="text-button" onClick={() => setMEParameters(current => ({ ...current, previewJobId: "" }))}>{t("清除小样绑定，保留当前参数")}</button>
                  </div> : null}
                  <p>{t([10, 11].includes(Number(parameters.maskMode)) ? "使用已复核 DST 遮罩时，系统仍会检查原帧绑定、身份与几何匹配；缺失或不匹配不能跳过。" : "遮罩决定替换范围。交集更保守，并集覆盖更广；先检查眼睛、嘴和下颌边缘。")}</p>
                  {onOpenMergeReview && workspace?.readiness?.merged ? <button type="button" className="text-button" onClick={() => { onClose(); onOpenMergeReview(); }}>{t("查看已有合成帧与遮罩")}</button> : null}
                </section> : null}
              </div>
            )}

            {step === 3 && (
              <div className="wizard-step-panel">
                {launchError ? <div className="preflight-banner is-failed" role="alert"><IconAlertTriangle size={18} /><div><strong>{t("创建任务遇到问题")}</strong><p>{t(launchError.message ?? "创建任务失败，请重新检查后重试。")}</p></div></div> : null}
                <div className={`preflight-banner is-${preflight.state}`} role="status" aria-live="polite">
                  {preflight.state === "checking" && <span className="status-pulse" />}
                  {preflight.state === "ready" && <IconCheck size={18} />}
                  {preflight.state === "failed" && <IconAlertTriangle size={18} />}
                  <div>
                    <strong>
                      {preflight.state === "checking" && t("正在执行前置检查")}
                      {preflight.state === "ready" && t("前置检查通过")}
                      {preflight.state === "failed" && t("前置检查未通过")}
                      {preflight.state === "idle" && t("等待前置检查")}
                    </strong>
                    <p>
                      {preflight.error?.message
                        ? t(preflight.error.message)
                        : (preflight.state === "ready"
                          ? t("素材、运行时和资源锁均可用于创建任务。")
                          : t("不会在检查期间启动 DFL 进程。"))}
                    </p>
                    {preflight.error?.code ? <code>{preflight.error.code}</code> : null}
                  </div>
                  {recovery ? (
                    <button className="button secondary preflight-recovery" type="button" onClick={resolvePreflight}>
                      {recovery.label}<IconArrowRight size={15} />
                    </button>
                  ) : null}
                  {preflight.state === "failed" ? <button className="button secondary" type="button" onClick={() => { setLaunchError(null); setPreflightAttempt(current => current + 1); }}>{t("重新检查")}</button> : null}
                </div>
                {resourceAdvice ? (
                  <section
                    className={`preflight-resource-advice is-${resourceAdvice.severity ?? "info"}`}
                    aria-label={t("训练资源建议")}
                  >
                    <IconSettings size={17} aria-hidden="true" />
                    <div>
                      <strong>{t("显存与安全起点")}</strong>
                      <p>{t(resourceAdvice.summary ?? "训练资源状态已检查")}</p>
                      <dl>
                        {resourceAdvice.gpu ? (
                          <div>
                            <dt>{t("GPU")}</dt>
                            <dd>{resourceAdvice.gpu.name ?? `GPU ${resourceAdvice.gpu.index ?? "—"}`} · {resourceAdvice.gpu.freeGiB ?? "—"} / {resourceAdvice.gpu.totalGiB ?? "—"} GB {t("可用")}</dd>
                          </div>
                        ) : null}
                        <div>
                          <dt>{t("建议参数")}</dt>
                          <dd>{t("分辨率 {resolution} · Batch {batch}", {
                            resolution: resourceAdvice.recommendation?.resolution ?? "—",
                            batch: resourceAdvice.recommendation?.batchSize ?? "—",
                          })}</dd>
                        </div>
                      </dl>
                      <small>{t("这是保守的安全起点，不保证最终显存占用；以 DFL 实际训练为准。")}</small>
                    </div>
                  </section>
                ) : null}
                {configuringME ? <METrainingReview schemas={parameterSchemas} parameters={parameters} importing={importingME} preflight={preflight.data} /> : <dl className="wizard-review">
                  <div><dt>{t("任务")}</dt><dd>{selectedCommand?.label}</dd></div>
                  {hasDeviceChoice ? <div><dt>{t("运行设备")}</dt><dd>{parameters.cpuOnly ? t("CPU（较慢）") : parameters.gpuIndexes ? `GPU ${parameters.gpuIndexes}` : t("自动选择 GPU")}</dd></div> : null}
                  {parameterSchemas.filter(schema => !["gpuIndexes","cpuOnly","silentStart"].includes(schema.id)).map((schema) => (
                    <div key={schema.id}><dt>{schema.label}</dt><dd>{formatParameter(schema)}</dd></div>
                  ))}
                </dl>}
              </div>
            )}
          </section>

          <aside className="wizard-summary">
            <h3>{t("执行摘要")}</h3>
            <dl>
              <div><dt>{t("任务")}</dt><dd>{selectedCommand?.label}</dd></div>
              <div><dt>{t("工作区")}</dt><dd title={workspacePath}>{workspacePath}</dd></div>
              <div><dt>{t("保存位置")}</dt><dd>{outputLocations.paths.length ? outputLocations.paths.map(location => <span className="wizard-output-path" key={location} title={displayWorkspacePath(workspacePath, location)}>{displayWorkspacePath(workspacePath, location)}</span>) : t("项目运行环境与缓存目录")}</dd></div>
            </dl>
            {outputLocations.note === "merge-back" ? <p className="wizard-summary-note">{t("先生成到此目录；处理完成后，终端会询问是否合并回原人脸集。")}</p> : null}
            {outputLocations.note === "independent-enhancement" ? <p className="wizard-summary-note">{t("保存完整独立副本，原件保留；本批增强结果需要复核。")}</p> : null}
            {outputLocations.note === "independent-merge-preview" ? <p className="wizard-summary-note">{t("仅用于复核当前参数的小样；不能作为全片导出的合成输入。")}</p> : null}
            {selectedCommand?.category === "encode" ? <p className="wizard-summary-note">{t("导出会更新同名成片和遮罩文件。请先备份需要保留的视频；生成记录会保留。")}</p> : null}
            <details><summary>{t("技术详情")}</summary><dl>
              <div><dt>{t("运行时")}</dt><dd>{selectedCommand?.id === "train.me" ? "ME PyTorch" : "PyTorch 素材工具"}</dd></div>
              <div>
                <dt><IconLock size={13} />{t("资源锁")}</dt>
                <dd>{selectedCommand?.locks?.length ? t("{count} 项", { count: selectedCommand.locks.length }) : t("无")}</dd>
              </div>
              <div><dt>{t("终端")}</dt><dd>{t("可交互 ConPTY")}</dd></div>
            </dl></details>
            {selectedCommand?.interactive ? <div className="wizard-summary-note">
              <IconCode size={16} />
              <p>{t("如有额外问题，任务会等待你的回答，并在终端提示。")}</p>
            </div> : null}
          </aside>
        </div>

        <footer>
          <div className="wizard-footer-start">
            <button className="button secondary" type="button" onClick={onClose} disabled={submitting}>{t("取消")}</button>
            {step > 1 && (
              <button className="button secondary" type="button" onClick={() => { setLaunchError(null); setStep((current) => current - 1); }} disabled={submitting}>
                <IconArrowLeft size={16} />{t("上一步")}
              </button>
            )}
          </div>
          <div className="wizard-footer-end">
            {step >= 2 && showAdvanced && (
              <button
                className="button cli-button"
                type="button"
                onClick={() => void launch("cli")}
                disabled={!serviceOnline || submitting || ["train.me", "model.import_me_tf", "merge.me", "merge.preview_me", "export.dfm_me"].includes(selectedCommand?.id)}
              >
                <IconCode size={16} />{t("保留 CLI 问答")}
              </button>
            )}
            {step < 3 ? (
              <button
                className="button primary"
                type="button"
                onClick={() => {
                  const invalid = [...(stepPanelRef.current?.querySelectorAll("input, select") ?? [])].find(field => !field.checkValidity());
                  if (step === 2 && invalid) {
                    let ancestor = invalid.parentElement;
                    while (ancestor && ancestor !== stepPanelRef.current) {
                      if (ancestor.tagName === "DETAILS") ancestor.open = true;
                      ancestor = ancestor.parentElement;
                    }
                    invalid.focus();
                    invalid.reportValidity();
                    return;
                  }
                  setStep((current) => current + 1);
                }}
                disabled={!selectedCommand || submitting || (step === 2 && Boolean(modelIssue))}
              >
                {t("下一步")}<IconArrowRight size={16} />
              </button>
            ) : (
              <button
                className="button primary"
                type="button"
                onClick={() => void launch("guided")}
                disabled={!serviceOnline || preflight.state !== "ready" || submitting}
              >
                <IconPlus size={17} />{t("启动任务")}
              </button>
            )}
          </div>
        </footer>
      </section>
    </div>
  );
}
