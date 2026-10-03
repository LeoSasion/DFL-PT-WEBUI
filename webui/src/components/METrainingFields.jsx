import { useId } from "react";
import { ME_CONFIG_GROUPS, getMePreset } from "../../shared/me-training-options.mjs";
import { applyMEPreset, initializeMEParameters, meConfigurationChanges, modelMEParameters } from "../domain/me-training-configuration.js";
import { useI18n } from "../i18n.jsx";
import "./METrainingFields.css";

const EXTERNAL_FIELDS = new Set(["forceModelName", "silentStart", "gpuIndexes", "cpuOnly"]);
const RESUME_OPERATIONS = new Set(["resetOptimizer", "resetDataState"]);
const PRESETS = [
  { id: "default", label: "默认训练" },
  { id: "rg-fp16", label: "RG + FP16" },
  { id: "pretrain", label: "预训练" },
  { id: "finetune", label: "后期微调" },
];

function MEField({ schema, value, disabled, onChange, t, prefix, models }) {
  const id = `${prefix}-${schema.id}`;
  const helpId = `${id}-help`;
  const change = raw => {
    const option = schema.type === "select" ? schema.options?.find(item => String(item.value) === String(raw)) : null;
    onChange(schema.id, schema.type === "boolean" ? Boolean(raw) : option?.value ?? raw);
  };
  if (schema.type === "boolean") return (
    <label className={`wizard-toggle${disabled ? " is-disabled" : ""}`} htmlFor={id}>
      <input id={id} type="checkbox" checked={Boolean(value)} disabled={disabled}
        aria-describedby={schema.help ? helpId : undefined} onChange={event => change(event.target.checked)} />
      <span><strong>{t(schema.label)}</strong>{schema.help ? <small id={helpId}>{t(schema.help)}</small> : null}</span>
    </label>
  );
  return (
    <label className="wizard-field" htmlFor={id}>
      <span>{t(schema.label)}</span>
      {schema.id === "initializeFrom" ? (
        <select id={id} value={value ?? ""} disabled={disabled} aria-describedby={schema.help ? helpId : undefined}
          onChange={event => change(event.target.value)}>
          <option value="">{t("从头训练")}</option>
          {models.map(model => <option key={model.name} value={model.name}>{model.name}</option>)}
          {value && !models.some(model => model.name === value) ? <option value={value}>{value}</option> : null}
        </select>
      ) : schema.type === "select" ? (
        <select id={id} value={value ?? ""} disabled={disabled} aria-describedby={schema.help ? helpId : undefined}
          onChange={event => change(event.target.value)}>
          {(schema.options ?? []).map(option => <option key={String(option.value)} value={option.value}>{t(option.label)}</option>)}
          {value !== undefined && !(schema.options ?? []).some(option => String(option.value) === String(value))
            ? <option value={value}>{String(value)}</option> : null}
        </select>
      ) : (
        <div className="wizard-input-wrap">
          <input id={id} type={schema.type} value={value ?? ""} disabled={disabled} min={schema.min} max={schema.max} maxLength={schema.maxLength}
            step={schema.integer ? 1 : "any"} required={schema.required ?? Boolean(schema.configKey)}
            placeholder={schema.placeholder ? t(schema.placeholder) : undefined}
            aria-describedby={schema.help ? helpId : undefined} onChange={event => change(event.target.value)} />
          {schema.suffix ? <span>{schema.suffix}</span> : null}
        </div>
      )}
      {schema.help ? <small id={helpId}>{t(schema.help)}</small> : null}
    </label>
  );
}

function TrainingNotices({ parameters, selectedModel, locked, importing, t }) {
  if (importing) return <p className="me-config-note">{t("导入仅转换网络权重；迭代、优化器、判别器和采样状态从零开始。请填写原模型的准确网络结构。")}</p>;
  return <div className="me-config-notices" aria-live="polite">
    {selectedModel ? <p className="me-config-note">{locked
      ? t("当前保留检查点配置。启用允许配置变更后，可调整训练参数；网络结构始终固定。")
      : t("本次续训允许修改训练参数；网络结构保持检查点配置。")}</p> : null}
    {parameters.initializeFrom ? <p className="me-config-note">{t("从所选原生模型复制网络权重，创建新的训练进度；网络结构沿用来源模型。")}</p> : null}
    {parameters.use_fp16 && parameters.cpuOnly ? <p className="me-config-note is-warning" role="alert">{t("FP16 训练需要 CUDA。请选择 GPU，或关闭 FP16。")}</p> : null}
    {parameters.pretrain ? <p className="me-config-note">{t("预训练使用下方专用人脸集，SRC、DST 共用该数据；GAN、TrueFace 和两项风格损失的生效权重均为 0。")}</p> : null}
    {!String(parameters.archi ?? "").startsWith("df") && Number(parameters.true_face_power) > 0
      ? <p className="me-config-note is-warning" role="alert">{t("TrueFace 仅支持 DF 架构；请将 TrueFace 权重设为 0。")}</p> : null}
    {parameters.retraining_samples && Number(parameters.retraining_capacity) < Number(parameters.batchSize)
      ? <p className="me-config-note is-warning" role="alert">{t("回放容量必须至少容纳一个 batch。")}</p> : null}
    {Number(parameters.gan_patch_size) > Number(parameters.resolution)
      ? <p className="me-config-note is-warning" role="alert">{t("GAN Patch 尺寸不能大于模型分辨率。")}</p> : null}
  </div>;
}

export function METrainingFields({ schemas = [], parameters, selectedModel, newModel, models = [], onChange, importing = false }) {
  const { t } = useI18n();
  const prefix = useId();
  const checkpoint = !newModel && !importing ? selectedModel : null;
  const locked = Boolean(checkpoint) && !parameters.allowConfigChange;
  const structuralLocked = !importing && (Boolean(checkpoint) || Boolean(parameters.initializeFrom));
  const changes = meConfigurationChanges(schemas, parameters, checkpoint);
  const fields = schemas.filter(schema => !EXTERNAL_FIELDS.has(schema.id) && schema.id !== "allowConfigChange"
    && (!newModel || !RESUME_OPERATIONS.has(schema.id))
    && (newModel || schema.id !== "initializeFrom")
    && (parameters.pretrain || schema.id !== "pretrainingDataDir"));
  const basicIds = importing ? ["sourceDirectory", "legacyModelName", "batchSize"] : ["resolution", "batchSize"];
  const basicFields = fields.filter(schema => basicIds.includes(schema.id));
  const groups = importing ? ME_CONFIG_GROUPS.filter(group => group.id === "architecture")
    : [{ id: "workflow", label: "数据与运行计划", help: "人脸集路径相对于当前工作区，可填写目录或 PAK / ZIP 文件。" }, ...ME_CONFIG_GROUPS];
  const sourceModels = models.filter(model => model.format === "me-pytorch" && model.ready !== false && model.files?.includes("me.pt"));
  const setField = (id, value) => onChange(current => {
    if (id === "initializeFrom") return initializeMEParameters(schemas, current, sourceModels.find(model => model.name === value));
    return { ...current, [id]: value };
  });
  const renderField = schema => <MEField key={schema.id} schema={schema} value={parameters[schema.id]}
    disabled={Boolean(schema.configKey && (locked || (schema.structural && structuralLocked)))}
    onChange={setField} t={t} prefix={prefix} models={sourceModels} />;
  const allowSchema = schemas.find(schema => schema.id === "allowConfigChange");
  return (
    <section className="me-training-fields" aria-label={t(importing ? "ME 权重导入配置" : "ME 训练配置")}
      onInvalid={event => {
        // Native validation must reveal fields inside collapsed groups before
        // the enclosing wizard reports validity and attempts to focus them.
        let group = event.target.closest("details");
        while (group && event.currentTarget.contains(group)) {
          group.open = true;
          group = group.parentElement?.closest("details");
        }
      }}>
      {!importing ? <div className="me-presets"><span>{t("快速配置")}</span><div>
        {PRESETS.map(preset => <button key={preset.id} type="button" className="button secondary" disabled={locked}
          onClick={() => onChange(current => applyMEPreset(schemas, current, getMePreset(preset.id), { structuralLocked }))}>
          {t(preset.label)}
        </button>)}
      </div><small>{t("预设仅调整所列训练参数，仍可逐项修改。")}</small></div> : null}
      {checkpoint && allowSchema ? renderField(allowSchema) : null}
      <TrainingNotices parameters={parameters} selectedModel={checkpoint} locked={locked} importing={importing} t={t} />
      {basicFields.length ? <div className="wizard-form-grid me-basic-fields">{basicFields.map(renderField)}</div> : null}
      {groups.map(group => {
        const grouped = fields.filter(schema => (schema.group ?? (schema.configKey ? "optimization" : "workflow")) === group.id && !basicIds.includes(schema.id));
        if (!grouped.length) return null;
        return <details key={group.id} className="me-config-group" open={group.id === "workflow" || importing}>
          <summary><strong>{t(group.label)}</strong><span>{t("{count} 项", { count: grouped.length })}</span></summary>
          {group.help ? <p>{t(group.help)}</p> : null}
          <div className="wizard-form-grid">{grouped.map(renderField)}</div>
        </details>;
      })}
      {checkpoint && changes.length ? <div className="me-change-summary" role="status">
        <strong>{t("与检查点的配置差异")} · {changes.length}</strong>
        <dl>{changes.map(change => <div key={change.schema.id}><dt>{t(change.schema.label)}</dt>
          <dd><span>{String(change.from)}</span> → <span>{String(change.to)}</span></dd></div>)}</dl>
        <button type="button" className="text-button" onClick={() => onChange(current => ({
          ...current, ...modelMEParameters(schemas, checkpoint), allowConfigChange: false, resetOptimizer: false, resetDataState: false,
        }))}>{t("恢复检查点配置")}</button>
      </div> : null}
    </section>
  );
}
