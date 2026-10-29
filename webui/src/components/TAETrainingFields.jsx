import { useId } from "react";
import { useI18n } from "../i18n.jsx";
import "./METrainingFields.css";

const EXTERNAL = new Set(["forceModelName", "silentStart", "gpuIndexes", "cpuOnly"]);
const BASIC = new Set(["pretraining", "vae_weights", "resolution", "batchSize", "decoder_mode", "precision", "training_mask_mode"]);
const GROUPS = [
  { id: "workflow", label: "数据与运行计划" },
  { id: "architecture", label: "TAEQI2.1 与 LIAE 中间层" },
  { id: "optimization", label: "优化参数" },
  { id: "loss", label: "重建与局部细节损失" },
  { id: "augmentation", label: "输入增强与采样" },
];

export function TAETrainingFields({ schemas = [], parameters, selectedModel, newModel, onChange }) {
  const { t } = useI18n();
  const prefix = useId();
  const checkpoint = !newModel && selectedModel;
  const fields = schemas.filter(schema => !EXTERNAL.has(schema.id)
    && !(parameters.pretraining && ["srcFaceset", "dstFaceset"].includes(schema.id))
    && !(!parameters.pretraining && schema.id === "pretrainingDataDir")
    && !(newModel && schema.id === "resetDataState"));
  const renderField = schema => {
    const id = `${prefix}-${schema.id}`;
    const disabled = Boolean(checkpoint && (schema.id === "pretraining" || (schema.configKey && !schema.resumeMutable)));
    const value = parameters[schema.id];
    const update = raw => {
      const option = schema.type === "select" ? schema.options?.find(item => String(item.value) === String(raw)) : null;
      onChange(current => ({ ...current, [schema.id]: schema.type === "boolean" ? Boolean(raw) : option?.value ?? raw }));
    };
    return schema.type === "boolean" ? <label key={schema.id} className={`wizard-toggle${disabled ? " is-disabled" : ""}`} htmlFor={id}>
      <input id={id} type="checkbox" checked={Boolean(value)} disabled={disabled}
        aria-describedby={schema.help ? `${id}-help` : undefined} onChange={event => update(event.target.checked)} />
      <span><strong>{t(schema.label)}</strong>{schema.help ? <small id={`${id}-help`}>{t(schema.help)}</small> : null}</span>
    </label> : <label key={schema.id} className="wizard-field" htmlFor={id}
      style={schema.id === "vae_weights" ? { gridColumn: "1 / -1" } : undefined}>
      <span>{t(schema.label)}</span>
      {schema.type === "select" ? <select id={id} value={value ?? ""} disabled={disabled}
        aria-describedby={schema.help ? `${id}-help` : undefined} onChange={event => update(event.target.value)}>
        {(schema.options ?? []).map(option => <option key={String(option.value)} value={option.value}>{t(option.label)}</option>)}
      </select> : <div className="wizard-input-wrap">
        <input id={id} type={schema.type} value={value ?? ""} disabled={disabled}
          min={schema.min} max={schema.max} maxLength={schema.maxLength}
          step={schema.step ?? (schema.integer ? 1 : "any")}
          required={schema.required ?? Boolean(schema.configKey || schema.id === "pretrainingDataDir")}
          placeholder={schema.placeholder} aria-describedby={schema.help ? `${id}-help` : undefined}
          onChange={event => update(event.target.value)} />
        {schema.suffix ? <span>{schema.suffix}</span> : null}
      </div>}
      {schema.help ? <small id={`${id}-help`}>{t(schema.help)}</small> : null}
      {schema.id === "training_mask_mode" && value === "expanded_feather" ? <small>
        {t("按 {resolution} 分辨率向外扩展 {radius} 像素；权重从人脸边界的 1 线性降至外缘的 0，整个新增区域均参与羽化。", {
          resolution: parameters.resolution,
          radius: Math.max(1, Math.round(Number(parameters.resolution) * .1)),
        })}
      </small> : null}
    </label>;
  };
  return <section className="me-training-fields" aria-label={t("TAEQI2.1 训练配置")}
    onInvalid={event => {
      let details = event.target.closest("details");
      while (details && event.currentTarget.contains(details)) {
        details.open = true;
        details = details.parentElement?.closest("details");
      }
    }}>
    <div className="me-config-notices">
      <p className="me-config-note">{t("TAEQI2.1 始终冻结 Encoder，训练独立的 InterAB / InterB 和脸部遮罩头。Decoder 可选择冻结、仅训练末段或训练全部。")}</p>
      {parameters.pretraining ? <p className="me-config-note">{t("通用预训练使用下方所选素材池，自动检查图片、排除重复并留出验证样本。数据检查完成后开始训练计时。")}</p> : null}
      {checkpoint ? <p className="me-config-note">{t("继续训练可调整眼嘴区域权重、基础学习率、训练精度、Decoder 范围、学习率倍率、Batch size、随机形变、色彩迁移和遮罩训练范围。变更后保留权重、总迭代与采样状态，重置 Adam 和预热并记录新阶段。网络结构与其他损失权重保持不变。")}</p> : null}
    </div>
    <div className="wizard-form-grid me-basic-fields">{fields.filter(schema => BASIC.has(schema.id)).map(renderField)}</div>
    {GROUPS.map(group => {
      const grouped = fields.filter(schema => !BASIC.has(schema.id)
        && (schema.group ?? (schema.configKey ? "optimization" : "workflow")) === group.id);
      return grouped.length ? <details key={group.id} className="me-config-group" open={group.id === "workflow"}>
        <summary><strong>{t(group.label)}</strong><span>{t("{count} 项", { count: grouped.length })}</span></summary>
        <div className="wizard-form-grid">{grouped.map(renderField)}</div>
      </details> : null;
    })}
  </section>;
}
