import { ME_CONFIG_GROUPS } from "../../shared/me-training-options.mjs";
import { useI18n } from "../i18n.jsx";
import "./METrainingReview.css";

const SUMMARY_FIELDS = new Set(["forceModelName", "sourceDirectory", "legacyModelName", "initializeFrom", "pretrainingDataDir", "srcFaceset", "dstFaceset", "evaluationSrcFaceset", "evaluationDstFaceset", "targetIterations", "stopAtTarget", "archi", "resolution", "batchSize", "pretrain", "use_rg", "use_fp16", "optimizer_on_cpu"]);
const OPTIONAL_EVALUATION_FIELDS = new Set(["evaluationSrcFaceset", "evaluationDstFaceset"]);
const OMITTED_FIELDS = new Set(["gpuIndexes", "cpuOnly", "silentStart"]);

export function METrainingReview({ schemas, parameters, importing, preflight }) {
  const { t } = useI18n();
  const formatValue = (schema, value) => {
    if (schema.type === "boolean") return value ? t("是") : t("否");
    if (schema.type === "select") return schema.options?.find(option => String(option.value) === String(value))?.label ?? String(value);
    return value === "" || value == null ? t("未设置") : String(value);
  };
  const renderRows = fields => fields.map(schema => <div key={schema.id}><dt>{schema.label}</dt><dd>{formatValue(schema, parameters[schema.id])}</dd></div>);
  const visibleSummary = schemas.filter(schema => SUMMARY_FIELDS.has(schema.id)
    && !(parameters.pretrain && ["srcFaceset", "dstFaceset"].includes(schema.id))
    && !(!parameters.pretrain && schema.id === "pretrainingDataDir")
    && !(schema.id === "initializeFrom" && !parameters.initializeFrom)
    && !(OPTIONAL_EVALUATION_FIELDS.has(schema.id) && !parameters[schema.id]));
  const changes = preflight?.training?.changes ?? [];
  const groups = [...ME_CONFIG_GROUPS, { id: "workflow", label: "训练流程" }];
  return <div className="me-training-review">
    <dl className="wizard-review">
      <div><dt>{t("操作")}</dt><dd>{importing ? t("导入网络权重") : parameters.debugSamples ? t("仅检查采样") : parameters.pretrain ? t("预训练") : parameters.silentStart ? t("继续训练") : t("新建训练")}</dd></div>
      <div><dt>{t("运行设备")}</dt><dd>{importing || parameters.cpuOnly ? t("CPU") : parameters.gpuIndexes ? `GPU ${parameters.gpuIndexes}` : t("自动选择 GPU")}</dd></div>
      {renderRows(visibleSummary)}
    </dl>
    {importing ? <p className="wizard-summary-note">{t("只导入网络权重；优化器、判别器和训练进度从零开始。")}</p> : null}
    {parameters.pretrain ? <p className="wizard-summary-note">{t("预训练使用下方专用人脸集，SRC、DST 共用该数据；GAN、TrueFace 和两项风格损失的生效权重均为 0。")}</p> : null}
    {changes.length ? <section className="me-change-review" aria-label={t("检查点配置变更")}>
      <strong>{t("检查点配置变更")}</strong>
      <dl className="wizard-review">{changes.map(change => {
        const schema = schemas.find(field => field.configKey === change.field) ?? {};
        return <div key={change.field}>
          <dt>{schema.label ?? change.field}</dt>
          <dd>{formatValue(schema, change.previous)} → {formatValue(schema, change.next)}</dd>
        </div>;
      })}</dl>
    </section> : null}
    {groups.map(group => {
      const fields = schemas.filter(schema => schema.group === group.id && !SUMMARY_FIELDS.has(schema.id) && !OMITTED_FIELDS.has(schema.id));
      return fields.length ? <details className="me-review-group" key={group.id}>
        <summary>{t(group.label)}</summary><dl className="wizard-review">{renderRows(fields)}</dl>
      </details> : null;
    })}
  </div>;
}
