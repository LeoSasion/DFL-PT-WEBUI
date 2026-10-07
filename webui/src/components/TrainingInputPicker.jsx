import { useEffect, useRef, useState } from "react";
import { runtimeApi } from "../runtime/api.js";
import { useI18n } from "../i18n.jsx";

export function TrainingInputPicker({ projectKey, value, onChange, configuration, onApply }) {
  const { t } = useI18n();
  const [data, setData] = useState(null), [plans, setPlans] = useState({ src: [], dst: [] });
  const [error, setError] = useState(null), [busy, setBusy] = useState(false), [refresh, setRefresh] = useState(0);
  const scope = useRef(projectKey); scope.current = projectKey;
  useEffect(() => {
    if (value?.projectKey === projectKey) setData(value);
  }, [value, projectKey]);
  useEffect(() => {
    let disposed = false;
    setData(null); setError(null); setBusy(true);
    if (!projectKey) { setBusy(false); return undefined; }
    void Promise.all([runtimeApi.trainingInputs({ projectKey }), runtimeApi.bestFacesetPlans("src"), runtimeApi.bestFacesetPlans("dst")])
      .then(([inputs, src, dst]) => { if (!disposed) { setData(inputs); setPlans({ src, dst }); onChange?.(inputs); } })
      .catch(failure => { if (!disposed) setError(failure.message); })
      .finally(() => { if (!disposed) setBusy(false); });
    return () => { disposed = true; };
  }, [projectKey, refresh]);
  const save = async (side, selected) => {
    if (busy || !data) return;
    const key = projectKey; setBusy(true); setError(null);
    try {
      const result = await runtimeApi.saveTrainingInput({ projectKey: key, expectedRevision: data.revision, side,
        selection: selected === "aligned" ? { kind: "aligned" } : { kind: "selected", planId: selected } });
      if (scope.current === key) { setData(result); onChange?.(result); }
    } catch (failure) { if (scope.current === key) setError(failure.message); }
    finally { if (scope.current === key) setBusy(false); }
  };
  const ready = !busy && data?.sides?.src?.valid && data?.sides?.dst?.valid;
  const paths = data ? { srcFaceset: data.sides.src.path, dstFaceset: data.sides.dst.path } : null;
  const differs = paths && (paths.srcFaceset !== configuration?.srcFaceset || paths.dstFaceset !== configuration?.dstFaceset);
  return <section className="training-input-picker" aria-label={t("项目训练输入")}>
    <header><div><strong>{t("项目采用的人脸集")}</strong><p>{t("两侧选择独立保存；填入本次任务后仍需检查数据变更。")}</p></div>
      <button type="button" className="text-button" disabled={busy} onClick={() => setRefresh(current => current + 1)}>{t("重新检查")}</button></header>
    {error ? <p role="alert" className="project-field-error">{t(error)}</p> : null}
    <div className="training-input-cards">{["src", "dst"].map(side => {
      const current = data?.sides?.[side], options = plans[side].filter(plan => plan.state === "published");
      return <article key={side} className={current?.valid === false ? "is-invalid" : ""}>
        <label><span>{side.toUpperCase()} · {t(side === "src" ? "来源人物" : "目标人物")}</span>
          <select aria-label={`${side.toUpperCase()} ${t("训练输入")}`} disabled={busy || !data}
            value={current?.kind === "selected" ? current.planId : "aligned"} onChange={event => void save(side, event.target.value)}>
            <option value="aligned">{t("原始 aligned")}</option>
            {current?.kind === "selected" && !options.some(plan => plan.planId === current.planId) ? <option value={current.planId}>{t("上次选集 · 已失效")}</option> : null}
            {options.map(plan => <option key={plan.planId} value={plan.planId}>{t("选集")} {plan.createdAt ? new Date(plan.createdAt).toLocaleString() : plan.planId.slice(-8)} · {plan.counts?.selected ?? "—"} {t("张")} · v{plan.reviewVersion ?? 0}</option>)}
          </select></label>
        <p>{current ? <>{current.count ?? "—"} {t("张")} · {t(current.kind === "selected" ? "独立 selected 副本" : "采用原始人脸")}</> : t("正在读取…")}</p>
        {current?.identityReferences?.length ? <small>{t("已确认 {count} 张同人参照", { count: current.identityReferences.length })}</small> : null}
        {current?.path ? <code title={current.path}>{current.path}</code> : null}
        {current?.validationMessage ? <p role="status" className="project-field-error">{t(current.validationMessage)}</p> : null}
      </article>;
    })}</div>
    <button type="button" className="button secondary" disabled={!ready || !differs} onClick={() => onApply?.(paths)}>
      {t(!data ? (busy ? "正在读取…" : "将两侧选择填入本次配置") : differs ? "将两侧选择填入本次配置" : "本次配置已采用两侧选择")}</button>
    {differs && ready ? <small>{t("已有模型会继续使用其当前数据配置，点击后才提出更换；采样状态确认仍需完成。")}</small> : null}
  </section>;
}
