import { useEffect, useRef, useState } from "react";
import { IconArrowRight, IconCheck, IconChevronLeft, IconChevronRight, IconPlayerStop, IconRefresh } from "@tabler/icons-react";
import { runtimeApi } from "../runtime/api.js";
import { useI18n } from "../i18n.jsx";
import { LoadingProgress } from "./ProgressFeedback.jsx";
import { BestFacesetPreview } from "./BestFacesetPreview.jsx";
import { bestFacesetIdentityContract } from "../domain/best-faceset-identity.js";
import { bestFacesetDraftState, bestFacesetHistoryLabel, bestFacesetImageLabel, bestFacesetReasonText, bestFacesetPublicationPreviewMatches, createBestFacesetDraftWriter, flushBestFacesetDraft, registerBestFacesetDraftWriter } from "../domain/best-faceset-workflow.js";
import "./best-faceset.css";

const PAGE = 60;
const STATUS = { selected: "推荐训练", review: "待复核或备选", rejected: "低价值或错误" };
const MANUAL = { keep: "保留", exclude: "排除", defer: "暂缓" };
const STATE = { analyzing: "待完成分析", finalized: "待发布", published: "已发布", withdrawn: "推荐已撤回" };
const previewItem = (item, side, pageIndex) => ({ ...item, name: bestFacesetImageLabel(item), pageIndex,
  imageUrl: item.imageUrl ?? `/api/assets/${side}/aligned/${encodeURIComponent(bestFacesetImageLabel(item))}` });
const formatDate = value => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
};

export function BestFacesetPanel({ side, projectKey, refreshVersion, initialPlanId, initialPlanRequest, onError, onNotice, onOpenCommand, onUseTrainingInput, onBusyChange }) {
  const { t } = useI18n();
  const [inventory, setInventory] = useState(null), [plans, setPlans] = useState([]), [plan, setPlan] = useState(null);
  const [resources, setResources] = useState(null), [qualityModel, setQualityModel] = useState("foreground_tenengrad");
  const [offset, setOffset] = useState(0), [targetCount, setTargetCount] = useState(2000), [refs, setRefs] = useState(new Set());
  const [confirmed, setConfirmed] = useState(false), [category, setCategory] = useState("all");
  const [busy, setBusy] = useState(null), [progress, setProgress] = useState(null), [preview, setPreview] = useState(null);
  const [previewBusy, setPreviewBusy] = useState(false), [reviewMembers, setReviewMembers] = useState(new Set());
  const [dryRun, setDryRun] = useState(null), [recovery, setRecovery] = useState(null), [loadError, setLoadError] = useState(null);
  const [revision, setRevision] = useState(0), [draftReload, setDraftReload] = useState(0), [draftReady, setDraftReady] = useState(false);
  const [draftStatus, setDraftStatus] = useState("loading"), [draftWarning, setDraftWarning] = useState(null), [sourceFingerprint, setSourceFingerprint] = useState(null);
  const requestVersion = useRef(0), operation = useRef(null), observation = useRef(null), draftWriter = useRef(null), gridRef = useRef(null), previewOpener = useRef(null), loadedInitialPlan = useRef(null);
  const scope = `${projectKey ?? ""}:${side}`, currentScope = useRef(scope), mounted = useRef(false);
  currentScope.current = scope;
  const active = Boolean(busy), blocked = active || !draftReady || draftStatus === "conflict";
  useEffect(() => { onBusyChange?.(active); return () => onBusyChange?.(false); }, [active, onBusyChange]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; observation.current?.abort(); void draftWriter.current?.flush(); }; }, []);

  useEffect(() => {
    const token = ++requestVersion.current; let cancelled = false;
    setDraftReady(false); setDraftStatus("loading"); setPlan(null); setInventory(null); setPreview(null); setOffset(0); setCategory("all");
    setRefs(new Set()); setConfirmed(false); setDryRun(null); setRecovery(null); setReviewMembers(new Set()); setLoadError(null);
    const valid = () => !cancelled && mounted.current && currentScope.current === scope && token === requestVersion.current;
    Promise.all([flushBestFacesetDraft(scope).then(() => runtimeApi.bestFacesetDraft(side, { projectKey })), runtimeApi.bestFacesetPlans(side), runtimeApi.bestFacesetStatus()]).then(async ([record, history, status]) => {
      if (!valid()) return;
      const restored = bestFacesetDraftState(record);
      const view = restored?.planId
        ? await runtimeApi.bestFacesetPlan(side, restored.planId, { offset: restored.offset, limit: PAGE, status: restored.category === "all" ? null : restored.category })
        : await runtimeApi.alignedAssets(side, { offset: restored?.offset ?? 0, limit: PAGE });
      if (!valid()) return;
      if (restored?.planId) setPlan({ ...view, sourceValid: view.sourceValid ?? record.planSourceValid }); else setInventory(view);
      setPlans(history); setResources(status); setSourceFingerprint(record.sourceFingerprint);
      if (restored) {
        setRefs(new Set(restored.identityReferences)); setConfirmed(restored.confirmed); setOffset(restored.offset); setCategory(restored.category);
        setTargetCount(restored.targetCount); setQualityModel(restored.qualityModel);
      }
      setDraftWarning(record.validationMessage ?? null);
      const writer = createBestFacesetDraftWriter({ revision: record.revision,
        save: payload => runtimeApi.saveBestFacesetDraft(side, { ...payload, projectKey }),
        onSaved: result => {
          if (!mounted.current || currentScope.current !== scope || draftWriter.current !== writer) return;
          setDraftStatus("saved");
          if (!result.sourceValid || result.draft?.confirmed && !result.identityConfirmationValid) {
            setConfirmed(false); setDraftWarning(result.validationMessage || t("原素材或参照已变化，请重新确认人物。"));
          } else setDraftWarning(result.validationMessage ?? null);
          if (result.sourceFingerprint) setSourceFingerprint(result.sourceFingerprint);
        },
        onError: error => {
          if (!mounted.current || currentScope.current !== scope || draftWriter.current !== writer) return;
          setDraftStatus("conflict"); setDraftWarning(error.message); onError(error);
        } });
      draftWriter.current = writer;
      registerBestFacesetDraftWriter(scope, writer);
      setDraftStatus(restored ? "restored" : "saved"); setDraftReady(true);
    }).catch(error => {
      if (!valid()) return;
      setLoadError(error.message); setDraftStatus("conflict"); onError(error);
    });
    return () => { cancelled = true; void draftWriter.current?.flush(); };
  }, [scope, draftReload]);

  useEffect(() => {
    if (!draftReady || draftWriter.current?.paused) return;
    const numericTarget = Number(targetCount);
    if (!Number.isInteger(numericTarget) || numericTarget < 1 || numericTarget > 500000) return;
    setDraftStatus("saving");
    draftWriter.current?.enqueue({ planId: plan?.planId ?? null, identityReferences: [...refs], confirmed,
      offset, category, targetCount: numericTarget, qualityModel: plan?.qualityModel ?? qualityModel, sourceFingerprint });
  }, [draftReady, plan?.planId, plan?.qualityModel, refs, confirmed, offset, category, targetCount, qualityModel, sourceFingerprint]);

  useEffect(() => {
    if (!draftReady) return;
    const token = ++requestVersion.current;
    const input = plan ? runtimeApi.bestFacesetPlan(side, plan.planId, { offset, limit: PAGE, status: category === "all" ? null : category }) : runtimeApi.alignedAssets(side, { offset, limit: PAGE });
    Promise.all([input, runtimeApi.bestFacesetPlans(side)]).then(([next, history]) => {
      if (token !== requestVersion.current || currentScope.current !== scope || !mounted.current) return;
      if (plan) setPlan(next); else setInventory(next);
      setPlans(history);
      if (preview) {
        const item = next.items.find(value => bestFacesetImageLabel(value) === preview.name);
        if (item) setPreview(previous => previous ? ({ ...previous, ...previewItem(item, side, previous.pageIndex) }) : null);
      }
    }).catch(error => { if (token === requestVersion.current && mounted.current) { setLoadError(error.message); onError(error); } });
    return () => { requestVersion.current++; };
  }, [draftReady, side, plan?.planId, offset, category, revision, refreshVersion, scope]);

  useEffect(() => {
    if (!draftReady || !initialPlanId) return;
    const focusKey = `${scope}:${initialPlanId}:${initialPlanRequest ?? ""}`;
    if (loadedInitialPlan.current === focusKey) return;
    loadedInitialPlan.current = focusKey;
    const token = ++requestVersion.current;
    runtimeApi.bestFacesetPlan(side, initialPlanId, { offset: 0, limit: PAGE }).then(next => {
      if (token !== requestVersion.current || !mounted.current || currentScope.current !== scope) return;
      setPlan(next); setOffset(0); setCategory("all"); setPreview(null); setReviewMembers(new Set()); setDryRun(null); setRecovery(null);
      setRefs(new Set(next.identityReferences ?? []));
      setConfirmed(Boolean(next.sourceValid === true && next.identityReferences?.length));
      if (next.sourceValid === false) setDraftWarning(next.sourceValidationMessage || t("原素材已变化，请新建选集并重新确认人物。"));
    }).catch(error => { if (token === requestVersion.current && mounted.current) onError(error); });
  }, [draftReady, scope, initialPlanId, initialPlanRequest]);

  const items = plan?.items ?? inventory?.items ?? [], total = plan?.total ?? inventory?.total ?? 0;
  const visibleTotal = plan?.selectedRange?.total ?? total, rangeEnd = Math.min(offset + items.length, visibleTotal);
  const counts = plan?.publication?.counts ?? plan?.selection?.counts, publication = plan?.publication;
  const identityContract = bestFacesetIdentityContract(plan, refs, confirmed);
  const sourceChanged = plan?.sourceValid === false;
  const editableReview = Boolean(plan?.parentPlanId && plan.state === "finalized" && !publication && !identityContract.dirty && !sourceChanged);
  const summary = plan?.reviewSummary;
  const dryRunCurrent = bestFacesetPublicationPreviewMatches(plan, dryRun);
  const canStart = !blocked && Boolean(total) && Number.isInteger(Number(targetCount)) && Number(targetCount) >= 1 && Number(targetCount) <= 500000 && confirmed && refs.size > 0;
  const toggleRef = name => {
    setConfirmed(false); setDryRun(null); setRefs(current => {
      const next = new Set(current); if (next.has(name)) next.delete(name); else if (next.size < 32) next.add(name); return next;
    });
  };
  const useResult = (result, retainPreview = false) => {
    if (result?.planId && result.items) {
      setPlan(result); setOffset(result.selectedRange?.offset ?? 0);
      setReviewMembers(new Set());
      setPreview(previous => {
        if (!previous) return null;
        const item = result.items.find(value => bestFacesetImageLabel(value) === previous.name);
        const next = item ?? (retainPreview ? result.items[Math.min(previous.pageIndex, result.items.length - 1)] : null);
        return next ? { ...previous, ...previewItem(next, side, result.items.indexOf(next)) } : null;
      });
    }
  };
  const run = async (action, fn) => {
    const operationScope = scope;
    requestVersion.current++; observation.current?.abort(); const controller = new AbortController(); observation.current = controller;
    setBusy(action); setProgress(null); setLoadError(null);
    try {
      const result = await fn({ signal: controller.signal, onProgress: value => { operation.current = value.id; setProgress(value); } });
      if (!controller.signal.aborted && mounted.current && currentScope.current === operationScope) {
        useResult(result, action === "review-decide" || action === "review-undo"); setRevision(value => value + 1); return result;
      }
    } catch (error) {
      if (error?.name !== "AbortError" && mounted.current && currentScope.current === operationScope) { setLoadError(error.message); onError(error); }
    } finally { if (observation.current === controller && mounted.current) { setBusy(null); operation.current = null; observation.current = null; } }
    return null;
  };
  const create = async () => {
    setDryRun(null); setRecovery(null);
    const result = await run("create", options => runtimeApi.createBestFaceset(side,
      { targetCount: Number(targetCount), identityReferences: confirmed ? [...refs] : [], qualityModel }, options));
    if (result) { setCategory("all"); onNotice(t("完整计划已建立，可以分批分析全部人脸。")); }
  };
  const start = async () => {
    setDryRun(null); setRecovery(null); const references = [...refs];
    const result = await run("start", async options => {
      let next = plan;
      if (!next) next = await runtimeApi.createBestFaceset(side, { targetCount: Number(targetCount), identityReferences: references, qualityModel }, options);
      if (options.signal.aborted) return next;
      useResult(next); setCategory("all");
      if (!next.globalReady) next = await runtimeApi.analyzeBestFaceset(side, next.planId, { offset: next.nextOffset ?? 0, limit: 500, all: true, device: "cpu" }, options);
      if (options.signal.aborted) return next;
      useResult(next);
      return runtimeApi.selectBestFaceset(side, next.planId, references, options);
    });
    if (result) { setOffset(0); setCategory("all"); onNotice(t("代表性选集已生成，请查看结果，再明确保存副本。")); }
  };
  const analyze = all => run("analyze", options => runtimeApi.analyzeBestFaceset(side, plan.planId, { offset: plan.nextOffset ?? 0, limit: 500, all, device: "cpu" }, options));
  const select = async () => {
    setDryRun(null);
    const result = await run("select", options => runtimeApi.selectBestFaceset(side, plan.planId, confirmed ? [...refs] : [], options));
    if (result) { setOffset(0); setCategory("all"); onNotice(t("全局选集已生成，请复核边界样本与保留理由。")); }
  };
  const publish = async isDry => {
    if (!identityContract.canPublish || sourceChanged) return;
    const result = await run(isDry ? "dry-run" : "publish", options => runtimeApi.publishBestFaceset(side, plan.planId, isDry,
      { ...options, expectedReferences: identityContract.expectedReferences, expectedRevision: plan.reviewRevision ?? 0 }));
    if (result && isDry) setDryRun({ ...result, clientPlanId: plan.planId, clientReviewRevision: plan.reviewRevision ?? 0 });
    if (!result) setDryRun(null);
    if (result && !isDry) { setDryRun(null); onNotice(t("三类独立副本已保存，原素材全部保留。")); }
  };
  const recover = async isDry => {
    const result = await run(isDry ? "recover-check" : "recover", options => runtimeApi.recoverBestFaceset(side, plan.planId, isDry, options));
    if (result) { setRecovery(result); if (!isDry) { setDryRun(null); onNotice(t("推荐已撤回，原件与已发布副本均已保留。")); } }
  };
  const load = async id => {
    setPreview(null); setReviewMembers(new Set());
    if (!id) { setPlan(null); setOffset(0); setCategory("all"); setInventory(null); setRefs(new Set()); setConfirmed(false); setDryRun(null); setRecovery(null); return; }
    const token = ++requestVersion.current;
    try {
      const next = await runtimeApi.bestFacesetPlan(side, id, { offset: 0, limit: PAGE });
      if (token !== requestVersion.current || !mounted.current) return;
      setPlan(next); setOffset(0); setCategory("all"); setDryRun(null); setRecovery(null);
      // Loading a history entry never turns its references into a new human confirmation.
    } catch (error) { if (token === requestVersion.current) onError(error); }
  };
  const forkReview = async () => {
    setDryRun(null);
    const result = await run("review-create", options => runtimeApi.runOperation("best-faceset-review-create", side,
      { planId: plan.planId, expectedRevision: plan.reviewRevision ?? 0, requestId: crypto.randomUUID() }, options));
    if (result) { setCategory("review"); setOffset(0); onNotice(t("人工复核草稿已建立；原算法建议和已发布版本均保留。")); }
  };
  const decide = async (decision, members) => {
    if (blocked || !editableReview || !members.length) return;
    setDryRun(null);
    return run("review-decide", async options => {
      const result = await runtimeApi.runOperation("best-faceset-review-decide", side,
        { planId: plan.planId, expectedRevision: plan.reviewRevision, requestId: crypto.randomUUID(), decision, members, identityReferences: identityContract.expectedReferences }, options);
      return inspectReviewedPage(result);
    });
  };
  const undo = async () => {
    if (blocked || !editableReview || !summary?.canUndo) return;
    setDryRun(null);
    return run("review-undo", async options => {
      const result = await runtimeApi.runOperation("best-faceset-review-undo", side,
        { planId: plan.planId, expectedRevision: plan.reviewRevision, requestId: crypto.randomUUID(), identityReferences: identityContract.expectedReferences }, options);
      return inspectReviewedPage(result);
    });
  };
  const inspectReviewedPage = async result => {
    let next = await runtimeApi.bestFacesetPlan(side, result.planId, { offset, limit: PAGE, status: category === "all" ? null : category });
    const visible = next.selectedRange?.total ?? next.total;
    if (offset >= visible && offset > 0) {
      next = await runtimeApi.bestFacesetPlan(side, result.planId, { offset: Math.max(0, Math.floor((visible - 1) / PAGE) * PAGE), limit: PAGE, status: category === "all" ? null : category });
    }
    return next;
  };
  const openPreview = (item, index, event) => { previewOpener.current = event.currentTarget; setPreview(previewItem(item, side, index)); };
  const closePreview = () => {
    setPreview(null);
    requestAnimationFrame(() => (previewOpener.current?.isConnected ? previewOpener.current : gridRef.current?.querySelector(".best-faceset-image"))?.focus());
  };
  const navigatePreview = async direction => {
    if (!preview || active || previewBusy) return;
    const nextIndex = preview.pageIndex + direction;
    if (nextIndex >= 0 && nextIndex < items.length) { setPreview(previewItem(items[nextIndex], side, nextIndex)); return; }
    const nextOffset = offset + direction * PAGE;
    if (nextOffset < 0 || nextOffset >= visibleTotal) return;
    setPreviewBusy(true); const token = ++requestVersion.current;
    try {
      const next = plan ? await runtimeApi.bestFacesetPlan(side, plan.planId, { offset: nextOffset, limit: PAGE, status: category === "all" ? null : category })
        : await runtimeApi.alignedAssets(side, { offset: nextOffset, limit: PAGE });
      if (token !== requestVersion.current || !mounted.current) return;
      if (plan) setPlan(next); else setInventory(next);
      setOffset(nextOffset); const index = direction > 0 ? 0 : next.items.length - 1;
      if (next.items[index]) setPreview(previewItem(next.items[index], side, index));
    } catch (error) { if (mounted.current) onError(error); }
    finally { if (mounted.current) setPreviewBusy(false); }
  };
  const changeCategory = value => { setCategory(value); setOffset(0); setPreview(null); setReviewMembers(new Set()); };
  const selectedForReview = items.filter(item => reviewMembers.has(bestFacesetImageLabel(item)));
  return <section className="best-faceset-panel" aria-label={t("最佳训练人脸")}>
    <header><div><h3>{t("最佳训练人脸")}</h3><p>{t("剔除坏图，减少重复，保留角度、眼口状态和光照覆盖。")}</p></div><span className="best-faceset-badge">{side.toUpperCase()} · {t("原 aligned 保留")}</span></header>
    <div className="best-faceset-draft-status" role="status"><span>{t({ loading: "读取项目草稿…", restored: "已恢复上次草稿与浏览位置", saving: "正在保存草稿…", saved: "草稿已自动保存", conflict: "草稿保存已暂停" }[draftStatus])}</span>
      {plan && <span>{t(STATE[plan.state] ?? plan.state)} · {t(plan.parentPlanId ? "人工修订版本 {version}" : "算法建议", { version: plan.reviewVersion })}</span>}
      {draftStatus === "conflict" && <button className="button secondary" disabled={active} type="button" onClick={() => setDraftReload(value => value + 1)}>{t("重新读取项目草稿")}</button>}</div>
    {draftWarning && <p className="best-faceset-error" role="alert">{t(draftWarning)}</p>}
    {sourceChanged && <p className="best-faceset-error" role="alert">{t("原素材已变化，这份历史结果只能查看。请新建选集并重新确认人物。")}</p>}
    <div className="best-faceset-identity"><strong>{t("1 · 确认目标人物")}</strong><p>{t("从下方选取 1–32 张清楚的同人参照，可包含不同角度。参照定义目标身份，与人工保留分别记录。")}</p>
      <div className="best-faceset-reference-list">{[...refs].map(name => <button key={name} type="button" disabled={blocked || Boolean(publication)} onClick={() => toggleRef(name)} title={name} aria-label={t("移除同人参照 {name}", { name })}>{name}<span aria-hidden="true"> ×</span></button>)}</div>
      <label><input type="checkbox" checked={confirmed} disabled={blocked || !refs.size} onChange={event => { setConfirmed(event.target.checked); setDryRun(null); }}/>{t("已确认所选 {count} 张属于同一目标人物", { count: refs.size })}</label>
      {plan && identityContract.dirty && <button className="button secondary" type="button" disabled={blocked || sourceChanged} onClick={() => { setRefs(new Set(plan.identityReferences)); setConfirmed(false); setDryRun(null); }}>{t("载入此选集的参照，重新确认")}</button>}
      {!confirmed && <small>{t("先勾选并确认参照，再开始。也可在详细控制中仅分析素材，稍后确认人物。")}</small>}</div>
    <div className="best-faceset-start"><label>{t("2 · 推荐数量上限")}<input type="number" min="1" max="500000" value={plan?.targetCount ?? targetCount} disabled={blocked || Boolean(plan)} onChange={event => setTargetCount(event.target.value)}/></label>
      <p>{t("当前 aligned：{total} 张。数量是上限；重复过多时提前停止，不为凑数补入。", { total })}</p>
      {!publication && plan?.state !== "withdrawn" && !plan?.parentPlanId && <button className="button primary" type="button" disabled={!canStart || sourceChanged} onClick={() => void start()}>{t(plan ? "继续分析并生成选集" : "分析并生成选集")}<IconArrowRight size={15}/></button>}
      {plan && <button className="button secondary" type="button" disabled={blocked} onClick={() => void load("")}>{t("新建选集")}</button>}</div>
    {plan?.selection && identityContract.dirty && <p className="best-faceset-error" role="alert">{t(plan.parentPlanId || publication ? "当前参照与这份选集不一致。载入该版本的参照并重新确认，或新建选集；旧版本不会改写。" : "身份参照或确认状态已变化，下方是旧选集；请重新生成代表性训练子集后再发布。")}</p>}
    <div className="best-faceset-progress" aria-live="polite"><strong>{t("当前查看 {start}–{end} / {total}", { start: items.length ? offset + 1 : 0, end: items.length ? rangeEnd : 0, total: visibleTotal })}</strong>
      <span>{t("分析每批最多 500 张；查看每页 60 张")}</span>{plan && <><span>{t("已分析 {done} / {total}，剩余 {remaining}", { done: plan.completedCount, total, remaining: total - plan.completedCount })}</span><progress value={plan.completedCount} max={Math.max(1, total)} aria-label={t("全量分析进度")}/></>}</div>
    {active && <div className="best-faceset-active"><LoadingProgress inline className="in-panel" label={t(progress?.stage ?? "准备处理")} detail={progress?.detail} current={progress?.current} total={progress?.total} percent={progress?.percent} elapsedSeconds={progress?.elapsedSeconds} etaSeconds={progress?.etaSeconds}/>
      <button className="button secondary" type="button" disabled={!operation.current || progress?.status === "cancelling"} onClick={() => void runtimeApi.cancelOperation(operation.current).catch(onError)}><IconPlayerStop size={15}/>{t("安全取消")}</button></div>}
    {loadError && <p className="best-faceset-error" role="alert">{t(loadError)}</p>}
    <details className="best-faceset-details"><summary>{t("选集记录与详细控制")}</summary><div className="best-faceset-controls"><label>{t("历史选集")}<select value={plan?.planId ?? ""} disabled={blocked} onChange={event => void load(event.target.value)}><option value="">{t("新建选集")}</option>{plans.map(item => <option value={item.planId} key={item.planId}>{bestFacesetHistoryLabel(item, { formatDate, translate: t })}</option>)}</select></label>
      <button className="button secondary" type="button" disabled={blocked} onClick={() => setRevision(value => value + 1)}><IconRefresh size={15}/>{t("刷新")}</button></div>
      <div className="best-faceset-controls">{!plan ? <button className="button secondary" type="button" disabled={blocked || !total || !Number.isInteger(Number(targetCount)) || Number(targetCount) < 1 || Number(targetCount) > 500000} onClick={() => void create()}>{t("仅建立完整选集计划")}</button> : !publication && plan.state !== "withdrawn" && !plan.parentPlanId && <>
        {!plan.globalReady && <><button className="button secondary" type="button" disabled={blocked || sourceChanged} onClick={() => void analyze(true)}>{t("分析全部人脸（分批 500 张）")}</button><button className="button secondary" type="button" disabled={blocked || sourceChanged} onClick={() => void analyze(false)}>{t("只分析下一批 {start}–{end}", { start: (plan.nextOffset ?? 0) + 1, end: Math.min((plan.nextOffset ?? 0) + 500, total) })}</button></>}
        <button className="button secondary" type="button" disabled={blocked || !plan.globalReady || sourceChanged} onClick={() => void select()}>{t("重新生成代表性训练子集")}</button></>}</div></details>
    <details className="best-faceset-details"><summary>{t("高级信息与研究对照")}</summary><ol className="best-faceset-steps"><li><strong>{t("剔除明显坏图")}</strong><span>{t("现有 YOLO / TUFA 检查检测、关键点与曝光")}</span></li><li><strong>{t("质量达到底线")}</strong><span>{t("既有综合质量以 Tenengrad 为主；Efficient-FIQA 本次未显示替换收益")}</span></li><li><strong>{t("选择代表性组合")}</strong><span>{t("TUFA 状态 + SFace 相似度 + 可靠时间信息")}</span></li></ol>
      <label>{t("质量评分")}<select value={plan?.qualityModel ?? qualityModel} disabled={blocked || Boolean(plan)} onChange={event => setQualityModel(event.target.value)}><option value="foreground_tenengrad">{t("既有综合质量（Tenengrad 为主）")}</option><option value="efficient-fiqa" disabled={!resources?.efficientFiqa?.available}>{t("Efficient-FIQA · 对照评测")}{!resources?.efficientFiqa?.available ? ` · ${t("资源尚未准备")}` : ""}</option></select></label><small>{t("质量分不是身份概率或准确率；只用于底线与同类样本择优。")}</small></details>
    {(plan?.qualityModel ?? qualityModel) === "efficient-fiqa" && <p className="best-faceset-error">{t("当前为研究对照：Efficient-FIQA 尚未达到替换门槛，这份结果不能发布为训练输入。")}</p>}
    {counts && <div className="best-faceset-summary">{Object.keys(STATUS).map(key => <button type="button" key={key} disabled={blocked} className={category === key ? "is-current" : ""} onClick={() => changeCategory(key)}><span>{key}/ · {t(STATUS[key])}</span><strong>{counts[key] ?? 0}</strong></button>)}</div>}
    {plan?.selection && <div className="best-faceset-review"><div className="best-faceset-controls"><label>{t("查看结果分类")}<select value={category} disabled={blocked} onChange={event => changeCategory(event.target.value)}><option value="all">{t("全部")}</option>{Object.keys(STATUS).map(key => <option key={key} value={key}>{key}/ · {t(STATUS[key])}</option>)}</select></label>
      {!editableReview && (!plan.parentPlanId || publication) && <button className="button secondary" type="button" disabled={blocked || sourceChanged || plan.state === "withdrawn" || plan.qualityModel !== "foreground_tenengrad"} onClick={() => void forkReview()}>{t("建立人工复核版本")}</button>}</div>
      {summary && <p>{t("待处理 {pending} 张 · 暂缓 {deferred} 张 · 质量达标备选 {reserve} 张 · 已决定 {decided} 张", { pending: summary.pendingCount, deferred: summary.deferredCount, reserve: summary.adequateReserveCount, decided: summary.decidedCount })}</p>}
      <small>{t("质量达标备选无需全部变成待办。人工修订另存新版本，原建议与原发布记录保留。")}</small>
      {editableReview && <div className="best-faceset-controls"><label><input type="checkbox" checked={Boolean(items.length) && items.every(item => reviewMembers.has(bestFacesetImageLabel(item)))} disabled={blocked || !items.length} onChange={event => setReviewMembers(event.target.checked ? new Set(items.map(bestFacesetImageLabel)) : new Set())}/>{t("选择本页用于批量复核")}</label>
        <span>{t("已选 {count} 张", { count: reviewMembers.size })}</span>
        <button className="button secondary" type="button" disabled={blocked || !selectedForReview.length || !confirmed || selectedForReview.some(item => !item.reviewEligibility?.canKeep)} onClick={() => void decide("keep", [...reviewMembers])}>{t("批量保留")}</button>
        <button className="button secondary" type="button" disabled={blocked || !reviewMembers.size} onClick={() => void decide("exclude", [...reviewMembers])}>{t("批量排除")}</button>
        <button className="button secondary" type="button" disabled={blocked || !reviewMembers.size} onClick={() => void decide("defer", [...reviewMembers])}>{t("批量暂缓")}</button>
        <button className="button secondary" type="button" disabled={blocked || !summary?.canUndo} onClick={() => void undo()}>{t("撤销上次决定")}</button></div>}</div>}
    {!draftReady && !loadError && <p>{t("读取 aligned 人脸与项目草稿…")}</p>}
    {items.length ? <div className="best-faceset-grid" ref={gridRef}>{items.map((item, index) => {
      const name = bestFacesetImageLabel(item), url = item.imageUrl ?? `/api/assets/${side}/aligned/${encodeURIComponent(name)}`;
      return <article key={name} className={`best-faceset-card is-${item.status ?? "pending"}`}><button className="best-faceset-image" type="button" aria-label={t("放大查看 {name}", { name })} onClick={event => openPreview(item, index, event)}><img src={url} alt={name} loading="lazy" decoding="async"/></button>
        <strong title={name}>{name}</strong>{item.status && <span>{item.status}/ · {t(STATUS[item.status])}</span>}<p>{item.reasons?.map(reason => bestFacesetReasonText(reason, t)).join(" / ") || t(item.analyzed ? "已分析，等待全局选集" : "待分析")}</p>
        {item.manualDecision && <span className="best-faceset-manual">{t("人工：{decision}", { decision: t(MANUAL[item.manualDecision.decision]) })}</span>}
        {editableReview && <label className="best-faceset-batch-check"><input type="checkbox" checked={reviewMembers.has(name)} disabled={blocked} onChange={() => setReviewMembers(current => { const next = new Set(current); if (next.has(name)) next.delete(name); else next.add(name); return next; })} aria-label={t("将 {name} 选入批量复核", { name })}/>{t("批量复核")}</label>}
        <label className="best-faceset-reference-check"><input type="checkbox" checked={refs.has(name)} disabled={blocked || Boolean(publication) || refs.size >= 32 && !refs.has(name)} onChange={() => toggleRef(name)} aria-label={t("将 {name} 设为同人参照", { name })}/>{t("同人参照")}</label></article>;
    })}</div> : draftReady && <p>{t(total ? "本页没有该类别样本，请切换页面或查看全部。" : "先提取 aligned 人脸，再选择训练子集。")}</p>}
    <div className="best-faceset-controls"><button className="button secondary" type="button" disabled={blocked || offset === 0} onClick={() => { setOffset(value => Math.max(0, value - PAGE)); setReviewMembers(new Set()); }}><IconChevronLeft size={15}/>{t("上一页")}</button><button className="button secondary" type="button" disabled={blocked || rangeEnd >= visibleTotal} onClick={() => { setOffset(value => value + PAGE); setReviewMembers(new Set()); }}>{t("下一页")}<IconChevronRight size={15}/></button></div>
    {plan?.state === "withdrawn" && <p>{t("该选集推荐已撤回，原件和副本均保留。需要新的推荐时，请新建选集。")}</p>}
    {plan?.selection && !publication && plan.state !== "withdrawn" && plan.qualityModel === "foreground_tenengrad" && <div className="best-faceset-publish"><strong>{t("3 · 检查并保存训练集")}</strong><p>{t("发布复制 selected、review、rejected 三类素材，保留原像素、DFL 元数据和关键点辅助记录。人工修订不会改写旧版本。")}</p>
      <button className="button secondary" type="button" disabled={blocked || !identityContract.canPublish || sourceChanged} onClick={() => void publish(true)}>{t("预演发布并检查原素材")}</button>{dryRunCurrent && !identityContract.dirty && <p>{t("预演通过：推荐 {selected} / 复核 {review} / 排除 {rejected}；原件全部保留。", dryRun.counts)}</p>}
      {dryRun && !dryRunCurrent && <p>{t("人工复核版本已变化，请重新预演并确认发布。")}</p>}
      <button className="button primary" type="button" disabled={blocked || !dryRunCurrent || !identityContract.canPublish || sourceChanged} onClick={() => void publish(false)}><IconCheck size={15}/>{t("保存三类独立副本")}</button></div>}
    {publication && <div className="best-faceset-publish"><strong>{t("Best Training Faceset 已准备")}</strong><code>{publication.outputDirectory}</code><p>{t("selected 推荐训练；review 包含边界情况与可用备选；rejected 包含明显错误与低价值重复。")}</p>
      {publication.trainingAllowed !== false ? <button className="button primary" type="button" disabled={!publication.counts.selected || blocked || plan.state === "withdrawn" || sourceChanged} onClick={() => onUseTrainingInput ? void onUseTrainingInput(side, plan.planId) : onOpenCommand("train.me", { [`${side}Faceset`]: publication.selectedDirectory })}>{t("在训练配置中使用 selected")}<IconArrowRight size={15}/></button> : <p>{t("Efficient-FIQA 本次未显示替换收益，仅用于对照评测；该结果不会推荐给训练。")}</p>}
      {plan.state !== "withdrawn" && <><button className="button secondary" type="button" disabled={blocked} onClick={() => void recover(true)}>{t("检查撤回发布")}</button>{recovery?.dryRun && <><p>{recovery.conflicts?.length ? recovery.conflicts.join("；") : t("副本校验通过，可以撤回推荐并保留全部副本。")}</p><button className="button secondary" type="button" disabled={blocked || recovery.conflicts?.length > 0} onClick={() => void recover(false)}>{t("撤回推荐并保留副本")}</button></>}</>}</div>}
    {preview && <BestFacesetPreview item={preview} current={offset + preview.pageIndex + 1} total={visibleTotal} category={t(category === "all" ? "全部" : STATUS[category])} t={t}
      onClose={closePreview} onNavigate={navigatePreview} navigating={previewBusy} busy={blocked} reference={refs.has(preview.name)}
      referenceDisabled={blocked || Boolean(publication) || refs.size >= 32 && !refs.has(preview.name)} onReference={() => toggleRef(preview.name)}
      reviewActive={editableReview} canKeep={Boolean(confirmed && preview.reviewEligibility?.canKeep)} decision={t(MANUAL[preview.manualDecision?.decision] ?? "未决定")}
      onDecision={decision => decide(decision, [preview.name])} canUndo={summary?.canUndo} onUndo={undo}/>}
  </section>;
}
