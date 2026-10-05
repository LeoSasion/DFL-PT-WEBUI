import { useEffect, useMemo, useRef, useState } from "react";
import { IconArchive, IconCheck, IconChevronLeft, IconChevronRight, IconRefresh } from "@tabler/icons-react";
import { useI18n } from "../i18n.jsx";
import { runtimeApi } from "../runtime/api.js";
import { clearSimilarityReview, pendingSimilarityReview, rememberSimilarityReview, reviewedSimilarityNames, similarityReceiptMatches, similarityResultMatches, similarityReviewFor, similarityScopeKey } from "../domain/similarity-review.js";
import { LoadingProgress } from "./ProgressFeedback.jsx";
import { useDialogFocus } from "./Overlays.jsx";

function QuarantineConfirmation({ action, busy, onClose, onConfirm }) {
  const { t } = useI18n();
  const { dialogRef, initialFocusRef } = useDialogFocus(Boolean(action), () => { if (!busy) onClose(); });
  if (!action) return null;
  return <div className="modal-backdrop"><section className="modal-card role-confirmation similarity-confirmation" role="dialog" aria-modal="true" aria-labelledby="similarity-confirm-title" ref={dialogRef}>
    <header><h2 id="similarity-confirm-title">{t("确认隔离相似候选")}</h2></header>
    <div className="role-confirm-body">
      <strong>{action.side.toUpperCase()} · {t("{count} 张待处理", { count: action.names.length })}</strong>
      <p>{t("仅移动本次复核选中的非代表图；每组代表图保留，全部共用一个恢复令牌。")}</p>
      <p>{t("素材有变化时会停止操作，请重新分析。")}</p>
      <details><summary>{t("查看所选文件")}</summary><ul>{action.names.map(name => <li key={name}>{name}</li>)}</ul></details>
    </div>
    <footer>
      <button ref={initialFocusRef} type="button" className="button secondary" disabled={busy} onClick={onClose}>{t("返回复核")}</button>
      <button type="button" className="button primary" disabled={busy} onClick={onConfirm}>{busy ? t("正在隔离…") : t("确认隔离")}</button>
    </footer>
  </section></div>;
}

export function SimilarityAuditPanel({ side, workspaceKey, refreshVersion, onError, onNotice, onNavigateDataset, onBusyChange }) {
  const { t } = useI18n();
  const [threshold, setThreshold] = useState(0.86);
  const [mode, setMode] = useState("batch");
  const [offset, setOffset] = useState(0);
  const [compareOffset, setCompareOffset] = useState(250);
  const [analysis, setAnalysis] = useState(null);
  const [failure, setFailure] = useState(null);
  const [totals, setTotals] = useState({});
  const [selected, setSelected] = useState([]);
  const [confirmation, setConfirmation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uncertain, setUncertain] = useState(() => pendingSimilarityReview(workspaceKey, side));
  const [checking, setChecking] = useState(false);
  const [retry, setRetry] = useState(0);
  const inFlight = useRef(false);
  const scope = useMemo(() => ({ threshold, offset, compareOffset: mode === "paired" ? compareOffset : null, limit: 500 }), [threshold, offset, compareOffset, mode]);
  const queryKey = JSON.stringify([workspaceKey, similarityScopeKey(side, scope), refreshVersion, retry]);
  const liveQuery = useRef(queryKey);
  liveQuery.current = queryKey;
  const data = analysis?.key === queryKey ? analysis.value : null;
  const review = similarityReviewFor(data, side, scope, workspaceKey);
  const error = failure?.key === queryKey ? failure.error : null;
  const windowSize = mode === "paired" ? 250 : 500;
  const totalKey = JSON.stringify([workspaceKey, side]);
  const liveContext = useRef(totalKey);
  liveContext.current = totalKey;
  const total = data?.total ?? totals[totalKey] ?? null;
  const pageCount = total === null ? null : Math.max(1, Math.ceil(total / windowSize));
  const page = Math.floor(offset / windowSize) + 1;
  const comparisonPage = Math.floor(compareOffset / windowSize) + 1;
  const selectedNames = review ? reviewedSimilarityNames(data, selected) : [];
  const selectedSet = new Set(selectedNames);
  const activeConfirmation = confirmation?.key === queryKey ? confirmation : null;
  const unresolved = uncertain?.workspaceKey === workspaceKey && uncertain.side === side ? uncertain : null;

  useEffect(() => {
    onBusyChange?.(busy);
    return () => onBusyChange?.(false);
  }, [busy, onBusyChange]);

  useEffect(() => {
    setMode("batch");
    setOffset(0);
    setCompareOffset(250);
    setUncertain(pendingSimilarityReview(workspaceKey, side));
  }, [side, workspaceKey]);

  useEffect(() => {
    if (!data) return;
    if (mode === "paired" && data.total <= 250) {
      setMode("batch");
      setOffset(0);
      return;
    }
    const lastOffset = Math.max(0, (Math.ceil(data.total / windowSize) - 1) * windowSize);
    const nextOffset = Math.min(offset, lastOffset);
    if (offset !== nextOffset) setOffset(nextOffset);
    if (mode === "paired") {
      const nextCompare = Math.min(compareOffset, lastOffset);
      if (nextCompare === nextOffset) setCompareOffset(nextOffset ? nextOffset - windowSize : windowSize);
      else if (compareOffset !== nextCompare) setCompareOffset(nextCompare);
    }
  }, [data, mode, offset, compareOffset, windowSize]);

  useEffect(() => {
    setSelected([]);
    setConfirmation(null);
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void runtimeApi.alignedSimilarity(side, { ...scope, refresh: refreshVersion > 0 || retry > 0, signal: controller.signal }).then(value => {
        if (controller.signal.aborted || liveQuery.current !== queryKey) return;
        if (!similarityResultMatches(value, side, scope, workspaceKey)) throw new Error("相似分析响应不完整，请重新分析");
        setAnalysis({ key: queryKey, value });
        setFailure(null);
        setSelected([]);
        setTotals(current => ({ ...current, [totalKey]: value.total }));
      }).catch(nextError => {
        if (controller.signal.aborted || liveQuery.current !== queryKey) return;
        setFailure({ key: queryKey, error: nextError });
        onError(nextError);
      });
    }, 180);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [onError, queryKey, refreshVersion, retry, scope, side, totalKey, workspaceKey]);

  const changeMode = nextMode => {
    if (busy) return;
    setMode(nextMode);
    if (nextMode === "paired") {
      const nextOffset = Math.floor(offset / 250) * 250;
      setOffset(nextOffset);
      setCompareOffset(total !== null && nextOffset + 250 >= total && nextOffset > 0 ? nextOffset - 250 : nextOffset + 250);
    } else setOffset(Math.floor(offset / 500) * 500);
  };
  const changePage = (value, comparison = false) => {
    const nextPage = Number(value);
    if (busy || !Number.isSafeInteger(nextPage) || nextPage < 1 || (pageCount !== null && nextPage > pageCount)) return;
    const nextOffset = (nextPage - 1) * windowSize;
    if (comparison) {
      if (nextOffset !== offset) setCompareOffset(nextOffset);
    } else {
      if (mode === "paired" && nextOffset === compareOffset) setCompareOffset(offset);
      setOffset(nextOffset);
    }
  };
  const toggle = name => {
    if (!review || busy || !reviewedSimilarityNames(data, [name]).length) return;
    setSelected(current => current.includes(name) ? current.filter(value => value !== name) : [...current, name]);
  };
  const selectGroupDuplicates = group => {
    if (!review || busy) return;
    setSelected(current => reviewedSimilarityNames(data, [...current, ...group.members.map(member => member.name)]));
  };
  const confirmQuarantine = async () => {
    const action = activeConfirmation;
    if (!action || inFlight.current || unresolved || liveQuery.current !== action.key) return;
    try { rememberSimilarityReview(action); }
    catch (nextError) { onError(nextError); return; }
    inFlight.current = true;
    setBusy(true);
    try {
      const result = await runtimeApi.quarantineAlignedBatch(action.side, action.names, action.review);
      if (!similarityReceiptMatches(result, action)) throw new Error("隔离响应不完整，结果尚未确认，请先检查");
      clearSimilarityReview(action);
      onNotice(t("已隔离 {count} 张相似候选，可在数据集页恢复。", { count: result.count }));
    } catch (nextError) {
      if (!nextError.status || nextError.status >= 500) {
        if (liveContext.current === JSON.stringify([action.workspaceKey, action.side])) setUncertain(action);
      }
      else clearSimilarityReview(action);
      onError(nextError);
    } finally {
      if (liveQuery.current === action.key) {
        setAnalysis(null);
        setSelected([]);
        setConfirmation(null);
        setRetry(value => value + 1);
      }
      inFlight.current = false;
      setBusy(false);
    }
  };
  const checkQuarantine = async () => {
    const action = unresolved;
    if (!action || checking || busy) return;
    setChecking(true);
    try {
      const snapshot = await runtimeApi.similarityReviewState(action.side);
      if (!snapshot || typeof snapshot.side !== "string" || typeof snapshot.workspaceKey !== "string"
          || !Array.isArray(snapshot.names) || snapshot.names.some(name => typeof name !== "string") || !/^[a-f0-9]{64}$/.test(snapshot.fingerprint ?? "")) {
        throw new Error("隔离状态读取不完整，请继续检查");
      }
      if (snapshot.side !== action.side || snapshot.workspaceKey !== action.workspaceKey) {
        throw new Error("当前项目与待检查的隔离记录不同，请返回原项目检查");
      }
      const available = new Set(snapshot.names);
      const remaining = action.names.filter(name => available.has(name)).length;
      clearSimilarityReview(action);
      if (liveContext.current === JSON.stringify([action.workspaceKey, action.side])) {
        setUncertain(null);
        setRetry(value => value + 1);
      }
      onNotice(t("检查完成：原选中 {total} 张，仍在 aligned {remaining} 张；恢复区详情见数据集页。", { total: action.names.length, remaining }));
    } catch (nextError) { onError(nextError); }
    finally { setChecking(false); }
  };

  return <div className="similarity-workbench">
    {busy ? <LoadingProgress compact label={t("正在隔离相似候选…")} detail={t("批次共用一个可恢复令牌")} /> : null}
    <header className="similarity-toolbar">
      <div><strong>{t("视觉相似候选")}</strong><span>{t("仅生成候选，需要人工复核")}</span></div>
      <label><span>{t("相似阈值")}</span><input aria-label={t("相似阈值")} disabled={busy} type="range" min="0.72" max="0.98" step="0.01" value={threshold} onChange={event => setThreshold(Number(event.target.value))} /><strong>{threshold.toFixed(2)}</strong></label>
      <button className="button secondary" disabled={busy} type="button" onClick={() => setRetry(value => value + 1)}><IconRefresh size={15} />{t("重新分析")}</button>
    </header>
    <section className="similarity-batches" aria-label={t("相似分析范围")}>
      <label><span>{t("分析模式")}</span><select value={mode} disabled={busy} onChange={event => changeMode(event.target.value)}><option value="batch">{t("单批检查")}</option><option value="paired" disabled={total === null || total <= 250}>{t("两批比较")}</option></select></label>
      <div className="similarity-page-picker">
        <button className="icon-button quiet" disabled={busy || page <= 1} type="button" aria-label={t("上一批")} onClick={() => changePage(page - 1)}><IconChevronLeft size={16} /></button>
        <label><span>{mode === "paired" ? t("批次 A") : t("批次")}</span><input aria-label={mode === "paired" ? t("批次 A") : t("批次")} type="number" min="1" max={pageCount ?? undefined} value={page} disabled={busy} onChange={event => changePage(event.target.value)} /></label>
        <span>{pageCount === null ? "…" : `/ ${pageCount}`}</span>
        <button className="icon-button quiet" disabled={busy || pageCount === null || page >= pageCount} type="button" aria-label={t("下一批")} onClick={() => changePage(page + 1)}><IconChevronRight size={16} /></button>
      </div>
      {mode === "paired" ? <label><span>{t("批次 B")}</span><input aria-label={t("批次 B")} type="number" min="1" max={pageCount ?? undefined} value={comparisonPage} disabled={busy || (pageCount !== null && pageCount < 2)} onChange={event => changePage(event.target.value, true)} /><span>{pageCount === null ? "…" : `/ ${pageCount}`}</span></label> : null}
      <p>{mode === "paired" ? t("每批最多 250 张，仅比较选定两批；不代表全库查重。") : t("每批最多 500 张，可切换批次查看后续素材。")}</p>
      <small>{t("DCT、色彩与边缘描述子；相似分数不代表身份识别。")}</small>
    </section>
    <div className="similarity-summary">
      <div><span>{t("已分析")}</span><strong>{data?.analyzedCount ?? "—"}</strong></div>
      <div><span>{t("候选组")}</span><strong>{data?.groupCount ?? "—"}</strong></div>
      <div><span>{t("已选择隔离")}</span><strong>{selectedNames.length}</strong></div>
      <p>{data ? <>{data.windows.map(window => window.count ? `${mode === "paired" ? `${window.batch === 0 ? "A" : "B"}: ` : ""}${window.start}–${window.end}` : t("当前批次为空")).join(" · ")} / {data.total}{data.invalidCount ? ` · ${t("无法分析 {count} 张", { count: data.invalidCount })}` : ""}</> : t("正在读取选定范围…")}</p>
    </div>
    {error ? <div className="tool-workbench-state" role="alert"><strong>{t("相似分析未完成")}</strong><span>{t(error.message)}</span><button type="button" className="button secondary" onClick={() => setRetry(value => value + 1)}>{t("重新分析")}</button></div>
      : !data ? <div className="tool-workbench-state is-loading"><LoadingProgress inline className="in-panel" label={t("正在建立视觉相似候选组…")} detail={t("仅处理选定范围；切换批次后旧选择会清空。")} /></div>
        : data.groups.length ? <div className="similarity-groups">{data.groups.map(group => <section key={group.id}>
          <header><div><strong>{t("候选组 {id}", { id: group.id.replace("similar-", "") })}</strong><span>{t("{count} 张 · 最低 {score}", { count: group.memberCount, score: group.minimumScore.toFixed(3) })}</span></div><button type="button" disabled={!review || busy} onClick={() => selectGroupDuplicates(group)}><IconCheck size={14} />{t("选中非代表图")}</button></header>
          <div className="similarity-members">{group.members.map(member => <article className={`${member.representative ? "is-representative" : ""} ${selectedSet.has(member.name) ? "is-selected" : ""}`} key={member.name}>
            <button type="button" disabled={member.representative || !review || busy} onClick={() => toggle(member.name)} aria-label={t("选择候选 {name}", { name: member.name })} aria-pressed={selectedSet.has(member.name)}><img src={member.imageUrl} alt="" loading="lazy" decoding="async" /><span>{member.representative ? t("代表图") : selectedSet.has(member.name) ? t("待隔离") : t("候选")}{mode === "paired" ? ` · ${member.batch === 0 ? "A" : "B"}` : ""}</span></button>
            <div><strong title={member.name}>{member.name}</strong><small>{member.score.toFixed(3)}</small></div><button type="button" disabled={busy} onClick={() => onNavigateDataset(side, member)}>{t("查看")}</button>
          </article>)}</div>
        </section>)}</div>
          : <div className="tool-workbench-state" role="status"><IconCheck size={26} /><strong>{mode === "paired" ? t("选定两批中没有跨批相似组") : t("当前阈值下没有相似组")}</strong><span>{t("可适当降低阈值生成更宽松的候选，但仍需人工复核。")}</span></div>}
    <footer className={`similarity-action-dock${unresolved ? " is-uncertain" : ""}`}>
      {unresolved ? <div className="similarity-uncertain" role="status"><strong>{t("隔离结果尚未确认")}</strong><span>{t("先检查当前素材与恢复区，再继续隔离；不会自动重复提交。")}</span><button type="button" className="button secondary" disabled={busy || checking} onClick={() => void checkQuarantine()}>{checking ? t("正在检查隔离结果…") : t("检查隔离结果")}</button><button type="button" className="button secondary" disabled={busy || checking} onClick={() => onNavigateDataset(unresolved.side)}>{t("查看数据集与恢复区")}</button></div> : <><div><IconArchive size={18} /><span><strong>{t("{count} 张待处理", { count: selectedNames.length })}</strong><small>{t("每组代表图保留；可在数据集页恢复")}</small></span></div><button className="button primary" type="button" disabled={!selectedNames.length || !review || busy || checking} onClick={() => setConfirmation({ key: queryKey, workspaceKey, side, names: selectedNames, review })}>{t("移入可恢复隔离区")}</button></>}
    </footer>
    <QuarantineConfirmation action={activeConfirmation} busy={busy} onClose={() => setConfirmation(null)} onConfirm={() => void confirmQuarantine()} />
  </div>;
}
