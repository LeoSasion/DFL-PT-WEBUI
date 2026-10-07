import { useEffect, useRef } from "react";
import { IconChevronLeft, IconChevronRight, IconX } from "@tabler/icons-react";
import { useDialogFocus } from "./Overlays.jsx";
import { bestFacesetPreviewKey, bestFacesetReasonText } from "../domain/best-faceset-workflow.js";

export function BestFacesetPreview({ item, current, total, category, t, onClose, onNavigate, navigating,
  reference, referenceDisabled, onReference, reviewActive, canKeep, decision, onDecision, canUndo, onUndo, busy }) {
  const { dialogRef, initialFocusRef } = useDialogFocus(true, onClose);
  const latest = useRef(null);
  latest.current = { current, total, navigating, reviewActive, canKeep, canUndo, busy, onNavigate, onDecision, onUndo };
  useEffect(() => {
    const listener = event => {
      const action = bestFacesetPreviewKey(event);
      if (!action) return;
      const state = latest.current;
      if (state.busy || state.navigating) return;
      if (action === "previous" && state.current > 1 || action === "next" && state.current < state.total) {
        event.preventDefault(); void state.onNavigate(action === "previous" ? -1 : 1);
      } else if (action === "undo" && state.reviewActive && state.canUndo) {
        event.preventDefault(); void state.onUndo();
      } else if (["keep", "exclude", "defer"].includes(action) && state.reviewActive && (action !== "keep" || state.canKeep)) {
        event.preventDefault(); void state.onDecision(action);
      }
    };
    document.addEventListener("keydown", listener);
    return () => document.removeEventListener("keydown", listener);
  }, []);
  return <div className="best-faceset-modal"><div role="dialog" aria-modal="true" aria-label={t("复核人脸：{name}", { name: item.name })} ref={dialogRef}>
    <header className="best-faceset-preview-header"><span aria-live="polite">{t("第 {current} 张 / 共 {total} 张", { current, total })} · {category}</span>
      <button className="button secondary" type="button" ref={initialFocusRef} onClick={onClose} aria-label={t("关闭人脸复核")}><IconX size={16}/>{t("关闭")}</button></header>
    <div className="best-faceset-preview-navigation"><button className="button secondary" type="button" disabled={busy || navigating || current <= 1} onClick={() => void onNavigate(-1)} aria-label={t("查看上一张人脸")}><IconChevronLeft size={16}/>{t("上一张")}</button>
      <span>{navigating ? t("读取下一张…") : t("左右键切换 · Esc 关闭")}</span>
      <button className="button secondary" type="button" disabled={busy || navigating || current >= total} onClick={() => void onNavigate(1)} aria-label={t("查看下一张人脸")}>{t("下一张")}<IconChevronRight size={16}/></button></div>
    <img src={item.imageUrl} alt={item.name}/><strong className="best-faceset-preview-name">{item.name}</strong>
    <p>{item.reasons?.map(reason => bestFacesetReasonText(reason, t)).join(" / ") || t("请确认人物与图片内容")}</p>
    {item.algorithmSuggestion && <p>{t("算法建议：{status}", { status: t({ selected: "推荐训练", review: "待复核或备选", rejected: "排除" }[item.algorithmSuggestion.status] ?? item.algorithmSuggestion.status) })}
      {item.manualDecision && item.algorithmSuggestion.reasons?.length ? <><br/>{item.algorithmSuggestion.reasons.map(reason => bestFacesetReasonText(reason, t)).join(" / ")}</> : null}</p>}
    {reviewActive && <div className="best-faceset-preview-decisions"><strong>{t("人工决定：{decision}", { decision })}</strong><div className="best-faceset-controls">
      <button className="button secondary" type="button" disabled={busy || !canKeep} onClick={() => void onDecision("keep")}>{t("保留 · K")}</button>
      <button className="button secondary" type="button" disabled={busy} onClick={() => void onDecision("exclude")}>{t("排除 · X")}</button>
      <button className="button secondary" type="button" disabled={busy} onClick={() => void onDecision("defer")}>{t("暂缓 · S")}</button>
      <button className="button secondary" type="button" disabled={busy || !canUndo} onClick={() => void onUndo()}>{t("撤销上次 · Ctrl+Z")}</button></div>
      {!canKeep && <p>{item.reviewEligibility?.blockedReasons?.map(reason => t(reason)).join(" / ") || t("这张素材未通过训练输入检查，无法人工保留。")}</p>}
      {item.reviewEligibility?.warnings?.length ? <p>{t("保留前请确认：{warnings}", { warnings: item.reviewEligibility.warnings.map(reason => t(reason)).join(" / ") })}</p> : null}</div>}
    <label className="best-faceset-preview-reference"><input type="checkbox" checked={reference} disabled={referenceDisabled} onChange={onReference} aria-label={t("将 {name} 设为同人参照", { name: item.name })}/>{t("同人参照（用于确认身份）")}</label>
    <small>{t("身份参照与保留决定分别记录。")}</small>
  </div></div>;
}
