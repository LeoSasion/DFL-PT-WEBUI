import { Component, Suspense, useEffect, useId, useRef } from "react";
import { IconAlertTriangle, IconCheck, IconPlayerStop, IconX } from "@tabler/icons-react";
import { useI18n } from "../i18n.jsx";
import { LoadingProgress } from "./ProgressFeedback.jsx";

export function useDialogFocus(open, onClose) {
  const dialogRef = useRef(null);
  const initialFocusRef = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open) return undefined;

    const previousFocus = document.activeElement;
    const dialog = dialogRef.current;
    initialFocusRef.current?.focus();

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }

      if (event.key !== "Tab" || !dialog) return;

      const focusable = [...dialog.querySelectorAll(
        "button:not(:disabled), select:not(:disabled), input:not(:disabled), textarea:not(:disabled), a[href], summary, [tabindex]:not([tabindex='-1'])",
      )].filter(element => element.getClientRects().length && !element.closest('[hidden], [inert]'));
      if (!focusable.length) { event.preventDefault(); return; }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!dialog.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [open]);

  return { dialogRef, initialFocusRef };
}

export function Toast({ message, tone = "success", onDismiss }) {
  const { t } = useI18n();
  if (!message) return null;
  return (
    <div className={`toast is-${tone}`} role="status">
      {tone === "warning" ? <IconAlertTriangle size={18} stroke={2} /> : <IconCheck size={18} stroke={2.2} />}
      <span>{message}</span>
      <button type="button" className="icon-button quiet" aria-label={t("关闭提示")} onClick={onDismiss}>
        <IconX size={16} />
      </button>
    </div>
  );
}

export function StopConfirmDialog({ open, onCancel, onConfirm }) {
  const { t } = useI18n();
  const { dialogRef, initialFocusRef } = useDialogFocus(open, onCancel);
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className="modal-card stop-dialog"
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="stop-title"
        aria-describedby="stop-description"
      >
        <span className="stop-icon"><IconPlayerStop size={22} stroke={2} /></span>
        <h2 id="stop-title">{t("安全停止训练？")}</h2>
        <p id="stop-description">{t("训练已开始时会先请求保存；若仍在模型名称等启动问答阶段，则直接结束。12 秒未响应时会自动终止，避免永久停留。")}</p>
        <footer>
          <button ref={initialFocusRef} className="button secondary" type="button" onClick={onCancel}>{t("继续训练")}</button>
          <button className="button danger" type="button" onClick={onConfirm}>{t("确认安全停止")}</button>
        </footer>
      </section>
    </div>
  );
}

function DialogLoading({ label, onClose, failed = false }) {
  const { t } = useI18n();
  const titleId = useId();
  const { dialogRef, initialFocusRef } = useDialogFocus(true, onClose);
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal-card stop-dialog" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <h2 id={titleId}>{label}</h2>
        {failed ? (
          <p role="alert">{t("窗口内容加载失败，请刷新页面后重试。后台任务会继续运行。")}</p>
        ) : (
          <LoadingProgress inline compact label={t("正在加载窗口内容…")} detail={t("可以关闭窗口，后台任务继续运行。")} operationKey={`dialog:${label}`} />
        )}
        <footer><button ref={initialFocusRef} className="button secondary" type="button" onClick={onClose}>{t("取消")}</button></footer>
      </section>
    </div>
  );
}

export class DeferredDialog extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    const { children, label, onClose } = this.props;
    if (this.state.failed) return <DialogLoading label={label} onClose={onClose} failed />;
    return <Suspense fallback={<DialogLoading label={label} onClose={onClose} />}>{children}</Suspense>;
  }
}
