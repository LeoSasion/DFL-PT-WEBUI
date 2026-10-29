import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";

/** One modal owner for upgrades, feedback and runtime cleanup decisions. */
export function Dialog({ title, description, children, actions, busy = false, onClose, returnFocus, className = "" }) {
  const titleId = useId();
  const descriptionId = useId();
  const surface = useRef(null);
  const backdrop = useRef(null);
  const latest = useRef({ busy, onClose });
  latest.current = { busy, onClose };

  useEffect(() => {
    const previous = returnFocus?.isConnected ? returnFocus : document.activeElement;
    const background = [...document.body.children].filter((element) => element !== backdrop.current);
    const saved = background.map((element) => [element, element.inert, element.getAttribute("aria-hidden")]);
    for (const [element] of saved) { element.inert = true; element.setAttribute("aria-hidden", "true"); }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const controls = () => [...(surface.current?.querySelectorAll(
      'button:not(:disabled), textarea:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), [tabindex="0"]',
    ) ?? [])].filter((element) => element.getClientRects().length > 0);
    (surface.current?.querySelector("[data-autofocus]") ?? controls()[0] ?? surface.current)?.focus();
    const keyboard = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (!latest.current.busy) latest.current.onClose?.();
        return;
      }
      if (event.key !== "Tab") return;
      const items = controls();
      if (!items.length) { event.preventDefault(); surface.current?.focus(); return; }
      const first = items[0]; const last = items[items.length - 1];
      if (event.shiftKey && (document.activeElement === first || !surface.current.contains(document.activeElement))) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !surface.current.contains(document.activeElement))) {
        event.preventDefault(); first.focus();
      }
    };
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("keydown", keyboard);
      for (const [element, inert, hidden] of saved) {
        element.inert = inert;
        if (hidden === null) element.removeAttribute("aria-hidden"); else element.setAttribute("aria-hidden", hidden);
      }
      document.body.style.overflow = previousOverflow;
      if (previous?.isConnected && previous !== document.body && !previous.closest("[inert]")) previous.focus?.();
      else document.querySelector("[data-dialog-return-focus]")?.focus();
    };
  }, []);

  return createPortal(
    <div ref={backdrop} className="launcher-modal-backdrop">
      <section ref={surface} tabIndex={-1} className={"launcher-modal " + className}
        role="dialog" aria-modal="true" aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined} aria-busy={busy || undefined}>
        <header className="launcher-modal__header"><h2 id={titleId}>{title}</h2></header>
        <div className="launcher-modal__body">
          {description ? <p id={descriptionId}>{description}</p> : null}
          {children}
        </div>
        <footer className="launcher-modal__actions">{actions}</footer>
      </section>
    </div>, document.body,
  );
}
