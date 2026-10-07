import { IconX } from "@tabler/icons-react";
import { useDialogFocus } from "./Overlays.jsx";
import { ProjectManagerPanel } from "./ProjectManagerPanel.jsx";
import { useI18n } from "../i18n.jsx";

export function ProjectPickerDialog({ open, health, activeJobCount, blockedReason, onClose, onSwitchProject, onNotice }) {
  const { t } = useI18n();
  const { dialogRef, initialFocusRef } = useDialogFocus(open, onClose);
  if (!open) return null;
  return <div className="modal-backdrop project-picker-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <section ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="project-picker-heading" className="project-picker-dialog" tabIndex={-1}>
      <header><div><h2 id="project-picker-heading">{t("项目与工作区")}</h2><p>{t("选择独立保存素材和模型的项目，或建立新项目。")}</p></div>
        <button ref={initialFocusRef} type="button" className="icon-button" aria-label={t("关闭项目菜单")} onClick={onClose}><IconX size={20}/></button></header>
      <ProjectManagerPanel health={health} activeJobCount={activeJobCount} blockedReason={blockedReason} onNotice={onNotice}
        onSwitchProject={project => { onClose(); onSwitchProject(project); }}/>
    </section>
  </div>;
}
