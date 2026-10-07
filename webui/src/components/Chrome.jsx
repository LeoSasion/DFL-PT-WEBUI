import { useEffect, useState } from "react";
import {
  IconBoxModel2,
  IconChartDots3,
  IconCheck,
  IconChevronRight,
  IconFileExport,
  IconFolder,
  IconLayoutDashboard,
  IconMasksTheater,
  IconMenu2,
  IconMinus,
  IconPencil,
  IconPlus,
  IconSettings,
  IconSquare,
  IconStack2,
  IconTool,
  IconUsers,
  IconX,
} from "@tabler/icons-react";
import { getNavigationWorkflowGroup, getWorkflowNavigation, workflowGroups } from "../data/dashboard.js";
import { useI18n } from "../i18n.jsx";

const DESKTOP_BREAKPOINT = 1020;
const DESKTOP_DESIGN_WIDTH = 1440;
const DESKTOP_DESIGN_HEIGHT = 900;
const MINIMUM_DESKTOP_SCALE = 1;
const MAXIMUM_DESKTOP_SCALE = 1.15;

function readDesktopUiScale() {
  if (typeof window === "undefined" || window.innerWidth <= DESKTOP_BREAKPOINT) return 1;
  return Math.max(
    MINIMUM_DESKTOP_SCALE,
    Math.min(
      MAXIMUM_DESKTOP_SCALE,
      window.innerWidth / DESKTOP_DESIGN_WIDTH,
      window.innerHeight / DESKTOP_DESIGN_HEIGHT,
    ),
  );
}

function useDesktopUiScale() {
  const [scale, setScale] = useState(readDesktopUiScale);

  useEffect(() => {
    const updateScale = () => {
      const nextScale = readDesktopUiScale();
      setScale((currentScale) => (
        Math.abs(currentScale - nextScale) < 0.001 ? currentScale : nextScale
      ));
    };

    window.addEventListener("resize", updateScale, { passive: true });
    return () => window.removeEventListener("resize", updateScale);
  }, []);

  return scale;
}

const navItems = [
  { id: "overview", label: "训练总览", icon: IconLayoutDashboard },
  { id: "video", label: "素材管理", icon: IconFolder },
  { id: "src", label: "SRC 数据", icon: IconUsers },
  { id: "dst", label: "DST 数据", icon: IconUsers },
  { id: "xseg", label: "XSeg 遮罩", icon: IconMasksTheater, tone: "violet" },
  { id: "training", label: "模型训练", icon: IconBoxModel2 },
  { id: "diagnostics", label: "质量诊断", icon: IconChartDots3 },
  { id: "merge", label: "模型应用", icon: IconStack2 },
  { id: "export", label: "视频导出", icon: IconFileExport },
  { id: "tools", label: "工具", icon: IconTool },
];

export function BrandBar() {
  const { t } = useI18n();
  return (
    <div className="brand-bar">
      <div className="brand-lockup">
        <img className="brand-mark" src="/assets/brand-mark.png" alt="" />
        <strong>DFL-PT-WEBUI</strong>
      </div>
      <div className="window-actions" aria-label={t("窗口控制")}>
        <button type="button" aria-label={t("最小化（桌面壳接入后可用）")} disabled><IconMinus size={15} stroke={1.6} /></button>
        <button type="button" aria-label={t("最大化（桌面壳接入后可用）")} disabled><IconSquare size={12} stroke={1.6} /></button>
        <button className="window-close" type="button" aria-label={t("关闭（桌面壳接入后可用）")} disabled><IconX size={15} stroke={1.6} /></button>
      </div>
    </div>
  );
}

export function Sidebar({ activeNav, onNavigate }) {
  const { t } = useI18n();
  const currentGroup = getNavigationWorkflowGroup(activeNav);
  const renderItem = ({ id, label, icon: Icon, tone }) => (
    <button className={`nav-item ${activeNav === id ? "is-active" : ""} ${tone === "violet" ? "is-violet" : ""}`}
      key={id} type="button" aria-label={t(label)} data-label={t(label)} title={t(label)}
      aria-current={activeNav === id ? "page" : undefined} onClick={() => onNavigate(id, t(label))}>
      <Icon size={20} stroke={1.8} /><span>{t(label)}</span>
    </button>
  );
  return (
    <aside className={`sidebar ${currentGroup ? "is-workflow" : "is-utility"}`} aria-label={t("主导航")}>
      <nav className="sidebar-nav">
        {workflowGroups.map(group => (
          <section className={`sidebar-group ${currentGroup === group.id ? "is-current-group" : ""}`} key={group.id} aria-label={t(group.label)}>
            <button className={`sidebar-group-title ${currentGroup === group.id ? "is-current" : ""}`}
              type="button" aria-current={currentGroup === group.id ? "true" : undefined}
              onClick={() => onNavigate(group.nav, t(group.label))}>{t(group.label)}</button>
            <div className="sidebar-group-items">{navItems.filter(item => group.pages.includes(item.id)).map(renderItem)}</div>
          </section>
        ))}
      </nav>
      <div className="sidebar-utilities">
      {navItems.filter(item => item.id === "tools").map(renderItem)}
      <button
        className={`nav-item sidebar-settings ${activeNav === "settings" ? "is-active" : ""}`}
        type="button"
        aria-label={t("设置")}
        data-label={t("设置")}
        title={t("设置")}
        aria-current={activeNav === "settings" ? "page" : undefined}
        onClick={() => onNavigate("settings", t("设置"))}
      >
        <IconSettings size={20} stroke={1.8} />
        <span>{t("设置")}</span>
      </button>
      </div>
    </aside>
  );
}

export function ProjectHeader({ projectName, workspacePath, serviceState, telemetry, onNewTask, onMenu, onProjectMenu }) {
  const { language, setLanguage, t } = useI18n();
  const serviceOnline = serviceState === "online";
  const gpu = telemetry?.gpus?.[0];
  return (
    <header className="project-header">
      <div className="project-copy">
        <div className="project-title-row">
          <h1 title={`DFL-PT-WEBUI${projectName ? ` · ${projectName}` : ""}`}><button className="project-switch-button" type="button" aria-label={t("打开项目菜单")} aria-haspopup="dialog" onClick={onProjectMenu}>{projectName || "DFL-PT-WEBUI"}<IconChevronRight size={17}/></button></h1>
        </div>
        <div className="workspace-path">
          <span>{t("项目目录")}</span>
          <code>{workspacePath}</code>
          <IconFolder size={15} stroke={1.7} />
        </div>
      </div>
      <div className="project-actions">
        <div className="language-switch" role="group" aria-label={t("界面语言")}>
          <button
            className={language === "zh" ? "is-active" : ""}
            type="button"
            aria-pressed={language === "zh"}
            onClick={() => setLanguage("zh")}
          >
            中文
          </button>
          <button
            className={language === "en" ? "is-active" : ""}
            type="button"
            aria-pressed={language === "en"}
            onClick={() => setLanguage("en")}
          >
            EN
          </button>
        </div>
        <div className={`environment-badge ${serviceOnline ? "" : "is-offline"}`}>
          <IconCheck size={15} stroke={2.5} />
          <span>{serviceOnline ? t("本地服务在线") : serviceState === "loading" ? t("正在检测服务") : t("本地服务离线")}</span>
        </div>
        <div className="gpu-summary">
          <span>{gpu ? `GPU ${gpu.index}` : t("运行时")}</span>
          <strong title={gpu?.name}>
            {gpu
              ? `${gpu.name} · ${(gpu.memoryUsedMiB / 1024).toFixed(1)} / ${(gpu.memoryTotalMiB / 1024).toFixed(1)} GB`
              : "DFL current / legacy"}
          </strong>
        </div>
        <button className="button primary new-task-button" type="button" onClick={onNewTask} disabled={!serviceOnline}>
          <IconPlus size={18} stroke={2} />
          {t("新建任务")}
        </button>
        <button className="icon-button menu-button" type="button" aria-label={t("打开工作区")} onClick={onMenu}>
          <IconMenu2 size={21} stroke={1.8} />
        </button>
      </div>
    </header>
  );
}

export function WorkflowBar({ activeNav = "overview", selectedStage, stageStates = {}, onSelectStage }) {
  const { t } = useI18n();
  const navigation = getWorkflowNavigation(activeNav);
  if (navigation.kind === "groups") return (
    <nav className="workflow-bar workflow-groups" aria-label={t("项目流程")}>
      {navigation.stages.map((group, index) => (
        <div className="workflow-segment" key={group.id}>
          <button className={`workflow-step ${navigation.group === group.id ? "is-current" : ""}`} type="button"
            aria-current={navigation.group === group.id ? "step" : undefined}
            onClick={() => onSelectStage({ id: group.stage, label: t(group.label) })}>
            <span className="stage-number">{index + 1}</span>
            <span className="stage-copy"><strong>{t(group.label)}</strong><small>{t(group.description)}</small></span>
          </button>
          {index < navigation.stages.length - 1 ? <IconChevronRight className="workflow-arrow" size={17} stroke={1.3} aria-hidden="true" /> : null}
        </div>
      ))}
    </nav>
  );
  return (
    <nav className="workflow-bar" aria-label={t("项目流程")}>
      {navigation.stages.map((stage, index) => {
        const actualState = stageStates[stage.id] ?? stage.state;
        const selected = selectedStage === stage.id;
        const stateLabel = actualState === "done"
          ? t(["train","merge","encode"].includes(stage.id) ? "成果可用" : "完成")
          : actualState === "active"
            ? t("进行中")
            : actualState === "available" ? t("检测到已有产物")
            : actualState === "unconfirmed" ? t("本次保存未确认")
            : actualState === "failed"
              ? t("失败")
              : actualState === "review" ? t("待复核") : t(stage.id === "mask" ? "可选" : "未运行");
        return (
          <div className="workflow-segment" key={stage.id}>
            <button
              className={`workflow-step is-${actualState} ${selected ? "is-current" : ""}`}
              type="button"
              onClick={() => onSelectStage({ ...stage, label: t(stage.label) })}
              aria-current={selected ? "step" : undefined}
              title={`${index + 1}. ${t(stage.label)} · ${stateLabel}`}
            >
              <span className="stage-number">
                {actualState === "done" ? <IconCheck size={14} stroke={2.6} /> : index + 1}
              </span>
              <span className="stage-copy">
                <strong>{t(stage.label)}</strong>
                <small>
                  {stateLabel}
                </small>
              </span>
            </button>
            {index < navigation.stages.length - 1 ? (
              <IconChevronRight className="workflow-arrow" size={17} stroke={1.3} aria-hidden="true" />
            ) : null}
          </div>
        );
      })}
    </nav>
  );
}

export function AppShell({ children, activeNav, onNavigate }) {
  const uiScale = useDesktopUiScale();
  const isScaled = Math.abs(uiScale - 1) > 0.001;
  const scaleStyle = isScaled
    ? {
        "--desktop-ui-scale": uiScale,
        "--desktop-ui-inverse": 1 / uiScale,
      }
    : undefined;

  return (
    <div className={`app-frame ${isScaled ? "is-desktop-scaled" : ""}`} style={scaleStyle}>
      <BrandBar />
      <div className="app-shell">
        <Sidebar activeNav={activeNav} onNavigate={onNavigate} />
        {children}
      </div>
    </div>
  );
}
