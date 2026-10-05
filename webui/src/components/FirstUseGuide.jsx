import { IconArrowRight, IconCheck, IconInfoCircle } from "@tabler/icons-react";
import { useI18n } from "../i18n.jsx";
import "./first-use-guide.css";

export function FirstUseGuide({ readiness, workspace, project, nextStep, onNext, onProjects }) {
  const { t } = useI18n();
  if (workspace?.readiness?.me) return null;
  const managed = project?.managed ?? (project?.id && project.id !== "default");
  const statuses = [
    ["WebUI", readiness.webReady ? "本地服务已连接" : "等待本地服务", readiness.webReady],
    ["工具资源", readiness.toolsReady === false ? "缺少运行入口，请检查环境" : readiness.toolsReady ? readiness.toolsVerified ? "工具检查通过" : "入口已找到，运行能力需预检" : "等待环境检查", readiness.toolsReady && readiness.toolsVerified],
    ["GPU 训练", readiness.trainingReady === true ? "训练环境检查通过" : readiness.trainingReady === false ? "训练条件未就绪" : readiness.gpuDetected ? "已检测 GPU，PyTorch CUDA 待预检" : "尚未确认可用 GPU / CUDA", readiness.trainingReady === true],
  ];
  return <section className="first-use-guide" aria-label={t("首次使用与下一步")}>
    <div className="first-use-statuses">
      {statuses.map(([title, detail, ready]) => <div key={title} className={ready ? "is-ready" : ""}>
        {ready ? <IconCheck size={15} /> : <IconInfoCircle size={15} />}<span><strong>{t(title)}</strong><small>{t(detail)}</small></span>
      </div>)}
    </div>
    <div className="first-use-actions">
      <div><strong>{t("下一步：{step}", { step: t(nextStep.label) })}</strong><p>{managed ? t("当前项目独立保存素材与模型。导入只发生在你选择本机文件后。") : t("可先创建独立项目，也可使用默认工作区导入 SRC / DST。导入后按步骤提帧、提脸与选择人物。")}</p></div>
      <div><button type="button" className="button secondary" onClick={onProjects}>{t(managed ? "管理项目" : "创建 / 选择项目")}</button>
        <button type="button" className="button primary" onClick={onNext} disabled={!readiness.webReady || nextStep.pending}>{t(nextStep.label)}<IconArrowRight size={15} /></button></div>
    </div>
    <small className="first-use-note">{t("打开训练配置后仍需检查环境与参数，并由你明确启动训练。")}</small>
  </section>;
}
