import { useCallback, useEffect, useState } from "react";
import { IconCopy, IconExternalLink, IconRefresh, IconShieldCheck } from "@tabler/icons-react";
import { runtimeApi } from "../runtime/api.js";
import { useI18n } from "../i18n.jsx";
import "./release-feedback.css";

const installationLabel = source => ({ "official-source": "官方固定源码", source: "源码安装", portable: "便携整合包", "local-source": "本地源码（未记录安装）" })[source] ?? "待检测";
const portableLabel = status => ({ "previous-release-not-rebuilt": "旧版便携包，尚未重打", built: "已构建", "not-rebuilt": "尚未重打" })[status] ?? status ?? "本机未验证便携包";

export function useReleaseInfo(serviceState) {
  const [release, setRelease] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const refresh = useCallback(async () => {
    setLoading(true);
    try { setRelease(await runtimeApi.release()); setError(null); }
    catch (failure) { setError(failure); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { if (serviceState === "online") void refresh(); }, [refresh, serviceState]);
  return { release, error, loading, refresh };
}

export function ReleaseStrip({ release, onOpen }) {
  const { t } = useI18n();
  return <div className="release-strip">
    <span>DFL-PT-WEBUI {release?.appVersion ?? t("版本待检测")}</span>
    <span>{t("启动器")} {release?.launcherVersion ?? t("未记录")}</span>
    <span>{t("来源")} {t(installationLabel(release?.installationSource))}</span>
    <button type="button" onClick={onOpen}>{t("版本与反馈")}</button>
  </div>;
}

const failureSteps = [
  ["start", "打开 WebUI"], ["install", "安装"], ["repair", "修复依赖"], ["upgrade", "升级 / 回退"],
  ["project", "创建 / 切换项目"], ["import", "导入素材"], ["extract", "提帧 / 提脸"],
  ["train", "训练 / 保存 / 恢复"], ["export", "合成 / 编码"], ["image", "图像编辑"], ["unknown", "其他步骤"],
];

export function ReleaseFeedbackPanel({ releaseState, onNotice, onError }) {
  const { t } = useI18n();
  const { release, loading, error, refresh } = releaseState;
  const [failureStep, setFailureStep] = useState("start");
  const [feedback, setFeedback] = useState(null);
  const [busy, setBusy] = useState(false);
  const prepare = async () => {
    if (busy) return;
    setBusy(true);
    setFeedback(null);
    try { setFeedback(await runtimeApi.prepareFeedback({ failureStep })); }
    catch (failure) { onError(failure); }
    finally { setBusy(false); }
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(feedback.preview); onNotice(t("反馈信息已复制，请检查后粘贴到 Issue。")); }
    catch { onNotice(t("无法写入剪贴板，请选中预览内容手动复制。"), "warning"); }
  };
  const revision = release?.source?.revision;
  const sourceVerified = release?.source?.verified === true;
  const sourceUrl = /^[a-f0-9]{40}$/i.test(revision ?? "") ? `https://github.com/LeoSasion/DFL-PT-WEBUI/tree/${revision}` : null;
  const issueUrl = feedback?.issueUrl && (() => {
    try { const url = new URL(feedback.issueUrl); return url.origin === "https://github.com" && url.pathname.startsWith("/LeoSasion/DFL-PT-WEBUI/issues/") ? url.href : null; }
    catch { return null; }
  })();
  return <section className="release-feedback-panel" aria-label={t("版本、来源与反馈")}>
    <header><div><IconShieldCheck size={19} /><h3>{t("版本、来源与反馈")}</h3></div><button type="button" className="button secondary" disabled={loading} onClick={() => void refresh()}><IconRefresh size={14} />{t("重新检测")}</button></header>
    {error ? <p role="alert">{t("版本来源读取失败，可重新检测后再准备反馈信息。")}</p> : null}
    <dl className="release-facts">
      <div><dt>{t("应用版本")}</dt><dd>{release?.appVersion ?? "—"}</dd></div>
      <div><dt>{t("启动器版本")}</dt><dd>{release?.launcherVersion ?? t("未记录")}</dd></div>
      <div><dt>{t("安装来源")}</dt><dd>{t(installationLabel(release?.installationSource))}</dd></div>
      <div><dt>{t("源码校验")}</dt><dd className={sourceVerified ? "is-ready" : ""}>{sourceVerified ? t("当前内容与来源记录一致") : t("未验证或本地内容已变化")}</dd></div>
      <div className="release-full-row"><dt>{t("源码快照")}</dt><dd>{sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">{revision}</a> : revision ?? t("未记录")}</dd></div>
      <div className="release-full-row"><dt>{t("源码内容树")}</dt><dd><code>{release?.source?.tree ?? t("未记录")}</code></dd></div>
      <div className="release-full-row"><dt>{t("内容 SHA-256")}</dt><dd><code>{release?.source?.contentSha256 ?? t("未记录")}</code></dd></div>
      <div className="release-full-row"><dt>{t("便携包")}</dt><dd>{release?.portable?.version ?? t("未记录")} · {t(portableLabel(release?.portable?.status))}</dd></div>
    </dl>
    <p className="release-upgrade-note">{t("升级由你在启动器中主动发起。先停止运行中的任务，再备份并升级；失败时使用启动器回退。素材、模型与本地配置保留在原工作区。")}</p>
    <div className="feedback-prepare"><label><span>{t("失败步骤")}</span><select value={failureStep} disabled={busy} onChange={event => { setFailureStep(event.target.value); setFeedback(null); }}>{failureSteps.map(([value, label]) => <option key={value} value={value}>{t(label)}</option>)}</select></label><button type="button" className="button secondary" onClick={() => void prepare()} disabled={busy}>{t(busy ? "正在整理…" : "准备反馈信息")}</button></div>
    <p className="feedback-privacy">{t("仅在本机整理版本、来源、失败步骤和诊断摘要。请预览后复制；日志、素材、模型、API Key 与私有绝对路径不会自动上传。")}</p>
    {feedback ? <div className="feedback-preview"><label htmlFor="feedback-preview-text">{t("待提交内容预览")}</label><textarea id="feedback-preview-text" readOnly value={feedback.preview ?? ""} rows={12} spellCheck={false} /><div><button type="button" className="button secondary" onClick={() => void copy()}><IconCopy size={14} />{t("复制反馈信息")}</button>{issueUrl ? <a className="button primary" href={issueUrl} target="_blank" rel="noreferrer"><IconExternalLink size={14} />{t("打开 Issue 页面")}</a> : null}</div><small>{t("打开 Issue 只打开提交页面；请自行补充复现步骤并确认提交。")}</small></div> : null}
  </section>;
}
