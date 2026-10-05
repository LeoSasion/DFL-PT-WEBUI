import { useEffect, useRef, useState } from "react";
import { IconPhoto } from "@tabler/icons-react";
import { useI18n } from "../i18n.jsx";
import "./image-result-comparison.css";

function ComparisonPanel({ image, label, zoom, pan, onPan }) {
  const { t } = useI18n();
  const drag = useRef(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [image?.url]);
  const move = event => {
    if (!drag.current) return;
    const rectangle = event.currentTarget.getBoundingClientRect();
    onPan({ x: drag.current.pan.x + (event.clientX - drag.current.x) / rectangle.width * 100,
      y: drag.current.pan.y + (event.clientY - drag.current.y) / rectangle.height * 100 });
  };
  return <figure className="image-comparison-panel">
    <figcaption><strong>{t(label)}</strong><span title={image?.name}>{image?.name ?? t("无本机图片")}</span></figcaption>
    <div className="image-comparison-viewport" tabIndex={image?.url && !failed ? 0 : undefined}
      aria-label={t("{label}预览，方向键同步平移", { label: t(label) })}
      onPointerDown={event => { if (zoom <= 1 || !image?.url || failed || event.button !== 0) return; drag.current = { x: event.clientX, y: event.clientY, pan }; event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerMove={move} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
      onKeyDown={event => { const directions = { ArrowLeft: [5, 0], ArrowRight: [-5, 0], ArrowUp: [0, 5], ArrowDown: [0, -5] }; const delta = directions[event.key]; if (delta && zoom > 1) { event.preventDefault(); onPan({ x: pan.x + delta[0], y: pan.y + delta[1] }); } }}>
      {image?.url && !failed ? <img src={image.url} alt={image.name} draggable={false} decoding="async" onError={() => setFailed(true)} style={{ transform: `translate(${pan.x}%, ${pan.y}%) scale(${zoom})` }} />
        : <div className="image-tools-empty"><IconPhoto size={28} /><strong>{t(failed ? "本机图片无法读取" : label === "原图" ? "此任务没有本机原图" : "结果尚未保存到本机")}</strong><p>{t(failed ? "原文件保留，请检查记录或继续查询原任务。" : "外部参考链接不会自动加载。")}</p></div>}
    </div>
  </figure>;
}

export function ImageResultComparison({ original, result }) {
  const { t } = useI18n();
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const limit = 50 * (zoom - 1);
  const boundedPan = next => setPan({ x: Math.max(-limit, Math.min(limit, next.x)), y: Math.max(-limit, Math.min(limit, next.y)) });
  return <section className="image-result-comparison" aria-label={t("原图与结果并排比较")}>
    <div className="image-comparison-toolbar"><label><span>{t("同步缩放")}</span><input type="range" min="1" max="4" step="0.25" value={zoom} aria-label={t("同步缩放")} onChange={event => { setZoom(Number(event.target.value)); setPan({ x: 0, y: 0 }); }} /><output>{Math.round(zoom * 100)}%</output></label><button className="button secondary" type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>{t("适应窗口")}</button><small>{t("放大后拖动任一图片，两侧同步平移。")}</small></div>
    <div className="image-comparison-pair"><ComparisonPanel image={original} label="原图" zoom={zoom} pan={pan} onPan={boundedPan} /><ComparisonPanel image={result} label="生成结果" zoom={zoom} pan={pan} onPan={boundedPan} /></div>
  </section>;
}
