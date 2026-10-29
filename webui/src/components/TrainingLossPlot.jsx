import {
  CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useI18n } from "../i18n.jsx";

function formatIteration(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return String(value);
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip">
    <strong>{formatIteration(label)} iters</strong>
    <span>SRC {payload[0]?.value}</span>
    <span>DST {payload[1]?.value}</span>
  </div>;
}

// The chart library is needed only after real training history is available.
// Keep its imports out of the workstation shell and the empty-history state.
export default function TrainingLossPlot({ iteration, lossHistory }) {
  const { language } = useI18n();
  return <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height: 127 }}>
    <LineChart data={lossHistory} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
      <CartesianGrid stroke="#163126" strokeOpacity={0.62} vertical={false} />
      <XAxis axisLine={false} dataKey="iteration" minTickGap={34}
        tick={{ fill: "#738079", fontSize: 11 }} tickFormatter={formatIteration} tickLine={false} />
      <YAxis axisLine={false} domain={[0, "auto"]} tick={{ fill: "#738079", fontSize: 11 }}
        tickFormatter={value => Number(value.toPrecision(3))} tickLine={false} width={44} />
      <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#2ce39f", strokeOpacity: 0.35 }} />
      {iteration > 0 ? <ReferenceLine x={iteration} stroke="#32e4a6" strokeDasharray="3 4" label={{
        value: iteration.toLocaleString(language === "zh" ? "zh-CN" : "en-US"),
        fill: "#b9f7dc", fontSize: 11, position: "insideTop", dy: 4,
      }} /> : null}
      <Line type="monotone" dataKey="gLoss" stroke="#2ce39f" strokeWidth={1.55} dot={false} isAnimationActive={false} />
      <Line type="monotone" dataKey="dLoss" stroke="#f3b83f" strokeWidth={1.45} dot={false} isAnimationActive={false} />
    </LineChart>
  </ResponsiveContainer>;
}
