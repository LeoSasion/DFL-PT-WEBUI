const CLASSES = new Set(["all", "selected", "review", "rejected"]);
const draftWriters = new Map();

export function registerBestFacesetDraftWriter(scope, writer) {
  draftWriters.set(scope, writer);
  if (draftWriters.size > 100) draftWriters.delete(draftWriters.keys().next().value);
}

export async function flushBestFacesetDraft(scope) {
  await draftWriters.get(scope)?.flush();
}

export function bestFacesetDraftState(record) {
  const draft = record?.draft;
  if (!draft) return null;
  return {
    planId: draft.planId ?? null,
    identityReferences: Array.isArray(draft.identityReferences) ? draft.identityReferences.slice(0, 32) : [],
    confirmed: Boolean(draft.confirmed && record.sourceValid && record.identityConfirmationValid),
    offset: Number.isInteger(draft.offset) && draft.offset >= 0 ? Math.floor(draft.offset / 60) * 60 : 0,
    category: CLASSES.has(draft.category) ? draft.category : "all",
    targetCount: Number.isInteger(draft.targetCount) && draft.targetCount >= 1 && draft.targetCount <= 500000 ? draft.targetCount : 2000,
    qualityModel: draft.qualityModel === "efficient-fiqa" ? "efficient-fiqa" : "foreground_tenengrad",
    sourceFingerprint: record.sourceFingerprint,
  };
}

// Only the latest unsaved state is needed, but writes must use the preceding
// server revision. A conflict pauses saving until the user reloads the draft.
export function createBestFacesetDraftWriter({ revision = 0, save, onSaved, onError }) {
  let currentRevision = revision, pending = null, inFlight = null, paused = false, lastSaved = null, lastResult = null;
  async function drain() {
    while (pending && !paused) {
      const draft = pending; pending = null;
      const serialized = JSON.stringify(draft);
      try {
        if (serialized === lastSaved) {
          // A normalized edit can be identical to the persisted draft. Finish
          // the UI's saving state without another write or a CAS increment.
          onSaved?.({ ...lastResult, unchanged: true });
          continue;
        }
        const result = await save({ expectedRevision: currentRevision, draft });
        currentRevision = result.revision;
        lastSaved = serialized;
        lastResult = structuredClone(result);
        onSaved?.(result);
      } catch (error) {
        paused = true;
        onError?.(error);
      }
    }
  }
  function begin() {
    if (inFlight || paused) return;
    inFlight = Promise.resolve().then(drain).finally(() => {
      inFlight = null;
      if (pending && !paused) begin();
    });
  }
  return {
    enqueue(draft) {
      if (paused) return;
      pending = structuredClone(draft);
      begin();
    },
    async flush() { await inFlight; },
    pause() { paused = true; pending = null; },
    get paused() { return paused; },
  };
}

export function bestFacesetHistoryLabel(item, { formatDate, translate = value => value }) {
  const date = item.updatedAt ?? item.createdAt ?? item.publication?.publishedAt;
  const time = date ? formatDate(date) : translate("较早记录");
  const counts = item.counts ?? item.selection?.counts ?? item.publication?.counts;
  const version = item.parentPlanId ? translate("人工修订") : translate("算法建议");
  const state = { analyzing: "待完成分析", finalized: "待发布", published: "已发布", withdrawn: "已撤回" }[item.state] ?? item.state;
  return `${time} · ${version} · ${item.total ?? 0} → ${counts?.selected ?? "—"} · ${translate(state)}`;
}

export function bestFacesetImageLabel(item) { return item?.member ?? item?.name ?? ""; }

export function bestFacesetPublicationPreviewMatches(plan, preview) {
  return Boolean(preview && preview.clientPlanId === plan?.planId && preview.clientReviewRevision === (plan?.reviewRevision ?? 0));
}

export function bestFacesetReasonText(reason, translate = value => value) {
  if (typeof reason !== "string") return reason;
  let match = reason.match(/^质量 ([\d.]+) 达保底 ([\d.]+)$/);
  if (match) return translate("质量 {score} 达保底 {floor}", { score: match[1], floor: match[2] });
  match = reason.match(/^质量 ([\d.]+) 达标的备选图$/);
  if (match) return translate("质量 {score} 达标的备选图", { score: match[1] });
  match = reason.match(/^头部偏转 ([+-]?[\d.]+)° \/ (闭嘴|微张嘴|张嘴) \/ (双眼张开|单眼闭合|双眼闭合)（几何估计）$/);
  if (match) return translate("头部偏转 {yaw}° / {mouth} / {eyes}（几何估计）", { yaw: match[1], mouth: translate(match[2]), eyes: translate(match[3]) });
  match = reason.match(/^非重复代表；同类候选 (\d+) 张(（稀缺姿态或眼口状态）)?$/);
  if (match) return translate(match[2] ? "非重复代表；同类候选 {count} 张（稀缺姿态或眼口状态）" : "非重复代表；同类候选 {count} 张", { count: match[1] });
  return translate(reason);
}

export function bestFacesetPreviewKey(event) {
  if (event.target?.closest?.("input, textarea, select, [contenteditable='true']")) return null;
  if (event.altKey) return null;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.shiftKey) return "undo";
  if (event.ctrlKey || event.metaKey) return null;
  return { ArrowLeft: "previous", ArrowRight: "next", k: "keep", K: "keep", x: "exclude", X: "exclude", s: "defer", S: "defer" }[event.key] ?? null;
}
