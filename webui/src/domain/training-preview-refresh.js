export const DEFAULT_PREVIEW_REFRESH_SECONDS = 180;
export const MIN_PREVIEW_REFRESH_SECONDS = 5;
export const MAX_PREVIEW_REFRESH_SECONDS = 3600;
export const PREVIEW_REFRESH_STORAGE_KEY = "dfl-webui-training-preview-refresh-seconds-v1";

export function parsePreviewRefreshSeconds(value) {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const text = String(value).trim();
  if (!/^\d+$/.test(text)) return null;
  const seconds = Number(text);
  return Number.isSafeInteger(seconds)
    && seconds >= MIN_PREVIEW_REFRESH_SECONDS
    && seconds <= MAX_PREVIEW_REFRESH_SECONDS ? seconds : null;
}

export function readPreviewRefreshSeconds(storage) {
  try {
    return parsePreviewRefreshSeconds((storage ?? globalThis.localStorage)?.getItem(PREVIEW_REFRESH_STORAGE_KEY))
      ?? DEFAULT_PREVIEW_REFRESH_SECONDS;
  } catch {
    return DEFAULT_PREVIEW_REFRESH_SECONDS;
  }
}

export function planTrainingPreviewDisplay(current, {
  jobId,
  latestVersion,
  manualRefresh,
  intervalSeconds,
  active,
}, now) {
  if (!jobId) return { displayed: null, delayMs: null };
  const version = latestVersion ?? null;
  if (current?.jobId !== jobId) {
    return {
      displayed: { jobId, version, shownAt: version === null ? 0 : now, manualRefresh, awaitManualVersion: false },
      delayMs: null,
    };
  }
  if (current.manualRefresh !== manualRefresh) {
    const nextVersion = version ?? current.version;
    return {
      displayed: {
        jobId,
        version: nextVersion,
        shownAt: now,
        manualRefresh,
        awaitManualVersion: nextVersion === current.version,
      },
      delayMs: null,
    };
  }
  if (version === null || version === current.version) return { displayed: current, delayMs: null };
  const dueAt = current.shownAt + intervalSeconds * 1000;
  if (current.version === null || !active || current.awaitManualVersion || now >= dueAt) {
    return {
      displayed: { ...current, version, shownAt: now, awaitManualVersion: false },
      delayMs: null,
    };
  }
  return { displayed: current, delayMs: dueAt - now };
}
