export function similarityScopeKey(side, { threshold, offset = 0, compareOffset = null, limit = 500 }) {
  return JSON.stringify([side, threshold, offset, compareOffset, limit]);
}

export function similarityReviewFor(data, side, scope, workspaceKey) {
  if (!data || data.side !== side || !/^[a-f0-9]{64}$/.test(data.fingerprint ?? "")) return null;
  if (!workspaceKey || data.workspaceKey !== workspaceKey) return null;
  if (similarityScopeKey(data.side, data) !== similarityScopeKey(side, scope)) return null;
  return {
    side,
    workspaceKey,
    fingerprint: data.fingerprint,
    threshold: data.threshold,
    offset: data.offset,
    compareOffset: data.compareOffset,
    limit: data.limit,
  };
}

// Keep an unresolved write visible through in-app navigation. A reload still
// revalidates the server fingerprint before any new isolation.
const pendingReviews = new Map();
const pendingKey = (workspaceKey, side) => JSON.stringify([workspaceKey, side]);

export function pendingSimilarityReview(workspaceKey, side) {
  return pendingReviews.get(pendingKey(workspaceKey, side)) ?? null;
}

export function rememberSimilarityReview(action) {
  const key = pendingKey(action.workspaceKey, action.side);
  if (pendingReviews.has(key) && pendingReviews.get(key) !== action) {
    throw new Error("隔离结果尚未确认，请先检查");
  }
  if (!pendingReviews.has(key) && pendingReviews.size >= 32) {
    throw new Error("已有较多隔离结果待检查，请先复核");
  }
  pendingReviews.set(key, action);
}

export function similarityReceiptMatches(result, action) {
  if (!result || result.side !== action.side || result.count !== action.names.length
      || result.recoverable !== true || !/^\d{14}-[a-f0-9]{10}$/.test(result.token ?? "")
      || !Array.isArray(result.names) || result.names.length !== action.names.length) return false;
  const names = new Set(result.names);
  return names.size === action.names.length && action.names.every(name => names.has(name));
}

export function clearSimilarityReview(action) {
  const key = pendingKey(action.workspaceKey, action.side);
  if (pendingReviews.get(key) === action) pendingReviews.delete(key);
}

export function reviewedSimilarityNames(data, names) {
  const candidates = new Set();
  const representatives = new Set();
  for (const group of Array.isArray(data?.groups) ? data.groups : []) {
    if (!group || !Array.isArray(group.members)) continue;
    representatives.add(group.representativeName);
    for (const member of group.members) {
      if (member.representative) representatives.add(member.name);
      else candidates.add(member.name);
    }
  }
  return [...new Set(names)].filter(name => candidates.has(name) && !representatives.has(name));
}

export function similarityResultMatches(data, side, scope, workspaceKey) {
  if (!similarityReviewFor(data, side, scope, workspaceKey)
      || !Array.isArray(data.windows) || !Array.isArray(data.groups)) return false;
  const paired = scope.compareOffset !== null;
  const windowSize = paired ? Math.min(Math.floor(scope.limit / 2), 250) : scope.limit;
  const count = value => Number.isSafeInteger(value) && value >= 0;
  if (data.mode !== (paired ? "paired" : "batch") || data.windowSize !== windowSize
      || ![data.total, data.selectedCount, data.analyzedCount, data.invalidCount, data.groupCount].every(count)
      || data.selectedCount > scope.limit || data.selectedCount > data.total
      || data.analyzedCount + data.invalidCount !== data.selectedCount
      || data.groupCount !== data.groups.length || data.windows.length !== (paired ? 2 : 1)) return false;
  const offsets = paired ? [scope.offset, scope.compareOffset] : [scope.offset];
  if (!data.windows.every((window, batch) => window && window.batch === batch && window.offset === offsets[batch]
      && [window.count, window.analyzedCount, window.invalidCount].every(count)
      && window.count <= windowSize && window.analyzedCount + window.invalidCount === window.count
      && (window.count ? window.start === window.offset + 1 && window.end === window.offset + window.count
        : window.start === null && window.end === null))
      || data.windows.reduce((sum, window) => sum + window.count, 0) !== data.selectedCount) return false;
  const names = new Set();
  return data.groups.every(group => {
    if (!group || typeof group.id !== "string" || !Array.isArray(group.members) || group.members.length < 2
        || group.memberCount !== group.members.length || !Number.isFinite(group.minimumScore)) return false;
    let representatives = 0;
    const batches = new Set();
    for (const member of group.members) {
      if (!member || typeof member.name !== "string" || names.has(member.name)
          || typeof member.representative !== "boolean" || !Number.isFinite(member.score)
          || !offsets.some((_, batch) => batch === member.batch)
          || member.imageUrl !== `/api/assets/${side}/aligned/${encodeURIComponent(member.name)}`) return false;
      names.add(member.name);
      batches.add(member.batch);
      if (member.representative) {
        if (member.name !== group.representativeName) return false;
        representatives++;
      }
    }
    return representatives === 1 && (!paired || batches.size === 2) && names.size <= data.analyzedCount;
  });
}
