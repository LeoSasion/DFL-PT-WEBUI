// Alternative commands (such as encoding formats) share one current outcome.
// Older successes and failures remain in history, but must not override it.
export function latestWorkflowJob(jobs = [], commandIds = []) {
  const allowed = new Set(commandIds);
  let latest = null, latestTime = -Infinity;
  for (const job of jobs) {
    if (!allowed.has(job.commandId)) continue;
    const parsed = Date.parse(job.createdAt ?? job.startedAt ?? "");
    const time = Number.isFinite(parsed) ? parsed : -Infinity;
    if (!latest || time > latestTime) { latest = job; latestTime = time; }
  }
  return latest;
}
