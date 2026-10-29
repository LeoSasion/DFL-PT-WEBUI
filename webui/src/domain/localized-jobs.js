// Cache by immutable source object, not by ID: metrics and controls must replace
// their view immediately, while another job's update can reuse this one's view.
export function createLocalizedJobSelector() {
  const cache = new WeakMap();
  let previous = [];
  return (jobs, commandById) => {
    const next = jobs.map(job => {
      const label = commandById.get(job.commandId)?.label ?? job.label;
      if (label === job.label) return job;
      const cached = cache.get(job);
      if (cached?.label === label) return cached;
      const localized = { ...job, label };
      cache.set(job, localized);
      return localized;
    });
    if (next.length !== previous.length || next.some((job, index) => job !== previous[index])) previous = next;
    return previous;
  };
}
