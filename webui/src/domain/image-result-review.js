// Versions share the exact locally staged original, rather than a filename or prompt.
export function imageResultVersions(tasks, selectedTask, originalId) {
  if (!selectedTask) return [];
  const related = (tasks ?? []).filter(task => originalId
    ? task.inputs?.some(input => input.inputId === originalId) : task.id === selectedTask.id);
  return related.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .flatMap(task => (task.results ?? []).map(result => ({ task, result, key: `${task.id}:${result.index}` })));
}
