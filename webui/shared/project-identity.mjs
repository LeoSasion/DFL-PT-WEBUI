export const PROJECT_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,47}$/;
const reservedDirectory = /^(?:default|con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

export function defaultProjectRegistry() {
  return { schemaVersion: 1, activeId: "default", projects: [
    { id: "default", name: "默认项目", managed: false, createdAt: null },
  ] };
}

export function normalizeProjectRegistry(value) {
  if (!Array.isArray(value?.projects) || typeof value.activeId !== "string"
      || !PROJECT_ID_PATTERN.test(value.activeId)
      || !value.projects.some(project => project?.id === value.activeId)
      || value.projects.some(project => typeof project?.id !== "string" || !PROJECT_ID_PATTERN.test(project.id))
      || new Set(value.projects.map(project => project.id)).size !== value.projects.length) {
    throw new Error("Invalid project registry");
  }
  return { schemaVersion: 1, activeId: value.activeId, projects: [
    defaultProjectRegistry().projects[0],
    ...value.projects.filter(project => project.id !== "default").map(project => ({
      id: project.id,
      name: typeof project.name === "string" && project.name.trim() ? project.name.trim().slice(0, 64) : project.id,
      managed: true,
      createdAt: typeof project.createdAt === "string" ? project.createdAt : null,
    })),
  ] };
}

export function parseProjectRegistry(text) {
  return normalizeProjectRegistry(JSON.parse(text.replace(/^\uFEFF/, "")));
}

export function normalizeProjectName(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ");
}

export function normalizeProjectId(value) {
  return String(value ?? "").trim().toLocaleLowerCase("en-US")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
}

export function projectIdentityIssues(name, id, projects = []) {
  return {
    name: !name || name.length > 64 || /[<>:"/\\|?*\u0000-\u001f]/.test(name) ? "PROJECT_NAME_INVALID" : null,
    id: !PROJECT_ID_PATTERN.test(id) || reservedDirectory.test(id) ? "PROJECT_ID_INVALID"
      : projects.some(project => project.id.toLocaleLowerCase("en-US") === id) ? "PROJECT_EXISTS" : null,
  };
}
