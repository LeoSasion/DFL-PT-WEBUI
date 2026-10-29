import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { build, mergeConfig } from "vite";
import config from "../vite.config.mjs";

const webuiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const normalized = value => value.replaceAll("\\", "/");
const isChartModule = id => /\/node_modules\/(recharts|d3-[^/]+)\//.test(normalized(id));

function staticDependencies(root, chunks, seen = new Set()) {
  if (seen.has(root.fileName)) return seen;
  seen.add(root.fileName);
  for (const fileName of root.imports) {
    const dependency = chunks.get(fileName);
    if (dependency) staticDependencies(dependency, chunks, seen);
  }
  return seen;
}

test("production shell keeps charts dynamic and shares one React runtime", { timeout: 120_000 }, async () => {
  // Validate actual emitted dependency edges and HTML, rather than assuming
  // that a source-level import() guarantees a split production payload.
  const result = await build(mergeConfig(config, {
    root: webuiRoot,
    configFile: false,
    logLevel: "error",
    build: { write: false, emptyOutDir: false },
  }));
  const output = (Array.isArray(result) ? result : [result]).flatMap(value => value.output);
  const chunks = new Map(output.filter(value => value.type === "chunk").map(chunk => [chunk.fileName, chunk]));
  const entry = [...chunks.values()].find(chunk => chunk.isEntry);
  assert.ok(entry, "expected the workstation entry chunk");
  const initial = staticDependencies(entry, chunks);
  const chartChunks = [...chunks.values()].filter(chunk => Object.keys(chunk.modules).some(isChartModule));
  assert.ok(chartChunks.length > 0, "the regression must exercise the real plotting library");
  for (const chunk of chartChunks) assert.equal(initial.has(chunk.fileName), false, `initial static graph includes ${chunk.fileName}`);

  const plot = [...chunks.values()].find(chunk => normalized(chunk.facadeModuleId ?? "").endsWith("/components/TrainingLossPlot.jsx"));
  assert.ok(plot?.isDynamicEntry, "the training plot must remain a dynamic entry");
  const plotDependencies = staticDependencies(plot, chunks);
  assert.ok(chartChunks.some(chunk => plotDependencies.has(chunk.fileName)), "the plot must reach the real chart library");

  const html = output.find(value => value.type === "asset" && value.fileName === "index.html");
  assert.ok(html, "expected the generated index.html asset");
  for (const chunk of chartChunks) assert.equal(String(html.source).includes(chunk.fileName), false, `HTML preloads ${chunk.fileName}`);

  const reactOwners = [...chunks.values()].flatMap(chunk => Object.keys(chunk.modules)
    .filter(id => /\/node_modules\/react\/cjs\/react\.production\.js$/.test(normalized(id)))
    .map(id => ({ id, file: chunk.fileName })));
  assert.equal(reactOwners.length, 1, "the build must include exactly one core React production module");
  assert.ok(initial.has(reactOwners[0].file), "the shell must own the shared React runtime");
  assert.ok(plotDependencies.has(reactOwners[0].file), "the lazy plot must reuse the shell's React runtime");
});
