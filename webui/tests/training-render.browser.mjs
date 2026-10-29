import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { chromium } from "playwright";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const webuiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const measured = ["PipelinePanel", "TrainingWorkspace", "StatusPanel", "TrainingChart"];
const delta = (after, before) => Object.fromEntries(measured.map(name => [name, (after[name] ?? 0) - (before[name] ?? 0)]));

async function launchBrowser() {
  for (const channel of [undefined, "msedge", "chrome"]) {
    try { return await chromium.launch({ headless: true, ...(channel ? { channel } : {}) }); } catch { /* Installed browsers only; never install dependencies. */ }
  }
  throw new Error("No installed browser is available for the isolated render regression");
}

test("terminal bursts leave training panels idle while metrics, previews and controls stay live", { timeout: 90000 }, async t => {
  const evidence = process.env.DFL_RENDER_QA_ROOT
    ? path.resolve(process.env.DFL_RENDER_QA_ROOT)
    : await mkdtemp(path.join(os.tmpdir(), "dfl-training-render-"));
  await mkdir(evidence, { recursive: true });
  const server = await createServer({
    configFile: false, root: webuiRoot, cacheDir: path.join(evidence, "vite-cache"),
    server: { host: "127.0.0.1", port: 0, strictPort: false, hmr: false },
    optimizeDeps: { include: ["react", "react-dom/client"] },
    plugins: [{
      name: "isolated-training-render-probes", enforce: "pre",
      transform(source, id) {
        const file = id.replaceAll("\\", "/").split("?")[0];
        if (file.endsWith("/src/components/TrainingView.jsx")) {
          let code = source;
          for (const name of measured) {
            const signature = new RegExp(`(function ${name}\\([\\s\\S]*?\\) \\{)`);
            assert.match(code, signature);
            code = code.replace(signature, `$1\n  useEffect(() => { window.__componentCommits["${name}"] = (window.__componentCommits["${name}"] ?? 0) + 1; });`);
          }
          return code;
        }
        if (file.endsWith("/src/runtime/useRuntime.js")) {
          return source.replace(/  return \{\r?\n    serviceState,/, "  window.__runtimeProbe = { jobs, selectedJob, get selectedEvents() { return terminalStoreRef.current.getSnapshot(selectedJobId).events; }, refresh, selectJob: setSelectedJobId };\n  return {\n    serviceState,");
        }
        if (file.endsWith("/src/App.jsx")) {
          return source.replace("export function App() {", "export function App() { window.__appRenderCount = (window.__appRenderCount ?? 0) + 1;")
            .replace("const [xsegDirty, setXsegDirty] = useState(false);", "const [xsegDirty, setXsegDirty] = useState(false); window.__setXSegDirty = setXsegDirty; window.__xsegDirty = xsegDirty;");
        }
      },
      configureServer(instance) {
        instance.middlewares.use(async (request, response, next) => {
          if (request.url?.split("?")[0] !== "/__training_render") return next();
          const html = await instance.transformIndexHtml(request.url, '<!doctype html><html><head><meta charset="utf-8"><title>DFL-PT-WEBUI isolated optimization QA</title></head><body><div id="root"></div><script type="module" src="/src/__tests__/training-render-fixture.jsx"></script></body></html>');
          response.setHeader("Content-Type", "text/html; charset=utf-8");
          response.end(html);
        });
      },
    }, react()],
  });
  t.after(() => server.close());
  await server.listen();
  const address = server.httpServer.address();
  const origin = `http://127.0.0.1:${address.port}`;
  const browser = await launchBrowser();
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: "zh-CN" });
  const errors = [];
  const consoleErrors = [];
  const unexpectedRequests = [];
  const plotRequests = [];
  let releasePlot;
  const plotGate = new Promise(resolve => { releasePlot = resolve; });
  await page.route("**/src/components/TrainingLossPlot.jsx*", async route => {
    await plotGate;
    await route.continue();
  });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (["error", "warning"].includes(message.type())) consoleErrors.push(message.text()); });
  page.on("request", request => { if (request.url().includes("/src/components/TrainingLossPlot.jsx")) plotRequests.push(request.url()); });
  await page.route("**/api/**", route => {
    if (route.request().url().includes("/preview?")) return route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6Bm0AAAAASUVORK5CYII=", "base64") });
    unexpectedRequests.push(route.request().url());
    return route.abort();
  });
  await page.goto(`${origin}/__training_render?lang=zh&emptyHistory=1`);
  await page.getByText("训练输出出现迭代数据后，将在此绘制真实损失曲线", { exact: true }).waitFor();
  assert.equal(plotRequests.length, 0, "empty history must not fetch the chart library's entry module");
  const emptyCanvas = await page.locator(".status-loss-chart .chart-canvas").boundingBox();
  assert.ok(emptyCanvas.height > 0);
  await page.evaluate(() => window.__trainingFixture.emit({ jobId: "training-a", sequence: 2, type: "job.metric", payload: { iteration: 1, srcLoss: 0.3, dstLoss: 0.4 } }));
  await page.getByText("正在加载训练损失曲线…", { exact: true }).waitFor();
  const loadingCanvas = await page.locator(".status-loss-chart .chart-canvas").boundingBox();
  releasePlot();
  await page.locator(".status-loss-chart .recharts-surface").waitFor();
  assert.equal(plotRequests.length, 1);
  const readyCanvas = await page.locator(".status-loss-chart .chart-canvas").boundingBox();
  assert.equal(readyCanvas.height, loadingCanvas.height, "lazy plotting must preserve the allocated loading canvas height");

  const failurePage = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: "zh-CN" });
  try {
    await failurePage.route("**/src/components/TrainingLossPlot.jsx*", route => route.fulfill({ status: 503, contentType: "text/javascript", body: "" }));
    await failurePage.goto(`${origin}/__training_render?lang=zh`);
    await failurePage.getByText("曲线暂时无法加载；后台训练会继续运行。", { exact: true }).waitFor();
    assert.equal(await failurePage.locator(".app-error-boundary").count(), 0);
    assert.equal(await failurePage.locator(".training-actions .button.primary").isEnabled(), true, "a plot chunk failure must leave training controls usable");
    await failurePage.locator(".training-actions .button.primary").click();
    await failurePage.waitForFunction(() => window.__trainingFixture.controls.length === 1);
    assert.equal(await failurePage.evaluate(() => window.__trainingFixture.controls[0].operation), "save");
  } finally { await failurePage.close(); }

  await page.goto(`${origin}/__training_render?lang=zh`);
  assert.match(await page.title(), /DFL-PT-WEBUI/);
  assert.equal(new URL(page.url()).pathname, "/__training_render");
  await page.waitForFunction(() => window.__runtimeProbe?.selectedJob?.id === "training-a" && window.__componentCommits?.TrainingChart > 0);
  await page.waitForTimeout(80);
  const snapshot = () => page.evaluate(() => ({ ...window.__componentCommits, root: window.__rootCommits, app: window.__appRenderCount }));
  const before = await snapshot();
  const started = Date.now();
  await page.evaluate(async () => {
    for (let sequence = 2; sequence <= 41; sequence += 1) {
      window.__trainingFixture.emit({ jobId: "training-a", sequence, type: "terminal.output", payload: { data: `line ${sequence}\r\n` } });
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  });
  await page.waitForFunction(() => window.__runtimeProbe.selectedEvents.at(-1)?.sequence === 41);
  assert.equal(await page.evaluate(() => window.__runtimeProbe.selectedEvents.length), 40);
  const after = await snapshot();
  const logBurst = delta(after, before);
  assert.deepEqual(logBurst, Object.fromEntries(measured.map(name => [name, 0])));
  assert.ok(after.root > before.root, "the actual React root must process the terminal events");
  assert.equal(after.app, before.app, "terminal subscriptions must not rerender App");

  await page.evaluate(() => window.__trainingFixture.emit({ jobId: "training-a", sequence: 42, type: "job.metric", payload: { iteration: 142, srcLoss: 0.2, dstLoss: 0.3, targetIterations: 1000 } }));
  await page.waitForFunction(() => window.__runtimeProbe.selectedJob.latestMetric.iteration === 142);
  const metricAfter = await snapshot();
  const metricChange = delta(metricAfter, after);
  assert.ok(metricChange.TrainingWorkspace > 0 && metricChange.StatusPanel > 0 && metricChange.TrainingChart > 0);
  assert.match(await page.locator(".status-panel").innerText(), /142/);

  await page.evaluate(() => window.__trainingFixture.emit({ jobId: "training-a", sequence: 43, type: "job.artifact", payload: { kind: "preview", version: 1 } }));
  await page.locator('.training-panel img[src*="preview?v=1"]').waitFor();
  const previewAfter = await snapshot();
  const previewChange = delta(previewAfter, metricAfter);
  assert.ok(previewChange.TrainingWorkspace > 0);
  assert.equal(previewChange.TrainingChart, 0, "preview changes must not rerender an unchanged loss chart");

  await page.evaluate(async () => {
    window.__trainingFixture.replaceJobs(window.__trainingFixture.jobs.map(job => ({ ...job, sequence: 60 })));
    await window.__runtimeProbe.refresh({ quiet: true });
  });
  await page.waitForTimeout(30);
  const pollAfter = await snapshot();
  assert.deepEqual(delta(pollAfter, previewAfter), Object.fromEntries(measured.map(name => [name, 0])));

  await page.evaluate(() => window.__trainingFixture.emit({ jobId: "training-a", sequence: 61, type: "job.state", payload: { state: "stopping", controls: [] } }));
  await page.waitForFunction(() => document.querySelector(".training-actions .button.primary")?.disabled === true);
  const stateAfter = await snapshot();
  assert.ok(stateAfter.TrainingWorkspace > pollAfter.TrainingWorkspace);
  assert.ok(stateAfter.StatusPanel > pollAfter.StatusPanel);

  await page.evaluate(async () => {
    const current = window.__trainingFixture.jobs[0];
    window.__trainingFixture.replaceJobs([
      { ...current, id: "training-b", createdAt: "2026-10-24T01:00:00Z", state: "running", sequence: 1, controls: ["save", "preview", "close"], parameters: { forceModelName: "fixture-new" } },
      { ...current, state: "cancelled", stopReason: "safe-stop", endedAt: "2026-10-24T00:59:00Z", sequence: 62 },
    ]);
    await window.__runtimeProbe.refresh({ quiet: true });
  });
  await page.locator(".training-actions .button.primary").click();
  await page.waitForFunction(() => window.__trainingFixture.controls.length > 0);
  assert.deepEqual(await page.evaluate(() => window.__trainingFixture.controls.at(-1)), { id: "training-b", operation: "save" });
  await page.screenshot({ path: path.join(evidence, "training-overview.png"), fullPage: true });

  const dialogs = [];
  page.on("dialog", async dialog => { dialogs.push(dialog.message()); await dialog.accept(); });
  await page.evaluate(() => window.__setXSegDirty(true));
  await page.waitForFunction(() => window.__xsegDirty === true);
  for (const side of ["src", "dst"]) {
    await page.evaluate(value => { window.location.hash = `#/aligned/${value}`; }, side);
    await page.locator(".dataset-view").waitFor();
    assert.match(await page.locator(".route-content").innerText(), new RegExp(side.toUpperCase()));
    assert.equal(await page.locator("vite-error-overlay").count(), 0);
  }
  assert.equal(dialogs.length, 1, "native navigation confirms unsaved masks exactly once");
  await page.screenshot({ path: path.join(evidence, "native-dst.png"), fullPage: true });
  await page.evaluate(() => { window.location.hash = "#/similarity/src"; });
  await page.locator(".similarity-workbench").waitFor();
  await page.waitForFunction(() => window.__trainingFixture.nativeQueries.length > 0);
  assert.match(await page.locator(".similarity-batches").innerText(), /500/);
  await page.locator(".similarity-batches select").selectOption("paired");
  await page.waitForFunction(() => window.__trainingFixture.nativeQueries.at(-1)?.compareOffset !== null);
  assert.match(await page.locator(".similarity-batches").innerText(), /250/);
  const nativeQueries = await page.evaluate(() => window.__trainingFixture.nativeQueries);
  assert.ok(nativeQueries.every(value => value.limit <= 500));
  assert.ok(nativeQueries.some(value => value.windowSize === 250 && value.offset !== value.compareOffset));
  assert.equal(await page.locator("vite-error-overlay").count(), 0);
  await page.screenshot({ path: path.join(evidence, "native-similarity.png"), fullPage: true });
  assert.deepEqual(errors, []);
  assert.deepEqual(consoleErrors, []);
  assert.deepEqual(unexpectedRequests, []);
  const report = { logBurst, rootCommitsDuringBurst: after.root - before.root, appRendersDuringBurst: after.app - before.app, metricChange, previewChange, sequenceOnlyPoll: delta(pollAfter, previewAfter), stateChange: delta(stateAfter, pollAfter), nativeQueries, title: await page.title(), url: page.url(), durationMs: Date.now() - started, errors, consoleErrors, unexpectedRequests };
  await writeFile(path.join(evidence, "render-counts.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ evidence, ...report }));
});
