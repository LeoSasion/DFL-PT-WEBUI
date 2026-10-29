/** Isolated local QA. DFL_CUDA_UI_QA_ROOT must point at a disposable evidence folder.
 * Uses installed Playwright through NODE_PATH; never calls a real launcher host.
 */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { createServer, build } from "vite";
import config from "../vite.config.mjs";
import { cudaPreview } from "../src/cuda-preview.js";
import { applicationVersion, launcherVersion } from "../src/release-version.js";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const root = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const evidence = process.env.DFL_CUDA_UI_QA_ROOT;
if (!evidence || !path.isAbsolute(evidence)) throw new Error("Set DFL_CUDA_UI_QA_ROOT to an absolute isolated evidence folder.");
await mkdir(path.join(evidence, "screenshots"), { recursive: true });
const common = { ...config, configFile: false, root, cacheDir: path.join(evidence, "vite-cache") };
const report = { syntheticOnly: true, startedAt: new Date().toISOString(), tests: [], screenshots: [], errors: [], consoleErrors: [] };
const server = await createServer({ ...common, server: { host: "127.0.0.1", port: 0, strictPort: false, allowedHosts: ["127.0.0.1"] } });
await server.listen();
const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, locale: "zh-CN", reducedMotion: "reduce" });
context.on("page", (p) => {
  p.on("pageerror", (error) => report.errors.push(error.message));
  p.on("console", (message) => { if (message.type() === "error") report.consoleErrors.push({ text: message.text(), location: message.location() }); });
});
const page = await context.newPage();
const notice = () => page.getByRole("region", { name: "CUDA 运行环境" });
const requests = () => page.evaluate(async () => (await import("/src/bridge.js")).previewRequests());
const mutations = async () => (await requests()).filter((r) => /^(prepare|keep|delete|recover)CudaRuntime$/.test(r.method));
const pause = (ms = 100) => new Promise((resolve) => setTimeout(resolve, ms));
async function shot(name) {
  const output = path.join(evidence, "screenshots", name + ".png");
  await page.screenshot({ path: output, fullPage: true }); report.screenshots.push(output);
}
async function check(name, action) {
  await action(); report.tests.push({ name, passed: true }); console.log("PASS", name);
}
async function demo(query) {
  await page.goto(origin + "/?demo=ready&" + query);
  await page.locator(".terminal-tabs h1").waitFor({ state: "attached" });
  await pause();
}
async function nativeFixture(runtime, extra = {}, controls = {}) {
  const native = await context.newPage();
  await native.addInitScript(({ runtime, extra, controls }) => {
    const callbacks = [];
    const state = { mode: "ready", projectReady: true, projectDir: "C:\\synthetic-project-a", installPath: "C:\\synthetic-project-a", webuiRunning: false, terminalRunning: false, cudaRuntime: runtime, ...extra };
    if (runtime) state.cudaRuntime = { ...runtime, projectDir: state.projectDir };
    const api = { requests: [], held: [], mutations: [], selectedProject: state.projectDir,
      emit(data) {
        if (data.event === "state") Object.assign(state, data.data);
        if (data.event === "cudaRuntime") data = { ...data, data: { projectDir: state.projectDir, ...data.data } };
        for (const callback of callbacks) callback({ data });
      },
      answerChecks(result) { for (const request of api.held.splice(0)) api.emit({ id: request.id, result }); },
      answer(request, result, error) { api.emit({ id: request.id, ...(error ? { error: { message: error } } : { result }) }); },
    };
    window.__native = api;
    window.chrome ??= {};
    window.chrome.webview = { addEventListener(type, callback) { if (type === "message") callbacks.push(callback); }, postMessage(request) {
      api.requests.push(request);
      if (request.method === "getState") queueMicrotask(() => api.emit({ id: request.id, result: { state } }));
      else if (request.method === "checkCudaRuntime") api.held.push(request);
      else if (request.method === "openExternal") queueMicrotask(() => api.answer(request, { opened: true }));
      else if ((controls.holdMutations && /^(prepare|keep|delete|recover)CudaRuntime$/.test(request.method))
        || (controls.holdLifecycle && ["rollbackUpdate", "repairDependencies", "runBootstrap", "retryBootstrap"].includes(request.method))) {
        api.mutations.push(request);
        if (request.method.endsWith("CudaRuntime") && request.params.expectedProjectDir !== api.selectedProject) queueMicrotask(() => api.answer(request, null, "项目目录已变化，请重新检测后操作。"));
      }
      else throw new Error("Unexpected native mutation in read-only fixture: " + request.method);
    } };
  }, { runtime, extra, controls });
  await native.goto(origin + "/?demo=ready&cuda=cleanup-choice");
  await native.locator(extra.mode === "install" ? ".install-main h1" : ".terminal-tabs h1").waitFor({ state: "attached" }); await pause();
  return native;
}

try {
  await check("release labels and offline feedback match the release manifest", async () => {
    await demo("cuda=compatible");
    assert.equal(await page.locator(".titlebar__brand small").textContent(), launcherVersion);
    const feedback = await page.evaluate(async () => (await import("/src/bridge.js")).launcherBridge.request("prepareFeedback", { step: "install" }));
    assert.ok(feedback.preview.includes("应用版本：" + applicationVersion));
    assert.ok(feedback.preview.includes("启动器版本：" + launcherVersion));
  });
  await check("compatible state does not prompt or mutate", async () => {
    await demo("cuda=compatible");
    assert.equal(await notice().count(), 0); assert.deepEqual(await mutations(), []);
  });
  await check("shared feedback and upgrade dialogs preserve focus and safe dismissal", async () => {
    const feedback = page.getByRole("button", { name: "准备反馈信息", exact: true });
    await feedback.click();
    await page.getByRole("dialog", { name: "反馈信息预览" }).waitFor();
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "关闭");
    await page.keyboard.press("Escape"); await page.getByRole("dialog").waitFor({ state: "detached" });
    assert.equal(await feedback.evaluate((element) => element === document.activeElement), true);
    const upgrade = page.getByRole("button", { name: /^升级应用/ });
    await upgrade.click(); await page.getByRole("dialog").waitFor();
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "取消");
    await page.keyboard.press("Escape"); await page.getByRole("dialog").waitFor({ state: "detached" });
    assert.equal(await upgrade.evaluate((element) => element === document.activeElement), true);
  });
  await check("changed GPU offers a named download with FP32 guidance, never automatic", async () => {
    await demo("cuda=change-required");
    await notice().getByText(/GTX 10 系列训练请选择 FP32/).waitFor();
    assert.equal(await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).isEnabled(), true);
    assert.deepEqual(await mutations(), []); await shot("01-change-required-desktop");
  });
  await check("download is deduplicated and verified before cleanup; safe focus and Escape", async () => {
    await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).evaluate((button) => { button.click(); button.click(); });
    await page.getByRole("progressbar", { name: "运行环境准备进度" }).waitFor();
    assert.equal(await page.getByRole("button", { name: /^启动 WebUI/ }).isDisabled(), true);
    await shot("02-downloading-desktop");
    const dialog = page.getByRole("dialog"); await dialog.waitFor();
    assert.match(await dialog.innerText(), /CUDA 12.6 是当前使用的版本/);
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "保留旧版");
    assert.equal(await page.locator("#root").evaluate((element) => element.inert), true);
    await page.keyboard.press("Shift+Tab");
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "删除旧版");
    await page.keyboard.press("Tab");
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "保留旧版");
    assert.equal((await mutations()).filter((r) => r.method === "prepareCudaRuntime").length, 1);
    assert.equal((await mutations()).filter((r) => r.method === "deleteCudaRuntime").length, 0);
    await shot("03-cleanup-choice-desktop");
    await page.keyboard.press("Escape"); await dialog.waitFor({ state: "detached" });
    assert.equal(await page.locator("#root").evaluate((element) => element.inert), false);
    assert.equal((await mutations()).filter((r) => r.method === "deleteCudaRuntime").length, 0);
    assert.match(await page.evaluate(() => document.activeElement?.textContent), /选择是否保留旧版/);
  });
  await check("keep is persisted and does not repeatedly open cleanup", async () => {
    await page.getByRole("button", { name: "选择是否保留旧版" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "保留旧版", exact: true }).click();
    await page.getByRole("dialog").waitFor({ state: "detached" });
    await page.getByRole("heading", { name: "旧版运行环境已保留" }).waitFor();
    await page.getByRole("button", { name: "重新检测", exact: true }).click(); await pause(550);
    assert.equal(await page.getByRole("dialog").count(), 0);
    assert.equal((await mutations()).filter((r) => r.method === "keepCudaRuntime").length, 1);
    assert.equal((await mutations()).filter((r) => r.method === "deleteCudaRuntime").length, 0);
    await shot("04-retained-desktop");
  });
  await check("explicit deletion stays pending and binds confirmation to its transaction", async () => {
    await page.getByRole("button", { name: "清理保留的旧版" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "删除旧版", exact: true }).click();
    assert.equal(await page.getByRole("dialog").getAttribute("aria-busy"), "true");
    await page.keyboard.press("Escape"); assert.equal(await page.getByRole("dialog").count(), 1);
    await page.getByRole("dialog").waitFor({ state: "detached" });
    const deletion = (await mutations()).find((r) => r.method === "deleteCudaRuntime");
    assert.equal(deletion.params.confirmed, true); assert.equal(deletion.params.transactionId, "11111111111111111111111111111111");
    assert.equal(deletion.params.expectedProjectDir, "C:\\DFL-PT-WEBUI\\workspace");
    assert.equal(await notice().count(), 0);
  });
  await check("download failure preserves original environment and can retry", async () => {
    await demo("cuda=change-required&cudaFailure=once");
    await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).click();
    await notice().getByRole("alert").waitFor();
    assert.match(await notice().innerText(), /原运行环境保持不变/);
    assert.match(await notice().innerText(), /当前 CUDA 12.8/);
    assert.equal(await page.getByRole("dialog").count(), 0); await shot("05-download-failure");
    await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).click();
    await page.getByRole("dialog").waitFor();
    assert.equal((await mutations()).filter((r) => r.method === "prepareCudaRuntime").length, 2);
  });
  await check("delete failure remains inside dialog and retries without losing safe keep", async () => {
    await demo("cuda=cleanup-choice&cudaDeleteFailure=once");
    const dialog = page.getByRole("dialog"); await dialog.waitFor();
    await dialog.getByRole("button", { name: "删除旧版", exact: true }).click();
    await dialog.getByRole("alert").waitFor();
    assert.match(await dialog.innerText(), /旧运行包正在被占用/);
    assert.equal(await dialog.getByRole("button", { name: "保留旧版", exact: true }).isEnabled(), true);
    await shot("06-delete-failure");
    await dialog.getByRole("button", { name: "删除旧版", exact: true }).click();
    await dialog.waitFor({ state: "detached" });
    assert.equal((await mutations()).filter((r) => r.method === "deleteCudaRuntime").length, 2);
  });
  await check("active WebUI blocks maintenance and explains how to proceed", async () => {
    await demo("cuda=change-required&cudaBusy=1");
    assert.equal(await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).isDisabled(), true);
    assert.match(await notice().innerText(), /先保存并结束所有任务/);
    assert.deepEqual(await mutations(), []);
  });
  await check("interrupted switch offers recovery in Ready and Install, no ordinary repair", async () => {
    await demo("cuda=recover");
    assert.equal(await page.getByRole("button", { name: "恢复原运行环境" }).isEnabled(), true);
    assert.equal(await page.getByRole("button", { name: "修复依赖", exact: true }).isDisabled(), true);
    assert.equal(await page.getByRole("button", { name: /下载并切换到/ }).count(), 0);
    await shot("07-recovery-required");
    await page.getByRole("button", { name: "恢复原运行环境" }).click();
    await notice().waitFor({ state: "detached" });
    assert.equal((await mutations()).find((r) => r.method === "recoverCudaRuntime").params.transactionId, "22222222222222222222222222222222");
    await page.goto(origin + "/?demo=install&cuda=recover");
    await page.getByRole("button", { name: "恢复原运行环境" }).waitFor();
    assert.equal(await page.getByRole("button", { name: /下载并切换到/ }).count(), 0);
  });
  await check("narrow screen keeps actions and modal fully reachable with reduced motion", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await demo("cuda=change-required"); await shot("08-change-required-narrow");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    await page.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).click();
    const dialog = page.getByRole("dialog"); await dialog.waitFor();
    const bounds = await dialog.boundingBox(); assert.ok(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= 391 && bounds.y + bounds.height <= 845);
    assert.equal(await dialog.getByRole("button", { name: "保留旧版", exact: true }).isVisible(), true);
    await shot("09-cleanup-choice-narrow");
    const scrollbar = await dialog.locator(".launcher-modal__body").evaluate((element) => getComputedStyle(element).scrollbarColor);
    assert.notEqual(scrollbar, "auto");
    assert.equal(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches), true);
    await page.setViewportSize({ width: 1280, height: 900 });
  });
  await check("native missing state ignores synthetic URL and never invents cleanup", async () => {
    const native = await nativeFixture(null);
    assert.equal(await native.getByRole("dialog").count(), 0);
    assert.equal(await native.getByRole("region", { name: "CUDA 运行环境" }).count(), 0);
    assert.equal((await native.evaluate(() => window.__native.requests)).filter((r) => r.method.includes("CudaRuntime")).length, 0);
    await native.close();
  });
  await check("stale hardware response cannot overwrite progress or another project", async () => {
    const native = await nativeFixture(cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    await native.evaluate(() => window.__native.emit({ event: "cudaRuntime", data: { status: "downloading", canSwitch: false, message: "正在下载最新选择。" } }));
    await native.evaluate((runtime) => window.__native.answerChecks({ cudaRuntime: runtime }), cudaPreview("change-required"));
    await native.getByRole("progressbar", { name: "运行环境准备进度" }).waitFor();
    assert.match(await native.getByRole("region", { name: "CUDA 运行环境" }).innerText(), /正在下载最新选择/);
    await native.evaluate((runtime) => window.__native.emit({ event: "cudaRuntime", data: runtime }), cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    await native.evaluate(() => window.__native.emit({ event: "state", data: { projectDir: "C:\\synthetic-project-b", installPath: "C:\\synthetic-project-b", cudaRuntime: null } }));
    await native.evaluate((runtime) => window.__native.answerChecks({ cudaRuntime: runtime }), cudaPreview("change-required"));
    await pause(); assert.equal(await native.getByRole("region", { name: "CUDA 运行环境" }).count(), 0);
    await native.close();
  });
  await check("cleanup decision expires when the native project changes", async () => {
    const native = await nativeFixture(cudaPreview("cleanup-choice"));
    await native.getByRole("dialog").waitFor();
    await native.evaluate(() => window.__native.emit({ event: "state", data: { projectDir: "C:\\synthetic-project-b", installPath: "C:\\synthetic-project-b", cudaRuntime: null } }));
    await native.getByRole("dialog").waitFor({ state: "detached" });
    assert.equal((await native.evaluate(() => window.__native.requests)).filter((r) => /^(prepare|keep|delete|recover)CudaRuntime$/.test(r.method)).length, 0);
    assert.equal(await native.locator("#root").evaluate((element) => element.inert), false);
    await native.close();
  });
  await check("historical records are reachable and interrupted deletion cannot be kept", async () => {
    await demo("cuda=history");
    await page.getByRole("button", { name: "管理保留的运行包（2）" }).click();
    let dialog = page.getByRole("dialog");
    assert.equal(await dialog.locator("li").count(), 2);
    await dialog.getByRole("button", { name: "查看清理选项" }).click();
    dialog = page.getByRole("dialog");
    assert.match(await dialog.innerText(), /CUDA 12.6/); assert.match(await dialog.innerText(), /3.5 GB/);
    await dialog.getByRole("button", { name: "保留旧版", exact: true }).click();
    await dialog.waitFor({ state: "detached" });
    assert.equal((await mutations()).find((r) => r.method === "keepCudaRuntime").params.transactionId, "33333333333333333333333333333333");
    await page.getByRole("button", { name: "管理保留的运行包（2）" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "继续清理", exact: true }).click();
    dialog = page.getByRole("dialog");
    assert.equal(await dialog.getByRole("button", { name: "保留旧版", exact: true }).count(), 0);
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "稍后处理");
    assert.match(await dialog.innerText(), /不能再次标记为保留/);
    assert.match(await dialog.innerText(), /本项目中选中的旧版 CUDA 运行包/);
    assert.doesNotMatch(await dialog.innerText(), /上一次保留的旧版/);
    assert.match(await dialog.innerText(), /原运行包大小/);
    assert.doesNotMatch(await dialog.innerText(), /可释放空间/);
    await shot("10-historical-interrupted-delete");
    await dialog.getByRole("button", { name: "继续删除旧版", exact: true }).evaluate((button) => { button.click(); button.click(); });
    await dialog.waitFor({ state: "detached" });
    const deletes = (await mutations()).filter((r) => r.method === "deleteCudaRuntime");
    assert.equal(deletes.length, 1); assert.equal(deletes[0].params.transactionId, "44444444444444444444444444444444");
    assert.equal(deletes[0].params.confirmed, true); assert.ok(deletes[0].params.expectedProjectDir);
    await page.getByRole("button", { name: "管理保留的运行包（1）" }).waitFor();
  });
  await check("cache-only preparation is distinct from an already retained runtime", async () => {
    const runtime = { ...cudaPreview("change-required"), cachedBytes: 2593299994, downloadRemainingBytes: 0, downloadRequired: false };
    const native = await nativeFixture(runtime);
    await native.getByRole("button", { name: "准备并切换到 CUDA 12.6" }).waitFor();
    assert.match(await native.getByRole("region", { name: "CUDA 运行环境" }).innerText(), /已有下载缓存/);
    assert.equal(await native.getByRole("button", { name: /切换到已保留的/ }).count(), 0);
    await native.close();
    const partial = await nativeFixture({ ...cudaPreview("change-required"), downloadRemainingBytes: 593299994, cachedBytes: 2000000000 });
    assert.match(await partial.getByRole("region", { name: "CUDA 运行环境" }).innerText(), /还需下载约 593.3 MB/);
    await partial.close();
    const downloading = await nativeFixture({ ...cudaPreview("downloading"), downloadBytes: 2593299994, cachedBytes: 2585299994, downloadRemainingBytes: 8000000, downloadedBytes: 4000000 });
    assert.equal(await downloading.getByRole("progressbar").getAttribute("value"), "50");
    assert.match(await downloading.getByRole("region", { name: "CUDA 运行环境" }).innerText(), /4 MB \/ 8 MB/);
    assert.doesNotMatch(await downloading.getByRole("region", { name: "CUDA 运行环境" }).innerText(), /\/ 2.6 GB/);
    await downloading.close();
    const preparing = await nativeFixture({ ...runtime, status: "downloading", downloadedBytes: 0 });
    assert.equal(await preparing.getByRole("progressbar").getAttribute("value"), null);
    await preparing.close();
  });
  await check("project-bound mutations reject a stale displayed project and late events", async () => {
    const native = await nativeFixture(cudaPreview("change-required"), {}, { holdMutations: true });
    await native.evaluate(() => { window.__native.selectedProject = "C:\\synthetic-project-b"; });
    await native.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).click();
    await native.getByRole("alert").waitFor();
    assert.equal((await native.evaluate(() => window.__native.mutations))[0].params.expectedProjectDir, "C:\\synthetic-project-a");
    await native.close();
    const late = await nativeFixture(cudaPreview("change-required"), {}, { holdMutations: true });
    await late.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).click();
    await late.evaluate(() => window.__native.emit({ event: "state", data: { projectDir: "C:\\synthetic-project-b", installPath: "C:\\synthetic-project-b", cudaRuntime: null } }));
    await late.evaluate(() => {
      window.__native.emit({ event: "cudaRuntime", data: { projectDir: "C:\\synthetic-project-a", status: "downloading", message: "OLD PROJECT" } });
      window.__native.answer(window.__native.mutations[0], { state: { projectDir: "C:\\synthetic-project-a", cudaRuntime: { status: "cleanup-choice", transactionId: "11111111111111111111111111111111" } } });
    });
    await pause(); assert.equal(await late.getByRole("region", { name: "CUDA 运行环境" }).count(), 0);
    assert.equal(await late.getByRole("dialog").count(), 0); await late.close();
  });
  await check("old project check does not block or clear a newer project's check", async () => {
    const native = await nativeFixture(cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    await native.evaluate((runtime) => window.__native.emit({ event: "state", data: { projectDir: "C:\\synthetic-project-b", installPath: "C:\\synthetic-project-b", cudaRuntime: { ...runtime, projectDir: "C:\\synthetic-project-b" } } }), cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    assert.equal(await native.evaluate(() => window.__native.held.length), 2);
    await native.evaluate((runtime) => window.__native.answer(window.__native.held[0], { projectDir: "C:\\synthetic-project-a", cudaRuntime: runtime }), cudaPreview("compatible"));
    assert.equal(await native.getByRole("button", { name: "检测中", exact: true }).isDisabled(), true);
    await native.evaluate((runtime) => window.__native.answer(window.__native.held[1], { projectDir: "C:\\synthetic-project-b", cudaRuntime: { ...runtime, projectDir: "C:\\synthetic-project-b" } }), cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).waitFor();
    assert.equal(await native.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).count(), 1);
    await native.close();
  });
  await check("recovery double-click is deduplicated and failure stays retryable", async () => {
    const native = await nativeFixture(cudaPreview("recover"), {}, { holdMutations: true });
    await native.getByRole("button", { name: "恢复原运行环境" }).evaluate((button) => { button.click(); button.click(); });
    assert.equal(await native.evaluate(() => window.__native.mutations.length), 1);
    await native.evaluate(() => window.__native.answer(window.__native.mutations[0], null, "合成恢复失败，可重试。"));
    await native.getByRole("alert").filter({ hasText: "合成恢复失败" }).waitFor();
    assert.equal(await native.getByRole("button", { name: "恢复原运行环境" }).isEnabled(), true);
    await native.getByRole("button", { name: "恢复原运行环境" }).click();
    assert.equal(await native.evaluate(() => window.__native.mutations.length), 2);
    assert.equal((await native.evaluate(() => window.__native.mutations))[1].params.expectedProjectDir, "C:\\synthetic-project-a");
    await native.close();
  });
  await check("manual check errors retain actions and clear after a successful retry", async () => {
    const native = await nativeFixture(cudaPreview("change-required"));
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    await native.evaluate(() => window.__native.answer(window.__native.held[0], null, "合成环境检测失败，请重试。"));
    await native.getByRole("alert").waitFor();
    assert.equal(await native.getByRole("button", { name: "下载并切换到 CUDA 12.6" }).isEnabled(), true);
    await native.getByRole("button", { name: "重新检测", exact: true }).click();
    await native.evaluate((runtime) => window.__native.answer(window.__native.held[1], { projectDir: "C:\\synthetic-project-a", cudaRuntime: runtime }), cudaPreview("change-required"));
    await native.getByRole("alert").waitFor({ state: "detached" });
    await native.close();
  });
  await check("committed runtime cleanup warning has a distinct explicit retry", async () => {
    await demo("cuda=cleanup-warning");
    assert.match(await notice().innerText(), /部分下载与临时残留/);
    assert.equal(await page.getByRole("button", { name: "恢复原运行环境" }).count(), 0);
    const retry = page.getByRole("button", { name: "重试清理", exact: true });
    await retry.evaluate((button) => { button.click(); button.click(); });
    await notice().waitFor({ state: "detached" });
    const mutation = await mutations(); assert.equal(mutation.length, 1);
    assert.equal(mutation[0].method, "recoverCudaRuntime");
    assert.equal(mutation[0].params.transactionId, "55555555555555555555555555555555");
    assert.ok(mutation[0].params.expectedProjectDir);
  });
  await check("cold source-upgrade recovery replaces ordinary installation and retries visibly", async () => {
    const native = await nativeFixture(null, { mode: "install", upgradePending: true, upgradeCanRollback: true, upgradeStatus: "interrupted", upgradeMessage: "合成升级中断，已保留备份。" }, { holdLifecycle: true });
    const recover = native.getByRole("button", { name: "恢复未完成的升级", exact: true });
    await recover.evaluate((button) => { button.click(); button.click(); });
    assert.equal(await native.evaluate(() => window.__native.mutations.length), 1);
    assert.equal((await native.evaluate(() => window.__native.mutations))[0].method, "rollbackUpdate");
    await native.evaluate(() => window.__native.answer(window.__native.mutations[0], null, "合成恢复失败，可以重试。"));
    await native.getByRole("alert").waitFor(); assert.equal(await recover.isEnabled(), true);
    await native.screenshot({ path: path.join(evidence, "screenshots", "11-source-cold-recovery.png") });
    await recover.click();
    await native.evaluate(() => window.__native.answer(window.__native.mutations[1], { state: { mode: "ready", environmentStatus: "ready", upgradePending: false, cudaRuntime: null } }));
    await native.getByRole("heading", { name: "交互终端" }).waitFor();
    assert.equal((await native.evaluate(() => window.__native.requests)).filter((request) => ["runBootstrap", "retryBootstrap"].includes(request.method)).length, 0);
    await native.close();
  });
  await check("unrecoverable or changing upgrade cannot start ordinary bootstrap", async () => {
    const native = await nativeFixture(null, { mode: "install", upgradePending: true, upgradeCanRollback: false, upgradeStatus: "invalid", upgradeMessage: "合成恢复记录不完整，请查看日志。" }, { holdLifecycle: true });
    assert.equal(await native.getByRole("button", { name: "恢复未完成的升级", exact: true }).isDisabled(), true);
    assert.match(await native.getByRole("status").innerText(), /恢复记录不完整/);
    assert.equal(await native.getByRole("button", { name: "开始安装", exact: true }).count(), 0);
    await native.evaluate(() => window.__native.emit({ event: "state", data: { upgradeCanRollback: true, environmentStatus: "changing" } }));
    assert.equal(await native.getByRole("button", { name: "恢复未完成的升级", exact: true }).isDisabled(), true);
    assert.equal(await native.evaluate(() => window.__native.mutations.length), 0); await native.close();
  });
  await check("cold Python runtime preparation uses repair rather than source rollback or bootstrap", async () => {
    const native = await nativeFixture(null, { mode: "install", runtimeRepairPending: true, upgradePending: false }, { holdLifecycle: true });
    const repair = native.getByRole("button", { name: "恢复运行环境", exact: true });
    await repair.click(); assert.equal(await repair.isDisabled(), true);
    const request = (await native.evaluate(() => window.__native.mutations))[0];
    assert.equal(request.method, "repairDependencies");
    await native.evaluate(() => window.__native.answer(window.__native.mutations[0], { state: { mode: "ready", environmentStatus: "ready", runtimeRepairPending: false, cudaRuntime: null } }));
    await native.getByRole("heading", { name: "交互终端" }).waitFor();
    assert.equal((await native.evaluate(() => window.__native.requests)).filter((item) => ["runBootstrap", "retryBootstrap", "rollbackUpdate"].includes(item.method)).length, 0);
    await native.close();
  });
  await check("missing auxiliary resources show official guidance, preserve cache on retry and report failures", async () => {
    const resource = { complete: false, failed: [{ group: "tufa", path: "weights/model.pth", manual: true,
      url: "https://drive.google.com/file/d/1rRAB3W74QK-J6W_EnjC7HhS-zjT9BuED/view", savePath: ".launcher-install/vision/manual/tufa/weights/model.pth",
      sha256: "3fe7071bef814af320c6e6ec7d724b4b1bb253677500e1a48a5e2a66aaf10899", sizeBytes: 143969647, error: "Synthetic download unavailable" }] };
    const native = await nativeFixture(null, { mode: "install", environmentStatus: "incomplete", resourcePreparation: resource }, { holdLifecycle: true });
    const panel = native.getByRole("region", { name: "辅助资源准备" });
    assert.match(await panel.innerText(), /已校验文件会复用/);
    assert.match(await panel.innerText(), /manual\/tufa\/weights\/model.pth/);
    await panel.getByRole("button", { name: "打开官方获取页面" }).click();
    assert.equal((await native.evaluate(() => window.__native.requests)).find((item) => item.method === "openExternal").params.url, resource.failed[0].url);
    const retry = native.getByRole("button", { name: "重试准备辅助资源", exact: true });
    await retry.evaluate((button) => { button.click(); button.click(); });
    assert.equal(await native.evaluate(() => window.__native.mutations.length), 1);
    assert.equal((await native.evaluate(() => window.__native.mutations))[0].method, "repairDependencies");
    await native.evaluate(() => window.__native.answer(window.__native.mutations[0], null, "辅助资源仍缺失，可以重试。"));
    await native.getByRole("alert").filter({ hasText: "辅助资源仍缺失" }).waitFor();
    assert.equal(await retry.isEnabled(), true);
    await native.setViewportSize({ width: 1000, height: 720 });
    await retry.scrollIntoViewIfNeeded();
    assert.equal(await retry.isVisible(), true);
    await native.screenshot({ path: path.join(evidence, "screenshots", "12-resource-guidance.png") });
    await retry.click();
    await native.evaluate(() => window.__native.answer(window.__native.mutations[1], { state: { mode: "ready", environmentStatus: "ready", resourcePreparation: { complete: true, failed: [] }, cudaRuntime: null } }));
    await native.getByRole("heading", { name: "交互终端" }).waitFor();
    await native.close();
  });
  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.consoleErrors, []);
  await build({ ...common, build: { ...config.build, outDir: path.join(evidence, "build/client"), emptyOutDir: true } });
  report.tests.push({ name: "isolated production build", passed: true });
  report.passed = true;
} catch (error) {
  report.passed = false; report.failure = error.stack; await shot("failure"); throw error;
} finally {
  report.finishedAt = new Date().toISOString();
  await writeFile(path.join(evidence, "browser-report.json"), JSON.stringify(report, null, 2));
  await browser.close(); await server.close();
}
