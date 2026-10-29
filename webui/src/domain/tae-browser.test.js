import assert from "node:assert/strict";
import test from "node:test";
import { chromium } from "playwright";
import { listCommands } from "../../server/command-registry.mjs";
import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const origin = process.env.TAE_TEST_ORIGIN ?? "http://127.0.0.1:4185";
const commands = listCommands().filter(command => ["train.me", "train.tae"].includes(command.id));

async function browserInstance() {
  for (const channel of [undefined, "msedge", "chrome"]) {
    try { return await chromium.launch({ headless: true, ...(channel ? { channel } : {}) }); } catch { /* Try installed fallback. */ }
  }
  throw new Error("No browser available for TAE UI smoke test");
}

test("TAE WebUI config, resume and controls use their separate model contract", { timeout: 60000 }, async () => {
  const browser = await browserInstance();
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: "zh-CN" });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  let fixture = { commands, models: [] };
  await page.route("**/__tae_ui*", route => route.fulfill({ contentType: "text/html; charset=utf-8", body: `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head><body><div id="root"></div>
    <script>window.taeFixture=${JSON.stringify(fixture).replaceAll("<", "\\u003c")}</script>
    <script type="module">import RefreshRuntime from '/@react-refresh';RefreshRuntime.injectIntoGlobalHook(window);window.$RefreshReg$=()=>{};window.$RefreshSig$=()=>type=>type;window.__vite_plugin_react_preamble_installed__=true;await import('/src/__tests__/tae-ui-fixture.jsx')</script></body></html>` }));
  try {
    await page.goto(`${origin}/__tae_ui?lang=zh`);
    const h3Fields = page.getByRole("region", { name: "TAEQI2.1 训练配置" });
    await h3Fields.waitFor();
    const resolutionInput = page.locator('select[id$="-resolution"]');
    assert.equal(await resolutionInput.inputValue(), "256");
    assert.deepEqual(await resolutionInput.locator("option").evaluateAll(options => options.map(option => option.value)), ["256", "448", "512", "640", "832"]);
    assert.equal(await page.locator('input[id$="-resolution"]').count(), 0);
    for (const size of ["448", "640", "832", "256"]) {
      await resolutionInput.selectOption(size);
      assert.equal(await resolutionInput.inputValue(), size);
    }
    const decoder = page.locator('select[id$="-decoder_mode"]');
    assert.equal(await decoder.inputValue(), 'frozen');
    for (const mode of ['tail','full','frozen']) {
      await decoder.selectOption(mode);
      assert.equal(await decoder.inputValue(), mode);
    }
    const vaeInput = page.locator('input[id$="-vae_weights"]');
    await vaeInput.fill("E:\\weights\\taeqi2_1.safetensors");
    await page.getByRole("textbox", { name: /新模型名称/ }).fill("h3-ui-contract");
    const evidence = await mkdtemp(path.join(os.tmpdir(), "dfl-h3-policy-ui-"));
    await page.screenshot({ path: path.join(evidence, "h3-configuration.png"), fullPage: true });
    await page.getByRole("button", { name: "下一步", exact: true }).click();
    await page.getByText("前置检查通过", { exact: true }).waitFor();
    const preflight = await page.evaluate(() => window.taePreflight);
    assert.equal(preflight.commandId, "train.tae");
    assert.equal(preflight.options.parameters.vae_weights, "E:\\weights\\taeqi2_1.safetensors");
    assert.equal(preflight.options.parameters.resolution, 256);
    assert.equal(preflight.options.parameters.half_resolution_probability, 0);
    assert.equal(preflight.options.parameters.inter_dims, 256);
    assert.equal(preflight.options.parameters.archi, undefined);
    assert.equal(await page.getByRole("button", { name: "启动任务", exact: true }).isEnabled(), true);

    fixture = { commands, models: [{ name: "saved-h3", type: "TAE", format: "tae-liae-pytorch", files: ["tae.pt"],
      config: { resolution: 448, batch_size: 2, random_flip: false }, vae_weights: "E:\\weights\\saved.safetensors" }] };
    await page.reload();
    await page.getByRole("region", { name: "TAEQI2.1 训练配置" }).waitFor();
    assert.equal(await resolutionInput.inputValue(), "448");
    assert.equal(await resolutionInput.isDisabled(), true);
    assert.equal(await decoder.isDisabled(), false);
    await decoder.selectOption('tail');
    assert.equal(await vaeInput.inputValue(), "E:\\weights\\saved.safetensors");
    assert.equal(await vaeInput.isDisabled(), false);
    await vaeInput.fill("E:\\moved\\saved.safetensors");
    await page.getByRole("button", { name: "下一步", exact: true }).click();
    await page.getByText("前置检查通过", { exact: true }).waitFor();
    assert.equal((await page.evaluate(() => window.taePreflight)).options.parameters.vae_weights, "E:\\moved\\saved.safetensors");
    await page.getByRole("button", { name: "上一步", exact: true }).click();
    await page.getByRole("button", { name: "更换任务", exact: true }).click();
    await page.getByRole("combobox", { name: "任务类型", exact: true }).selectOption("train.me");
    await page.getByRole("button", { name: "下一步", exact: true }).click();
    await page.getByRole("region", { name: "ME 训练配置", exact: true }).waitFor();
    assert.equal(await page.getByRole("region", { name: "TAEQI2.1 训练配置" }).count(), 0);
    assert.equal(await page.getByRole("combobox", { name: "使用模型", exact: true }).inputValue(), "__new__");

    fixture = { mode: "controls", commands, models: [] };
    await page.reload();
    await page.getByRole("heading", { name: "训练任务 · TAEQI2.1" }).waitFor();
    assert.equal(await page.getByRole("button", { name: "保存", exact: true }).count(), 1);
    assert.equal(await page.getByRole("button", { name: "刷新预览", exact: true }).count(), 1);
    assert.equal(await page.getByRole("button", { name: "安全停止", exact: true }).count(), 1);
    assert.equal(await page.getByRole("button", { name: "备份", exact: true }).count(), 0);
    assert.equal(await page.getByRole("button", { name: "评估快照", exact: true }).count(), 0);
    assert.deepEqual(pageErrors, []);
  } catch (error) {
    console.error((await page.locator("body").innerText()).slice(0, 6000));
    console.error("browser errors", pageErrors);
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
});
