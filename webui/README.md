# DFL-PT-WEBUI 工作台

React 界面与本机 Node 服务。换脸训练、合成和 DFM 注册项只接受 ME PyTorch；XSeg 与数据工具使用同一独立 Python 环境。本地服务只监听 `127.0.0.1`，白名单命令直接启动 Python，不执行任意 shell。

从仓库根目录双击 `启动 WebUI.bat`。页面 <http://127.0.0.1:4173>，Runtime `127.0.0.1:4174`。环境与完整使用步骤见 [根 README](../README.md)。

在本目录开发时先将 `../_internal/node/bin` 加入 PATH，使用其 `corepack.cmd pnpm`：

```powershell
pnpm install --frozen-lockfile
pnpm build
pnpm serve:local
pnpm status:local
pnpm stop:local
pnpm test
```

`pnpm test` 创建隔离目录，避免测试修改当前素材/模型。`pnpm test:e2e` 使用已有 Playwright 或系统 Chrome/Edge；真实训练烟测需要显式设置 `DFL_E2E_MUTATING=1`，并先准备人工测试项目和 aligned 数据。该测试会创建/继续模型并保存停止。

主界面中“模型训练”选择任务和模型，“总览”显示 ME 预览、损失及控制按钮，“质量诊断”读取只读评估快照。辅助 XSeg 的配置问答仍在终端交互。模型和任务分别保存在活动项目的 `model` 和 `.webui`，不提交 Git。

`server/command-registry.mjs` 是任务白名单，`paths.mjs` 描述本目录运行路径；模型发现采用 `metadata.json + me.pt`。旧版本 UI 文档与验收结论保留在旧仓库，新版本证据见 [验证记录](../docs/VALIDATION.md)。
