# Windows 一键启动器

[下载 DFL-PT-WEBUI.Launcher.exe](https://github.com/LeoSasion/DFL-PT-WEBUI/releases/download/launcher-v0.1.2-preview/DFL-PT-WEBUI.Launcher.exe)，双击后选择独立空目录并开始安装。首次安装无需预装 Git、Python 或 Node；安装完成后，启动器保留在项目根目录 `DFL-PT-WEBUI.exe`。以后双击它，点击“启动 WebUI”即可使用。

启动器也可以选择已有的完整 PT 项目，检查本地运行环境并启动 WebUI。原版 DFL-WEBUI 与 PT 版必须使用不同目录，不共享 Python 环境、启动器配置或工作区。两个项目默认使用相同端口；切换时先在另一个项目中停止 WebUI。启动器不会结束其他项目的服务。

## 首次安装

启动器从固定的本项目官方 GitHub `main` 源码地址下载项目，不需要 Git，也不会下载原版项目。源码先展开到安装目录内的暂存区，检查 PT 产品标识和 ME/PyTorch 入口后才发布到所选目录，再调用源码内的 `launcher/install-source.ps1` 准备 Python 3.12、PyTorch 2.9.1 CUDA 12.8、Node 24.19.0、FFmpeg、锁定的 WebUI 依赖、辅助权重及通用 XSeg 推理权重，并构建 WebUI。

请选择独立空目录，例如 `C:\DFL-PT-WEBUI`。不要选择原版项目目录，不要覆盖正在使用的环境。下载与解压需要额外磁盘空间；首次安装需要访问 GitHub、Node.js、PyTorch 和 Python/Node 包仓库。已有运行时或依赖会按照源码安装入口的规则检查和复用，不匹配时给出原因。离线安装参数、固定来源与散列、原生依赖要求见[运行环境说明](../launcher/runtime-README.md)。

运行条件是 Windows 10/11 x64、.NET Framework 4.8。若缺少 WebView2 Runtime，启动器会下载并验证 Microsoft 官方签名的安装程序。GPU 训练另外需要支持 CUDA 12.8 的 NVIDIA 驱动；CUDA/cuDNN 来自 PyTorch wheel。

安装失败时保留日志和已完成的依赖下载，处理界面指出的网络、空间或环境问题后重试。请在 PT 安装目录内补齐环境。安装日志及复用规则见[运行环境说明](../launcher/runtime-README.md#已有环境与失败重试)。

## 安装后的使用与更新

安装完成后使用项目根目录 `DFL-PT-WEBUI.exe`，原有 `启动 WebUI.bat` 和 `传统命令菜单.bat` 仍可使用。启动器的检查和安装本身不运行训练，不下载用户素材。首次进入 WebUI 后再建立自己的项目并导入素材。

启动器设置单独保存在 `%LocalAppData%/DFL-PT-WEBUI/Launcher`，素材和模型保存在所选 PT 项目的 `workspace` 或 `workspaces`。原版启动器配置不会迁入这里。

启动器的 Git 更新和 EXE 自动更新均继续关闭。`launcher-v0.1.2-preview` 仅交付独立启动器；它安装的是当前官方 `main`，该源码已包含图像服务功能。旧 `v0.1.1-preview` 源码 ZIP、便携分卷和对应摘要不会被这个发布替换。需要升级时先停止任务并备份工作区，通过发行页的说明准备新的环境。

## 发行资产与验证范围

启动器发行页提供三个文件：

- `DFL-PT-WEBUI.Launcher.exe`：内嵌界面、安装脚本、运行时检查、终端桥及 WebView2 SDK 和许可说明的单文件启动器，同时保留 React、图标和终端界面运行包的 MIT 版权与许可全文。
- `SHA256SUMS.launcher`：EXE 和来源记录的 SHA-256，独立于旧便携包的 `SHA256SUMS`。
- `DFL-PT-WEBUI.Launcher.provenance.json`：公开源码提交和代码树、启动器版本、EXE 摘要、内嵌资源摘要与许可来源，不包含本机目录、账户、Key 或工作区材料。

构建与打包会检查 PT 产品标识、Windows x64 格式、版本、内嵌脚本与界面的字节一致性以及公开源码身份。这些检查不会执行训练，也不等于所有全新 Windows 机器上的在线下载安装已实机通过。不同网络、系统和驱动环境下的首次安装结果仍需用户反馈；提交问题时请附安装日志并移除个人 Key。

维护者先完成独立 PT 启动器构建，在代码树与公开安全提交一致且没有待提交源码的目录执行：

```powershell
.\.venv\Scripts\python.exe tools\package-launcher.py --source-revision codex/public-preview
```

输出默认位于 `release-output/launcher-v0.1.2-preview`。打包脚本只验证并写入新资产，不安装运行时、不启动程序、不发布 GitHub release，也不覆盖已有发行文件。
