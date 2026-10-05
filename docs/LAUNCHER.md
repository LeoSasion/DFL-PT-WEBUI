# Windows 一键启动器

从[DFL-PT-WEBUI 发行页](https://github.com/LeoSasion/DFL-PT-WEBUI/releases)下载启动器，双击后选择独立空目录并开始安装。当前源码中的应用版本为 `0.1.2-preview`，启动器版本为 `0.1.3-preview`，二者独立显示。首次安装无需预装 Git、Python 或 Node；安装完成后，启动器保留在项目根目录 `DFL-PT-WEBUI.exe`。以后双击它，点击“启动 WebUI”即可使用。源码版本不代表同版本 EXE 或便携包已经发布，请以发行页资产和 provenance 为准。

启动器也可以选择已有的完整 PT 项目，检查本地运行环境并启动 WebUI。原版 DFL-WEBUI 与 PT 版必须使用不同目录，不共享 Python 环境、启动器配置或工作区。两个项目默认使用相同端口；切换时先在另一个项目中停止 WebUI。启动器不会结束其他项目的服务。

## 首次安装

新启动器从内嵌 `release/source-pin.json` 的官方公开固定提交下载源码，不跟随移动的 `main`。下载内容必须匹配内嵌 SHA-256、固定 ZIP 根目录和应用版本；未固定来源的开发构建拒绝在线安装。源码先展开到暂存区，检查 PT 产品标识和 ME/PyTorch 入口后才发布到所选目录，再调用 `launcher/install-source.ps1` 准备 Python 3.12、PyTorch 2.9.1 CUDA 12.8、Node 24.19.0、FFmpeg、锁定的 WebUI 依赖、辅助权重和通用 XSeg 推理权重，并构建 WebUI。`release/installation.json` 记录实际安装来源、源码提交和下载摘要；WebUI 的源码清单还能检验文件内容是否完整或已被修改。

请选择独立空目录，例如 `C:\DFL-PT-WEBUI`。不要选择原版项目目录，不要覆盖正在使用的环境。下载与解压需要额外磁盘空间；首次安装需要访问 GitHub、Node.js、PyTorch 和 Python/Node 包仓库。已有运行时或依赖会按照源码安装入口的规则检查和复用，不匹配时给出原因。离线安装参数、固定来源与散列、原生依赖要求见[运行环境说明](../launcher/runtime-README.md)。

运行条件是 Windows 10/11 x64、.NET Framework 4.8。若缺少 WebView2 Runtime，启动器会下载并验证 Microsoft 官方签名的安装程序。GPU 训练另外需要支持 CUDA 12.8 的 NVIDIA 驱动；CUDA/cuDNN 来自 PyTorch wheel。

安装失败时保留日志和已完成的依赖下载，处理界面指出的网络、空间或环境问题后重试。请在 PT 安装目录内补齐环境。安装日志及复用规则见[运行环境说明](../launcher/runtime-README.md#已有环境与失败重试)。

## 安装后的使用与更新

安装完成后使用项目根目录 `DFL-PT-WEBUI.exe`，原有 `启动 WebUI.bat` 和 `传统命令菜单.bat` 仍可使用。启动器的检查和安装本身不运行训练，不下载用户素材。首次进入 WebUI 后再建立自己的项目并导入素材。

启动器设置单独保存在 `%LocalAppData%/DFL-PT-WEBUI/Launcher`，素材和模型保存在所选 PT 项目的 `workspace` 或 `workspaces`。原版启动器配置不会迁入这里。

EXE 自动更新继续关闭。用户下载并运行新版启动器后，点击“升级应用 / 备份与回退”，预览升级说明，再点击“备份并升级”。目标是该启动器内嵌来源记录对应的固定应用版本。运行中的 WebUI、传统终端和项目 Node/Python/FFmpeg 任务会阻止修复或升级；请先保存并结束任务。

升级前保留所有待替换源码、当前界面构建和 Node 依赖。完成后备份留在 `.launcher-install/upgrade-backup-*`；若应用或构建检查失败，自动恢复原源码、界面构建与依赖；意外中断时使用“恢复未完成的升级”。备份损坏时不会先移动当前可用构建，界面会要求保留缓存。工作区、模型、API 配置、环境变量文件和运行时不会作为升级目标。Git 工作副本不使用 ZIP 覆盖，请自行选择已审查的公开固定提交。

历史 `launcher-v0.1.2-preview` 安装时跟随当时的官方 `main`，不能推断为某个固定应用版本。历史 `v0.1.1-preview` 便携包仍对应原功能集，原资产和摘要保留；没有重打新的便携包时不能把它标记为已更新。以版本栏、安装来源、源码提交、资产摘要和各自的 provenance 判断实际内容。

安装或启动失败后点击“准备反馈信息”，预览并复制 PT 产品、版本、安装来源、源码摘要和失败步骤，再自行打开 Issue 模板。没有可用 Node 时提供同一模板的离线摘要。不会自动上传日志、媒体、模型或凭据，摘要不包含私有绝对路径。

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

输出默认位于 `release-output/launcher-v0.1.3-preview`。先公开审查后的应用源码提交 P，获取其官方 codeload ZIP 并固定 SHA-256，再将来源记录接到公开提交 Q 并构建启动器。启动器 provenance 记录 Q；实际下载的应用安装记录保持 P。源码/便携构建则记录该归档实际对应的公开提交和代码树。打包脚本只验证并写入新资产，不安装运行时、不启动程序、不发布 GitHub release，也不覆盖已有发行文件。
