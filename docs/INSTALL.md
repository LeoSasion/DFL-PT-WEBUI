# Windows 安装与首次启动

Windows x64 用户可使用独立一键启动器、源码安装包或便携包。当前维护源码为应用 `0.1.2-preview` / 启动器 `0.1.3-preview`；实际可下载的资产版本以发行页为准，历史 `0.1.1-preview` 便携包仍保留原功能集。运行环境固定 Python 3.12.14、PyTorch 2.9.1 + CUDA 12.8、Node.js 24.19.0；驱动由系统提供，CUDA/cuDNN 随 PyTorch wheel 提供。

建议安装到独立目录，例如 `C:\DFL-PT-WEBUI`，保留足够空间容纳下载、解压和 PyTorch 依赖。个人视频、aligned、模型和任务状态在本机项目工作区创建，发行包不提供这些材料。

## 方式一：一键启动器

1. 从[启动器发行页](https://github.com/LeoSasion/DFL-PT-WEBUI/releases)下载 `DFL-PT-WEBUI.Launcher.exe`。可同时下载 `SHA256SUMS.launcher` 核对文件摘要；来源和内嵌资源记录在同页的 provenance JSON。
2. 双击 EXE，选择独立空目录，例如 `C:\DFL-PT-WEBUI`，点击开始安装。无需预装 Git、Python 或 Node；新启动器按内嵌公开源码提交与 ZIP SHA-256 下载固定版本，再获取独立运行时、固定依赖与辅助权重，包括通用 XSeg。
3. 等待检查完成，再点击“启动 WebUI”。安装完成后，启动器保留在项目根目录 `DFL-PT-WEBUI.exe`，以后双击这个文件即可。

启动器要求 Windows 10/11 x64 和 .NET Framework 4.8。缺少 WebView2 Runtime 时会通过 Microsoft 官方签名的安装程序补齐。原版 DFL-WEBUI 和 PT 版不得使用同一安装目录或同一份环境。若默认端口被另一个项目占用，请先在那个项目中停止 WebUI，再返回启动 PT 版；启动器不会结束其他项目的服务。

EXE 自动更新继续关闭。应用与启动器版本分别显示，安装来源和实际源码提交可以核对。历史 `launcher-v0.1.2-preview` 跟随当时的 `main`，新启动器改为固定来源；现有 `0.1.1-preview` 便携分卷和摘要保留不变。用户主动升级时在新版启动器选择“升级应用 / 备份与回退”，先保存并停止所有任务、终端和 WebUI；升级前备份，失败恢复。Git 工作副本使用公开固定提交升级。详细行为和验证范围见[启动器说明](LAUNCHER.md)。

## 方式二：便携包

从 [发行页](https://github.com/LeoSasion/DFL-PT-WEBUI/releases/tag/v0.1.1-preview) 下载便携包的全部 `.zip.partNNN`、对应 `.archive.json`、`.manifest.json`、`SHA256SUMS` 和 `restore-release.ps1`，放在同一目录。小于分卷阈值的版本会提供单个 `.zip`，恢复脚本同样适用。

在下载目录打开 PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\restore-release.ps1 `
  -ArchiveIndex .\DFL-PT-WEBUI-0.1.1-preview-portable.archive.json `
  -Destination .\unpacked
```

脚本验证下载部件、恢复 ZIP，再将文件展开到新目录。保留全部分卷直至恢复完成；已有目标目录会被保护，不要解压覆盖正在使用的项目。解压后进入 `unpacked\DFL-PT-WEBUI`，双击 `启动 WebUI.bat`。便携包携带本地运行时、WebUI 构建、辅助权重和许可；现有启动流程会修正便携 venv 的 base 路径。

直接运行项目内 Python，例如 `.\.venv\Scripts\python.exe`；不要依赖复制自其他目录的 venv 激活脚本。升级时先保存并停止任务，将工作区和模型备份后迁移到新的发行目录。

## 方式三：源码安装

已安装 Git 时，在 PowerShell 执行；也可以从发行页下载源码 ZIP，解压后直接从安装命令开始：

```powershell
git clone https://github.com/LeoSasion/DFL-PT-WEBUI.git
Set-Location DFL-PT-WEBUI
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\install-source.ps1
```

双击根目录 `install-source.bat` 可执行相同安装。入口从固定官方/上游分发下载 Python、Node、FFmpeg 和必要辅助权重并验证 SHA-256，在项目目录内建立 `.venv`，安装固定 PyTorch/requirements 与前端依赖，再构建 WebUI。不会替换正在使用的运行时；已有不匹配材料会报告原因，需在独立目录安装。

源码安装需要访问 GitHub、Node.js、PyTorch wheel index 和 Python/Node 包仓库。如果 node-pty 在当前环境不能使用预构建原生模块，npm 的构建错误可能要求 Microsoft C++ Build Tools；将完整安装日志和 Node/Windows 版本附在反馈中。使用便携包可复用本次附带的原生模块。

专用参数、离线 archive/wheelhouse 放置方法与 `-SkipVisionAssets` 的影响见 [运行环境说明](../launcher/runtime-README.md)。通用 XSeg 推理权重、WF 元数据及来源许可已随两类发行包提供，无需另外下载发行页的同名权重资产；Git 克隆安装会获取相同的固定权重。若主动跳过其他辅助材料，人脸提取、增强和人物分组可能不可用。

## 启动和使用

使用一键安装器完成安装后，双击项目根目录 `DFL-PT-WEBUI.exe` 并点击“启动 WebUI”。便携包及手动源码安装可继续双击 `启动 WebUI.bat`，打开 <http://127.0.0.1:4173>。传统工具入口为 `传统命令菜单.bat`。首次使用按“下一步”向导先创建项目、导入 SRC/DST 素材，再依据工具与数据状态提帧、提脸和整理数据。WebUI 可启动、工具资源齐备、GPU 可训练是不同条件；不会自动训练或上传素材。完整流程见 [README](../README.md#工作流程)。

本版本已有单卡短程开发证据，尚未将全新用户环境、最终画质、真实双卡及摄像头实时表现作为已通过结果。用户实测中的安装或功能问题请通过 [反馈入口](FEEDBACK.md) 提交。
