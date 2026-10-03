# Windows 安装与首次启动

0.1.0-preview 提供源码安装和 Windows x64 便携包。运行环境固定 Python 3.12.14、PyTorch 2.9.1 + CUDA 12.8、Node.js 24.19.0；驱动由系统提供，CUDA/cuDNN 随 PyTorch wheel 提供，不需要安装旧 TensorFlow/CUDA 11 环境。

建议安装到独立目录，例如 `C:\DFL-PT-WEBUI`，保留足够空间容纳下载、解压和 PyTorch 依赖。个人视频、aligned、模型和任务状态在本机项目工作区创建，发行包不提供这些材料。

## 方式一：便携包

从 [发行页](https://github.com/LeoSasion/DFL-PT-WEBUI/releases/tag/v0.1.0-preview) 下载便携包的全部 `.zip.partNNN`、对应 `.archive.json`、`.manifest.json`、`SHA256SUMS` 和 `restore-release.ps1`，放在同一目录。小于分卷阈值的版本会提供单个 `.zip`，恢复脚本同样适用。

在下载目录打开 PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\restore-release.ps1 `
  -ArchiveIndex .\DFL-PT-WEBUI-0.1.0-preview-portable.archive.json `
  -Destination .\unpacked
```

脚本验证下载部件、恢复 ZIP，再将文件展开到新目录。保留全部分卷直至恢复完成；已有目标目录会被保护，不要解压覆盖正在使用的项目。解压后进入 `unpacked\DFL-PT-WEBUI`，双击 `启动 WebUI.bat`。便携包携带本地运行时、WebUI 构建、辅助权重和许可；现有启动流程会修正便携 venv 的 base 路径。

直接运行项目内 Python，例如 `.\.venv\Scripts\python.exe`；不要依赖复制自其他目录的 venv 激活脚本。升级时先保存并停止任务，将工作区和模型备份后迁移到新的发行目录。

## 方式二：源码安装

已安装 Git 时，在 PowerShell 执行；也可以从发行页下载源码 ZIP，解压后直接从安装命令开始：

```powershell
git clone https://github.com/LeoSasion/DFL-PT-WEBUI.git
Set-Location DFL-PT-WEBUI
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\install-source.ps1
```

双击根目录 `install-source.bat` 可执行相同安装。入口从固定官方/上游分发下载 Python、Node、FFmpeg 和必要辅助权重并验证 SHA-256，在项目目录内建立 `.venv`，安装固定 PyTorch/requirements 与前端依赖，再构建 WebUI。不会替换正在使用的运行时；已有不匹配材料会报告原因，需在独立目录安装。

源码安装需要访问 GitHub、Node.js、PyTorch wheel index 和 Python/Node 包仓库。如果 node-pty 在当前环境不能使用预构建原生模块，npm 的构建错误可能要求 Microsoft C++ Build Tools；将完整安装日志和 Node/Windows 版本附在反馈中。使用便携包可复用本次附带的原生模块。

专用参数、离线 archive/wheelhouse 放置方法与 `-SkipVisionAssets` 的影响见 [运行环境说明](../launcher/runtime-README.md)。没有辅助权重时，人脸提取、增强和人物分组可能不可用；通用预训练 XSeg 权重未附带，可以自行训练兼容的 XSeg。

## 启动和使用

安装完成后双击 `启动 WebUI.bat`，打开 <http://127.0.0.1:4173>。传统工具入口为 `传统命令菜单.bat`。首次使用在设置里新建项目，再导入 SRC/DST 素材；完整数据处理、训练和应用流程见 [README](../README.md#工作流程)。

本版本已有单卡短程开发证据，尚未将全新用户环境、最终画质、真实双卡及摄像头实时表现作为已通过结果。用户实测中的安装或功能问题请通过 [反馈入口](FEEDBACK.md) 提交。
