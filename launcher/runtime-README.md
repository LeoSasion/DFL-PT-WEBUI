# DFL-PT-WEBUI 运行环境与源码安装

源码安装支持 Windows 10/11 x64。克隆或解压本项目源码后，在项目根目录双击 `install-source.bat`，或运行：

```powershell
.\install-source.bat
# PowerShell 入口（从任意目录使用时可指定 -ProjectRoot）：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\launcher\install-source.ps1
```

安装完成后使用根目录的 `启动 WebUI.bat`。安装入口会准备独立 Python base、项目 `.venv`、Node、FFmpeg、提取/增强/人物分组辅助权重和通用 XSeg 推理权重，以及锁定的 WebUI 依赖和首次前端构建。安装本身不启动 WebUI，不训练、合成或下载用户的视频、数据集和 ME 模型。系统需有适合 CUDA 12.8 的 NVIDIA 驱动；CUDA/cuDNN DLL 由 PyTorch wheel 提供，无需另装 CUDA Toolkit。

## 运行时与来源

| 组件 | 新安装固定版本 / 来源 | 项目位置 |
| --- | --- | --- |
| Python | CPython 3.12.14，Astral [python-build-standalone 20260814](https://github.com/astral-sh/python-build-standalone/releases/tag/20260814) 的 Windows x64 `install_only_stripped` 构建 | `_internal/python_base` |
| PyTorch | `torch==2.9.1+cu128`，[PyTorch 官方 cu128 wheel 索引](https://download.pytorch.org/whl/cu128) | `.venv` |
| Python 工具依赖 | 根 `requirements.txt` 固定版本，[PyPI 官方索引](https://pypi.org/simple) | `.venv` |
| Node | 24.19.0，[Node 官方发行目录](https://nodejs.org/dist/v24.19.0/) | `_internal/node/bin` |
| pnpm | 11.19.0，由项目内 Node 的 Corepack 获取，[npm 官方 registry](https://registry.npmjs.org/) | `.launcher-install/source/corepack` |
| FFmpeg | 9.0.1 essentials，[GyanD 固定发行版](https://github.com/GyanD/codexffmpeg/releases/tag/9.0.1)，该 Windows 构建方由 [FFmpeg 下载页](https://ffmpeg.org/download.html)列出 | `_internal/ffmpeg` |
| 辅助权重 | 固定上游提交及 SHA-256，见 `tools/prepare-vision-runtime.ps1` | `facelib/*.npy`、`_internal/vision_models`、`_internal/model_generic_xseg` |

Python 便携构建包含标准库、`venv` 和 `ensurepip`，无需注册到系统或修改全局 PATH。Python base 已存在时接受有效的独立 Python 3.12 x64；安装入口不会把现有 3.12.x 强制替换为 3.12.14。所有运行时目录和用户工作区均被 Git 忽略。

三个运行时压缩包在解压前必须通过脚本内固定的 SHA-256 校验：

| 文件 | SHA-256 |
| --- | --- |
| `cpython-3.12.14+20260814-x86_64-pc-windows-msvc-install_only_stripped.tar.gz` | `89f18f6932917163b74339ebcec2645c8e47ae7f1c5f2ac37f2b4f4cf3beb647` |
| `node-v24.19.0-win-x64.zip` | `57f71ab3652e797d84acddc79c81cc9ff1c6ddb2a1974cdb83f00fee9bff4c73` |
| `ffmpeg-9.0.1-essentials_build.zip` | `fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9` |

Python 和 FFmpeg 散列来自相应 GitHub 上游发行资产的 SHA-256 digest，Node 散列来自官方 [SHASUMS256.txt](https://nodejs.org/dist/v24.19.0/SHASUMS256.txt)。辅助权重逐文件校验；WebUI 使用 `pnpm-lock.yaml` 和包完整性校验。下载只使用脚本内固定的 HTTPS 官方/上游 URL，不使用旧项目 Git 远端、TF 环境或旧版 CUDA 下载链路。

## 已有环境与失败重试

运行前先停止 WebUI、Python 任务和项目内 FFmpeg 进程。安装入口检测本项目的运行中进程并停止安装，且通过独占锁防止两个安装入口同时运行。

已有 Python base、`.venv`、Node、FFmpeg 和 `webui/node_modules` 会直接验证和复用。该入口不覆盖已有目录，不改已有 `.venv/pyvenv.cfg`，不向已有 `.venv` 或 `node_modules` 再装包。现成 Python 环境须满足固定的 torch/CUDA 与根 requirements 版本；已有 Node 须为 24.19.0。若现成环境不完整、版本不符或辅助权重散列不符，脚本会给出路径并停止。请先保留该目录的备份，再在新的源码目录安装；不要边运行边修复环境。

新运行时先在项目缓存内解压，校验后移入不存在的目标目录。下载的压缩包保留在 `.launcher-install/source/archives` 供重试使用；失败时只清理本次创建且尚未安装完整的 venv、依赖或构建目录，已安装完整的环境继续保留。源码安装不会写入 `workspace`/`workspaces`。

## 离线使用

已经准备好的完整环境可以离线检查与复用：

```powershell
.\install-source.bat -NoNetwork
```

`-NoNetwork` 禁止脚本下载运行时、权重、Python 包或 Corepack 包。缺少依赖时会停止并说明所缺内容。离线源码环境须已有完整的 `webui/node_modules`；安装入口不会在离线模式运行可能下载编译输入的 npm 生命周期脚本。

需要从本地材料补齐运行时，可将上表文件放到 `.launcher-install/source/archives`，或指定只读的材料目录。新建 `.venv` 的离线 wheelhouse 必须包含 Windows x64 / CPython 3.12 的 `torch==2.9.1+cu128`、根 requirements 及其所有依赖：

```powershell
.\install-source.bat -NoNetwork -ArchiveDirectory D:\DFL-offline\runtimes -WheelhousePath D:\DFL-offline\wheels
```

`-WheelhousePath` 一旦指定，Python 包安装仅使用该目录，即使没有 `-NoNetwork` 也不访问包索引。仍需准备 WebUI 依赖与辅助权重。视觉权重可以预先安装到它们的目标路径，或放到 `.launcher-install/vision` 缓存；缓存名为 `S3FD.npy`、`2DFAN.npy`、`3DFAN.npy`、`FaceEnhancer.npy`、`sface-2021dec.onnx`、`XSeg_256.pth`，仍执行散列校验。

源码包和便携包已经含通用 XSeg 推理权重、WF 元数据及来源许可，不依赖首次联网下载。仅 Git 克隆需要安装入口从本项目固定发行资产获取相同权重，并校验 `release/generic-xseg.json` 中的 SHA。它没有原训练状态或优化器，项目专用 XSeg 仍需标注和训练。

如果已有 ME 训练材料，并明确暂不使用提取、增强、人物分组和通用 XSeg，可选择 `-SkipVisionAssets`。它不下载辅助权重，也不保证这些工具可用。`-SkipWebuiBuild` 只跳过首次构建；正常启动所需的 WebUI 依赖仍须就绪。

## WebUI 原生依赖

WebUI 依赖由项目内 Node/Corepack 执行 `pnpm install --frozen-lockfile`，保持仓库的 `pnpm-workspace.yaml` 构建许可。固定的 `node-pty` 1.1.0 发布包包含 Windows x64 预编译模块，脚本优先复用它，并在安装后检查模块是否能加载。

如果预编译模块无法使用，`node-pty` 的安装脚本会回退到 `node-gyp` 编译。此时需要自行安装 Microsoft Visual Studio 2022 Build Tools 的“使用 C++ 的桌面开发”、MSVC v143、Windows SDK，必要时包括 Spectre 库；Python 由本项目的 Python 3.12 base 提供。安装入口不会自动安装全局编译工具，也不会自动降低 Node 或替换锁文件。编译失败时参考 [node-pty 上游说明](https://github.com/microsoft/node-pty#dependencies)和 [node-gyp Windows 安装说明](https://github.com/nodejs/node-gyp#on-windows)。

## 启动器本地检查

完整便携发布包须同时包含 Python base 和 venv。Windows venv 会记录 base 的绝对路径；项目移动后，原有 `setup-runtime.ps1` 的本地检查会将 `pyvenv.cfg` 重新指向当前项目 Python base，WebUI 管理器在启动前调用它。

```powershell
# 已有便携环境的本地检查与迁移路径修复：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\launcher\bootstrap.ps1 -ProjectRoot . -NoNetwork
```

`setup-runtime.ps1 -InstallDependencies` 是原有显式包修复接口，需要先有 `_internal/python_base`；新用户使用上面的 `install-source.bat` 完成独立环境准备。启动器在线项目克隆、Git 更新和自更新继续受本项目的发布策略控制；运行时安装入口不配置 Git 远端或更新通道。

本轮交付仅对安装脚本做静态语法与来源检查，没有执行新安装、下载运行时、启动程序或功能验收。具体新机安装和功能使用结果需由开源用户实测反馈。
