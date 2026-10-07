# DFL-PT-WEBUI 运行环境与源码安装

源码安装支持 Windows 10/11 x64。克隆或解压本项目源码后，在项目根目录双击 `install-source.bat`，或运行：

```powershell
.\install-source.bat
# PowerShell 入口（从任意目录使用时可指定 -ProjectRoot）：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\launcher\install-source.ps1
```

可从发行页单独下载 `DFL-PT-WEBUI.Launcher.exe`，选择独立空目录并点击“开始安装”。启动器获取本项目官方源码，再调用上述固定版本安装入口；不需要预装 Git。安装完成后在项目根目录保存 `DFL-PT-WEBUI.exe`，也可继续使用 `启动 WebUI.bat`。安装入口会准备独立 Python base、项目 `.venv`、Node、FFmpeg、提取/增强/人物分组辅助权重和通用 XSeg 推理权重，以及锁定的 WebUI 依赖和首次前端构建。安装本身不启动 WebUI，不训练、合成或下载用户的视频、数据集和 ME 模型。系统需有适合 CUDA 12.8 的 NVIDIA 驱动；CUDA/cuDNN DLL 由 PyTorch wheel 提供，无需另装 CUDA Toolkit。

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

2026-10-07 起安装使用 `release/python-locks/*-win-cp312.txt` 的全传递 SHA-256 锁，见 `release/python-sbom.json`。默认 production 61 包；restoration 78、validation 66、evaluation 89、scene 64 包分别用于修复、ONNX 数值校验、比较工具和场景检测。production 不带 h5py、onnxruntime、BasicSR/GFPGAN、tb-nightly、ffmpeg-python 或 pytest；桌面环境只装 GUI OpenCV wheel，不能叠装 headless。旧 timm 0.4.12 与 BasicSR 1.4.2 在完成同权重数值等价前保留，不把库更新当成质量提升。

YOLO/TUFA/BiSeNet 等安装到 `_internal/vision_models/production/<id>`，源码、权重、许可证、原始来源逐件锁定，多个工作区共用；研究 cache 仅作旧兼容读源，不依赖 active 项目。新项目须提供独立核验资源包：`-ResourcePackPath <vision-local-review.zip>`；可选 `-RuntimeProfile restoration` 准备 MambaIRv2/Real-ESRGAN 人脸链及既有 SwinIR 源帧链。Aligned 与 ME 预测脸只使用 MambaIRv2/Real-ESRGAN，旧 FaceEnhancer 已移除。在线安装不会使推理自动下载。TUFA 的官方 Drive 权重暂无固定无交互下载 URL，缺资源时明确停止，不能静默回退 FAN。离线命令示例：

```powershell
.\launcher\install-source.ps1 -NoNetwork -WheelhousePath <wheelhouse> -ResourcePackPath <resource-pack>
```

`tools/runtime-dependencies.py generate --directory <maintainer-cache> --download` 是显式维护锁定入口；消费者只安装锁定 wheel。`tools/prepare-production-vision.py pack` 生成并核验独立本地复核包；`--public-distribution` 必须所有资源都有独立准入，未证实权重再分发许可的组会被拒绝。上游代码许可证不自动等同权重许可证，不改变本项目 GPL。generic XSeg 原权重及许可仍在 source/portable 两类主包中固定携带。

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

`-WheelhousePath` 一旦指定，Python 包安装仅使用该目录，即使没有 `-NoNetwork` 也不访问包索引。仍需准备 WebUI 依赖与辅助权重。视觉权重可以预先安装到它们的目标路径，或放到 `.launcher-install/vision` 缓存；缓存名为 `S3FD.npy`、`2DFAN.npy`、`3DFAN.npy`、`sface-2021dec.onnx`、`XSeg_256.pth`，仍执行散列校验。

源码包和便携包已经含通用 XSeg 推理权重、WF 元数据及来源许可，不依赖首次联网下载。仅 Git 克隆需要安装入口从本项目固定发行资产获取相同权重，并校验 `release/generic-xseg.json` 中的 SHA。它没有原训练状态或优化器，项目专用 XSeg 仍需标注和训练。

如果已有 ME 训练材料，并明确暂不使用提取、增强、人物分组和通用 XSeg，可选择 `-SkipVisionAssets`。它不下载辅助权重，也不保证这些工具可用。`-SkipWebuiBuild` 只跳过首次构建；正常启动所需的 WebUI 依赖仍须就绪。

`-SkipWebuiPreparation` 是启动器修复流程的内部参数，将 WebUI 依赖修复与强制构建交给启动器处理；Python、Node、FFmpeg 和辅助权重仍由安装入口准备、验证。

## WebUI 原生依赖

WebUI 依赖由项目内 Node/Corepack 执行 `pnpm install --frozen-lockfile`，保持仓库的 `pnpm-workspace.yaml` 构建许可。固定的 `node-pty` 1.1.0 发布包包含 Windows x64 预编译模块，脚本优先复用它，并在安装后检查模块是否能加载。

如果预编译模块无法使用，`node-pty` 的安装脚本会回退到 `node-gyp` 编译。此时需要自行安装 Microsoft Visual Studio 2022 Build Tools 的“使用 C++ 的桌面开发”、MSVC v143、Windows SDK，必要时包括 Spectre 库；Python 由本项目的 Python 3.12 base 提供。安装入口不会自动安装全局编译工具，也不会自动降低 Node 或替换锁文件。编译失败时参考 [node-pty 上游说明](https://github.com/microsoft/node-pty#dependencies)和 [node-gyp Windows 安装说明](https://github.com/nodejs/node-gyp#on-windows)。

## 启动器本地检查

完整便携发布包须同时包含 Python base 和 venv。解包或移动后先使用启动器、WebUI 入口或下方 bootstrap；它们将 `pyvenv.cfg` 重新指向当前项目 Python base。完成初始化后再调用 `.venv/Scripts/python.exe` 或安装可选依赖；不能把移动后尚未初始化的 venv 当作直接可运行环境。

```powershell
# 已有便携环境的本地检查与迁移路径修复：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\launcher\bootstrap.ps1 -ProjectRoot . -NoNetwork
```

`setup-runtime.ps1 -InstallDependencies` 是原有显式包修复接口，需要先有 `_internal/python_base`；新用户可使用启动器或上面的 `install-source.bat` 完成独立环境准备。启动器使用官方源码 ZIP 进行首次安装；Git 更新和 EXE 自动更新仍关闭，运行时安装入口不配置 Git 远端或更新通道。

本轮已实测官方 GitHub 源码获取，完成 Windows PowerShell 5.1/7 的离线安装路由与保护测试、已有环境的 `-NoNetwork` 验证，以及完整 UI/native 构建和打包校验；尚未在全新的 Windows 环境下载整套多 GB 运行时并完成启动、训练验收。

## 最佳训练人脸

在「工具 → 质量方案 → 最佳训练人脸」确认同人参考，以每批最多 500 张累计分析，再生成全量代表性训练子集。推荐数量是上限，多样性增益不足时提前停止；发布独立 `selected/`、`review/`、`rejected/` 与回执，保留源 JPEG、DFL 元数据和辅助记录。更改参考后必须重新生成才能发布；训练按钮只填写有效 `selected` 路径，不启动训练。使用说明见 `docs/BEST_TRAINING_FACESET_WEBUI.md`。

生产仍使用以 Tenengrad 为主的既有综合质量分。唯一 Efficient-FIQA 对照未达到预先固定的替换收益门槛，资源不纳入生产包或生产 Python 依赖；它可通过显式评测工具准备，不会在推理时自动下载。该功能没有改变 ME 训练桥接，也没有新增训练架构。
