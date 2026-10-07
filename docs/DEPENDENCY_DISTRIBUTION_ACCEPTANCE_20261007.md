# 依赖、项目环境、Qt 与本地发行验收

本次实现生产依赖分层、完整传递依赖哈希锁、模型资源安装与独立离线验收。原项目 `.venv` 没有安装、卸载或升级包；原素材、训练 checkpoint、活动项目注册表及正在运行的服务没有被本项工作修改。维护工具、受控资源和独立测试环境的写入均在项目目录内。

## 固定依赖

| 配置 | 包数 | 用途 |
| --- | ---: | --- |
| production | 61 | ME/XSeg 与默认 YOLO/TUFA 提取、原生遮罩、PySide6 |
| restoration | 78 | 生产配置及 BasicSR/GFPGAN 修复和候选比较依赖 |
| validation | 66 | 生产配置及 ONNX Runtime 导出回读 |
| evaluation | 89 | 修复/验证及 LPIPS、Hydra/iopath 等评测工具 |
| scene | 64 | 生产配置及固定 PySceneDetect 0.7.1 |

`release/python-locks/*-win-cp312.txt` 固定每一个传递包的版本和轮子 SHA256。`release/python-sbom.json` 记录对应 Windows x64/CPython 3.12 的轮子名、大小、tags、依赖和上游元数据许可证。`tools/runtime-dependencies.py` 负责显式维护生成、验证和离线安装；消费者使用 `--require-hashes --only-binary=:all:`，离线安装再加 `--no-index`，不在安装时构建源码包或漂移解析依赖。

完整轮子目录位于 `.launcher-install/locked-runtime-20261007/wheelhouse`。旧纯 Python 源码包由显式维护步骤构建为固定 SHA 的本地轮子；这类可选配置须使用该轮子目录，不能假定 PyPI 有同 SHA 的二进制。轮子目录里的历史多余文件不属于任何选中锁，安装器不会安装它们。

便携包只含 production 的 61 个包；其中没有 BasicSR，不能直接运行 Mamba/Real 修复。现有工作站环境和独立 optional 环境的通过记录，不等于便携 production 已包含 optional 配置。若要在新的源码/便携树启用可选修复和场景算法，先关闭该树的服务和任务，指定已验证轮子目录，依次显式安装 restoration、scene，再安装同一资源包的三组配置（这些命令只作用于所选新树）：

```powershell
.\.venv\Scripts\python.exe tools/runtime-dependencies.py install --directory <verified-wheelhouse> --profile restoration
.\.venv\Scripts\python.exe tools/runtime-dependencies.py install --directory <verified-wheelhouse> --profile scene
.\.venv\Scripts\python.exe tools/prepare-production-vision.py install-pack --profiles production restoration scene --output <verified-resource-pack.zip>
```

上述两个依赖配置的联集为 80 包。独立新 optional venv 实测两个模型完整上下文 FP32 严格加载、16×16 功能推理及三算法真实媒体检测；这是固定模型的离线可用性验收，视觉评分另见人脸修复/Scene 盲评报告。

移除未使用的 h5py，将 ONNX Runtime 移到 validation；onnx 保留，实际 SFace/导出路径仍需要它。VideoEd 已改用固定参数的本地 FFmpeg 后，确认生产源码没有 `import ffmpeg/from ffmpeg`，再删除 ffmpeg-python。`requirements-me.txt` 兼容入口引用唯一根配置，避免 GUI/headless OpenCV 叠装。timm 0.4.12 与 BasicSR 1.4.2 的旧 API 兼容暂保留，未用未经数值等价的新版替换它们。

## 受控模型资源

`release/vision-assets.json` 逐文件固定原始权重、官方源、许可证、来源记录的 SHA/bytes。安装目标为 `_internal/vision_models/production/<id>`，生产推理不下载。资源丢失、改动或安装未完成时失败，不静默回退研究缓存。

| 配置 | 组 ID |
| --- | --- |
| production | yolo11m-face、yolo12l-face、yolo26s-face、tufa、bisenet-celebamaskhq |
| restoration | swinir-psnr、realesrgan-x4plus、mambairv2 |
| scene | transnetv2 |

SwinIR 只保留现有源帧链；aligned/ME 合成的人脸增强按本轮用户指定使用 MambaIRv2/Real-ESRGAN。FaceEnhancer.npy 不在安装或发行清单。原缓存作为来源只读，不删除或改写。

`tools/prepare-production-vision.py` 提供 install/install-pack/verify/pack/verify-pack：路径逃逸、重复 ZIP 项、大小/SHA 改变均拒绝；资源包绑定完整固定 manifest；每组先在临时目录写齐，再原子发布。已存在且不匹配的组保留并失败。公开准入与本地安装权限分开记录，未擅改项目 GPL 或把源码许可证推断为独立权重授权。YOLO AGPL、TUFA GPL-2.0 及各模型来源/权重条款均留存供发行审查；此次没有公开发布。

完整本地资源包 `.launcher-install/locked-runtime-20261007/vision-all-upgrades-local-review-v2.zip` 含九组、85 个文件，SHA256 `1289f50828845aacef5d8052d25a192d423de99b6638cd925dc115f5f2d3daa3`，`publicDistribution=false`。同一份包可在新项目树内显式离线安装：

```powershell
python tools/prepare-production-vision.py install-pack --profiles production restoration scene --output <verified-resource-pack.zip>
```

generic XSeg 仍采用 `release/generic-xseg.json` 的原始发行权重、元数据、许可证和来源摘要；源码及便携包都会逐项锁定、验证并包含，不使用随机或 QA checkpoint 替代。

## 项目与 Qt

`launcher/resolve-active-project.py`、`_internal/setenv.bat` 和 C# DflEnvironment 使用同一注册表的已登记 activeId 解析工作区。CMD 一次返回 id/绝对路径；未登记或损坏的选择失败。旧菜单显示活动项目及完整路径，并指出原生 WebUI 查看/相似图复核入口。全部兼容 BAT 在 setenv 失败时退出，不继续使用上一项目。

独立两项目 CMD 证据在 `.validation/legacy-two-project-context-20261007.json`，仅修改独立夹具注册表并恢复；原活动项目未切换。Launcher Vite 已与 WebUI 对齐到 6.4.3，Windows PowerShell 5.1 真实构建 .NET 4.8 C# 并更新本地 `launcher/bin`，`uiBuildPerformed=true`。打包会核对 native EXE 的 build record/SHA 和 Git 快照里的 C#/UI 源码，防止新源码配旧 launcher。

Windows PowerShell 5.1 的 advanced script 在 param 默认表达式中不能稳定使用 `$PSScriptRoot`；install-source/setup-runtime 改为正文解析默认项目路径。独立新树实测省略 `-ProjectRoot` 的两个原生入口，并保留显式项目参数。native provenance 要求真实原始构建 bytes/SHA 及 Git blob 一致，仅允许 `.gitattributes` 的 LF/CRLF 文本规范化，包含混合换行和内容篡改回归。

Qt 只支持 PySide6。`QImage_to_np` 处理真实 memoryview 和 bytesPerLine，包含非四字节对齐宽度。`tools/xseg-qt-acceptance.py` 创建真实编辑器/loader worker，用 Qt 鼠标事件画四点多边形、保存并重载 DFL JPEG，验证图像像素不变和 259 宽 QImage 往返。使用 offscreen 平台，记录中明确注明；这不是人工可见窗口操作。源码干净环境和真正解包的便携环境均保留独立验收记录。

## 真正的安装与归档验收

独立源码环境在 `.launcher-install/locked-runtime-20261007/` 内创建全新 venv，以项目本地 Python/Node/FFmpeg 和固定轮子/资源包离线准备。WebUI 使用固定 pnpm lock 与本地已有包缓存进行真实离线安装/构建。源码 archive 包含实现与固定 generic XSeg；离线源码准备的本地 runtimes/轮子/Node 缓存和模型包是明确提供的输入。便携 archive 包含所选生产运行时，单独模型包显式安装。

实际功能包括默认 YOLO26s+TUFA CPU 提取，68 点嵌入及原生 98 点 sidecar 精确一致；BiSeNet 草稿的嵌入遮罩回读及 JPEG 解码像素一致；修复配置的真实 SwinIR/Real/Mamba 严格权重加载与有限 FP32 推理；PySide6 真实打开、绘制、保存。模型功能通过不等于视觉质量评分，后者另见各盲评报告。此依赖工作没有进行训练。

首次完整 portable 归档 SHA、分卷和 57,879 个 payload 文件虽全部通过，真正提取/Qt 仍失败：旧 `**/tests/**` 过滤删掉了 NumPy 2.2.6 的 `_core/tests/_natype.py`，它被生产路径 SciPy→numpy.testing 引用。原失败保留在 `.validation/PORTABLE_ACCEPTANCE_20261007.json`，没有补改归档或重新标签为通过。发布器现仅精确保留这一个由选中轮子拥有的运行文件，仍排除测试套件。

修订快照重新构建、全量校验并在新的空目录真正解包；提取/遮罩/Qt 均通过，记录为 `.validation/PORTABLE_FIXED_ACCEPTANCE_20261007.json`。这些早期快照在 Mamba/scene 最终接入前，不能冒充整轮最终发行。

整轮稳定后的源码、便携、分卷及资源包准确 SHA/大小/完整 payload 验证和实际干净验收统一由外部记录 `.validation/FINAL_DISTRIBUTION_ACCEPTANCE_20261007.json` 与对应本地 artifact manifests 给出，避免把归档自身 SHA 写回归档内形成循环。原失败与修订证据始终保留。最终包只在本机构建、核验，没有 push 或公开 release。
