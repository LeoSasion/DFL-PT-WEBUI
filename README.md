# DFL-PT-WEBUI

Windows 本地工作台：**ME PyTorch 训练 + WebUI + 数据处理、XSeg、合成和视频工具**。

当前源码的应用版本为 **0.1.2-preview**，启动器版本为 **0.1.3-preview**。换脸训练使用 ME/PyTorch，XSeg 是辅助遮罩模型；旧 ME 网络权重可以通过显式导入命令迁移。运行环境与项目材料保持独立。源码版本、EXE 版本与已发布便携包分别记录，以发行资产及 provenance 判断实际内容。

- [历史 0.1.1-preview 便携包](https://github.com/LeoSasion/DFL-PT-WEBUI/releases/tag/v0.1.1-preview)：保持原功能集及原摘要，随包提供通用 XSeg 推理权重。
- [启动器及源码发行资产](https://github.com/LeoSasion/DFL-PT-WEBUI/releases)：无需预装 Git、Python 或 Node；新启动器按固定公开提交及 SHA-256 安装。历史 launcher-v0.1.2-preview 跟随安装时的 main；详见[启动器说明](docs/LAUNCHER.md)。
- [完整安装步骤](docs/INSTALL.md)、[版本说明](docs/RELEASE_NOTES.md)、[已知限制](docs/KNOWN_ISSUES.md)。
- [反馈问题](https://github.com/LeoSasion/DFL-PT-WEBUI/issues/new/choose)；反馈需要的信息见 [FEEDBACK.md](docs/FEEDBACK.md)。

## 启动

首次使用可双击下载的 **`DFL-PT-WEBUI.Launcher.exe`**，选择独立空目录并开始安装。完成后启动器保留在项目根目录 **`DFL-PT-WEBUI.exe`**，以后双击它，再点击“启动 WebUI”，打开 <http://127.0.0.1:4173>。原有 **`启动 WebUI.bat`** 和 **`传统命令菜单.bat`** 仍可使用。

便携包带独立 Python 3.12.14、PyTorch 2.9.1 + CUDA 12.8、Node.js 24.19.0、FFmpeg 和必要辅助权重。源码安装可在仓库根目录运行 `install-source.bat`，自动准备这些材料、安装依赖并构建 WebUI。需要 Windows x64；GPU 训练需要支持 CUDA 12.8 的 NVIDIA 驱动。Python base 在 `_internal/python_base`，venv 在 `.venv`。

运行检查：

```powershell
Set-Location C:\DFL-PT-WEBUI
.\.venv\Scripts\python.exe tools\smoke_test.py
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\bootstrap.ps1 -ProjectRoot C:\DFL-PT-WEBUI -NoNetwork
```

新启动器按 `release/source-pin.json` 从本项目官方 GitHub 固定提交下载源码并校验 ZIP SHA-256，再调用 PT 安装入口准备独立运行环境和通用 XSeg 权重。源码 Git 不提交运行时、依赖包、个人素材和模型；克隆后先执行安装入口。便携包不带个人素材、训练检查点或本机任务状态。源码安装支持准备离线材料，详见 [运行环境说明](launcher/runtime-README.md)。自动更新关闭；用户在新版启动器中主动选择升级，运行任务阻止维护，升级失败恢复保留的源码、构建与依赖备份。

当前开发证据覆盖单卡、小参数的保存、停止、恢复和初步收敛趋势。正式案例画质、真实双卡和新用户不同机器上的完整流程等待实测反馈，详见 [短程验收](docs/ME_SHORT_RUN_ACCEPTANCE.md)。

## 工作流程

本轮流程与交互改进已实现，并完成隔离项目的功能验收。导航进一步按预处理、训练、后处理分组；顶部只展示当前组的相关流程，已有 PAK / aligned 可直接进入训练。操作步骤见 [用户指南](docs/USER_GUIDE.md)，此前实现范围见 [UX 开发计划](docs/UX_DEVELOPMENT_PLAN_20261007.md)，此前实际通过、跳过和未验证范围见 [本轮验收记录](docs/UX_ITERATION_ACCEPTANCE_20261007.md)。

1920×1080 完整实跑已验证新建小模型、保存、续训、安全停止、预测、100帧合成和导出；训练全段保守计时6分39秒。合并前审查修复、回归结果与画质限制见 [最新审查与 E2E 记录](docs/PR_REVIEW_E2E_20261008.md)。

1. 从页面标题处的项目菜单创建或切换项目，在“素材管理”导入 SRC 和 DST 视频。
2. 配置提取视频帧，核对输入与输出后预检并启动。
3. 提取带 DFL 元数据的 aligned 人脸，在 SRC/DST 原生查看器中复核。
4. 按需使用人物分组、质量审查、相似图片、姿态图谱和隔离恢复工具。可在“工具 → 质量方案 → 最佳训练人脸”建立选集、人工复核并另存新版本；原始 aligned 仍是合法输入，不强制筛选。选集草稿按项目与侧别保存，后台结果可从终端上方结果栏找回。
5. 按需手工修正或自动应用 XSeg 遮罩；“查看遮罩用途”说明三种方式，场景专用辅助训练仍可单独配置。
6. 在“模型训练”选择 ME，核对项目采用的 SRC/DST 人脸集，并明确填入本次配置。两侧选择分别持久化，不自动改写检查点。新模型默认 128 / batch 4；续训先查看继承摘要、设备和运行计划，完整参数可展开。
7. 在“总览”查看预览、损失和保存状态，按需保存、备份和安全停止。质量诊断是可选趋势检查，需要两个可比较快照；已有模型仍可经预检进入合成。
8. 在“模型应用”配置 ME 合成。可先指定 1–20 帧生成独立小样，检查原帧、结果与遮罩后，把相同参数填入全片合成并重新预检。小样不替换正式合成序列。
9. 在“视频导出”先查看已有成片，再检查正式合成帧、遮罩和音轨，配置母版/播放版或其他格式；也可单独导出 ONNX 格式 DFM。

本轮已实测两帧 CPU 小样及相同参数的正式合成预检，未执行全片合成或新成片导出，也未启动新训练。实际浏览器 200% 缩放未验证，CSS 小视口只作内容容量模拟。独立发行验收随 v7 外部报告记录（该文档不自证包 SHA）。

项目素材分别保存在 `workspace` 或 `workspaces/<项目标识>`。ME 模型保存为 `model/<模型名>/me.pt`，并带有 `metadata.json`、预览和损失记录。检查点包含网络、判别器、优化器、AMP scaler、迭代与更新计数、随机数、采样状态和困难样本回放。

ME 训练由独立于 Web 服务的监督进程承载；Web 服务重启后会依据持久状态重连训练、日志和控制，并继续占用原模型/GPU 锁。“训练心跳”区分正常、检测中和疑似停滞，疑似停滞不会自动杀死训练。自动备份默认每 1000 次迭代保存一次并保留最近 3 代，手动备份不受这项轮换影响；备份包含检查点、元数据和现有损失历史。在“工作区”的模型恢复点中可查看和选择校验过的备份，恢复前会保留当前版本，运行中的任务不可恢复模型。

ME 向导提供“默认训练”“RG + FP16”“预训练”“后期微调”快速配置，以及网络结构、优化与训练、显存与精度、重建与风格损失、GAN 与 TrueFace、采样与增强、困难样本回放的完整参数分组。参数仍可逐项调整；FP16 训练需要 CUDA，TrueFace 仅适用于 DF。预训练需要指定专用人脸集，SRC 和 DST 共用该数据，GAN、TrueFace 和风格损失在此阶段不生效。

选择已有模型会恢复检查点配置。调整续训参数必须启用“允许续训配置变更”，向导会显示差异；网络结构保持固定。切换优化器类型或已启用 GAN 的判别器形状需“重置优化器”，更换数据集或进入/退出预训练阶段需“重置采样状态”。预训练转微调可继续同一模型，启用配置变更、选择“后期微调”，再重置采样状态并指定 SRC/DST；新模型可通过“从现有模型初始化”只复制网络，建立新的训练进度。

“数据与运行计划”支持当前项目工作区内的自定义 SRC/DST 目录、PAK 或 ZIP 文件。目标迭代默认只估算进度；启用自动停止后才作为总迭代上限。“仅检查采样”可在 CPU 上输出增强与遮罩检查图，不执行训练或覆盖检查点；CPU 检查时关闭 FP16。姿态评测现支持目录、PAK、ZIP、自定义及预训练人脸集，以数据集指纹和包内成员名固定样本；只有默认的非打包 SRC/DST aligned 目录会跳转到对应姿态图谱。

## 配套工具

0.1.2-preview 源码包含“工具 → 图像工具”的奇智 API 接入：本机 Key 配置、文生图/图像编辑/参考图合成、后台进度与结果保存，以及原图/结果同步对比、版本选择和另存。现有 0.1.1-preview 便携包尚不含这项功能，使用与能力边界见 [图像服务说明](docs/IMAGE_SERVICE.md)。

当前 `main` 提取默认采用 YOLO26s-face、TUFA68 对齐和独立 TUFA98 审阅，保留 S3FD、FAN 及 HEAD 兼容路径。旧 FaceEnhancer 链已由 MambaIRv2 / Real-ESRGAN 人脸修复替代，可选依赖与固定资源需显式准备；ME 预测脸增强强度默认0。视频提帧/封装、错脸筛选、排序、姿态筛选、PAK/ZIP、重设尺寸、DFL 元数据和 XSeg，以及 Web 人物选择、审核、修复、标注、相似度、隔离恢复、诊断、日志和任务管理继续支持。当前源码选择与有限验证见 [全流程升级](docs/ALL_PIPELINE_UPGRADE_20261007.md)，工具映射见 [迁移说明](MIGRATION.md)；历史发行包保留其原功能集。

ME 训练直接读取带 DFL 元数据的 aligned JPG/JPEG，以及 PAK/ZIP 打包人脸和人物子目录。通用 XSeg 推理权重随源码包和便携包提供，位于 `_internal/model_generic_xseg`，可直接使用“内置遮罩应用”；Git 克隆安装会按固定散列获取相同文件。权重来自官方 DFL 通用模型，保留 WF 人脸类型并转换全部张量供 PyTorch 加载；它不是项目 XSeg 的完整训练状态，专用遮罩仍可在项目内标注和训练。来源、许可证和转换记录见 [来源与许可](docs/SOURCES.md)。

aligned 预览与清洗默认使用 WebUI 的 SRC/DST 人脸浏览器，可检查定位点、标注和遮罩，并隔离或恢复素材，无需安装 XnViewMP。当前 `main` 的“工具 → 数据审计 → 相似组清洗”支持选择每批最多 500 张，或比较两批各最多 250 张，无需另装 VisiPics。候选由本地 DCT、色彩和边缘描述子生成，需人工复核并保留代表图；隔离前核对项目、SRC/DST、分析范围及素材指纹，共用一个恢复令牌。操作结果失联时先检查，不自动重复隔离。流程与覆盖范围见[相似组复核](docs/SIMILARITY_REVIEW.md)；现有 0.1.1-preview 发行包仍只预检前 500 张。

EbSynth 的关键帧结果传播尚未实现，列为后续 WebUI 功能候选；现有场景检测和分段提帧不包含这项能力。这三项外部工具仍保留可选适配入口，供已有程序的用户通过 `legacy-cli/external-tools.ps1` 配置，不作为 WebUI 安装要求。

## ME 范围与验证

ME 后端支持 DF/LIAE 的 u/d/t/c 架构选项、RG 重算、CUDA FP16、GAN、DF TrueFace、预训练与微调、颜色迁移、数据增强、uniform yaw、并行数据加载、困难样本回放及优化器 CPU offload。RG 保持网络拓扑和权重名称；FP16 使用 AMP，主权重及优化器状态保持 FP32。默认仍为 `liae-ud`、128、batch 4。配置和完整命令见 [ME 后端说明](_internal/DeepFaceLab/ME_README.md)。

在“模型训练”选择“导入旧 TF ME 网络权重”，填写本机旧权重目录的绝对路径、准确文件前缀、原网络结构和新模型名。导入只创建新目录，不覆盖已有模型；导入后开始新的优化器和迭代，旧判别器、优化器、数据采样和 RNG 不会恢复，无需 TensorFlow 运行环境。已有 version 1 原生检查点可加载。

CLI 保留 `--config 文件.json`，并支持 `--config-json '<JSON 对象>'`；两种来源互斥。训练配置可包含全部选项，也可用部分配置更新续训；实际变更需 `--allow-config-change`，采样重置和优化器重置分别使用 `--reset-data-state`、`--reset-optimizer`。`import-tf` 必须提供文件或内联配置之一。Web 向导通过结构化内联配置调用同一训练与导入入口，配置和完整命令见 [ME 后端说明](_internal/DeepFaceLab/ME_README.md)。

此前 ME 后端阶段已实测单 GPU 联合训练、Web 控制与续训、原生 CLI、旧 2002 次迭代模型和 DFM。受控短基准中 RG 显存峰值降低约 14.3%，耗时增加约 21.1%，五步后权重一致；收益取决于配置。RG/FP16/CPU 优化器的八组同步阶段基准及配置建议见 [ME 性能对比](docs/ME_PERFORMANCE_PROFILE.md)。[ME 长训与成片验收](docs/ME_LONGRUN_ACCEPTANCE.md)提供损失历史连续性、固定样本预测、DFM 数值契约和可选视频完整解码工具；本机仅用人工素材完成短程技术自测。多 GPU 参数已接入向导和后端，但本机只有一张物理 GPU；双卡验收脚本、门槛和本机单卡结果见 [ME 双卡验收](docs/ME_DUAL_GPU_ACCEPTANCE.md)。真实双卡、长期真人质量与 DeepFaceLive 现场兼容性仍未验收。最新范围与证据见 [ME 训练验证](docs/ME_TRAINING_VALIDATION.md)，首轮迁移集成记录见 [验证记录](docs/VALIDATION.md)。

## 目录

| 目录 | 用途 |
| --- | --- |
| `_internal/DeepFaceLab/me_backend` | ME 网络、损失、优化器、数据、检查点、Web 控制桥和 DFM |
| `_internal/DeepFaceLab` | 配套 PyTorch 人脸工具、XSeg、Merger 与 CLI |
| `webui` | React 界面、本地服务、固定命令注册表和 Python 数据工具 |
| `launcher` | 在线首次安装、本地运行环境检查、修复和 Windows 启动器 |
| `legacy-cli` | 传统命令菜单和工具入口 |
| `tools` | 运行检查、依赖准备、构建和打包 |
| `workspace` / `workspaces` | 忽略的用户素材、模型、输出和任务记录 |

本项目保留 GPL-3.0 许可和来源信息，见 [来源与许可](docs/SOURCES.md)。
