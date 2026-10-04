# DFL-PT-WEBUI

Windows 本地工作台：**ME PyTorch 训练 + WebUI + 数据处理、XSeg、合成和视频工具**。

**0.1.1-preview** 面向用户实测反馈。换脸训练使用 ME/PyTorch，XSeg 是辅助遮罩模型；旧 ME 网络权重可以通过显式导入命令迁移。运行环境与项目材料保持独立。

- [下载预览版](https://github.com/LeoSasion/DFL-PT-WEBUI/releases/tag/v0.1.1-preview)：源码包和 Windows x64 便携包，均随包提供通用 XSeg 推理权重。
- [完整安装步骤](docs/INSTALL.md)、[版本说明](docs/RELEASE_NOTES.md)、[已知限制](docs/KNOWN_ISSUES.md)。
- [反馈问题](https://github.com/LeoSasion/DFL-PT-WEBUI/issues/new/choose)；反馈需要的信息见 [FEEDBACK.md](docs/FEEDBACK.md)。

## 启动

双击根目录 **`启动 WebUI.bat`**，打开 <http://127.0.0.1:4173>。需要传统命令工具时双击 **`传统命令菜单.bat`**。

便携包带独立 Python 3.12.14、PyTorch 2.9.1 + CUDA 12.8、Node.js 24.19.0、FFmpeg 和必要辅助权重。源码安装可在仓库根目录运行 `install-source.bat`，自动准备这些材料、安装依赖并构建 WebUI。需要 Windows x64；GPU 训练需要支持 CUDA 12.8 的 NVIDIA 驱动。Python base 在 `_internal/python_base`，venv 在 `.venv`。

运行检查：

```powershell
Set-Location C:\DFL-PT-WEBUI
.\.venv\Scripts\python.exe tools\smoke_test.py
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\bootstrap.ps1 -ProjectRoot C:\DFL-PT-WEBUI -NoNetwork
```

源码 Git 不提交运行时、辅助权重、依赖包、个人素材和模型；克隆后先执行安装入口。便携包不带个人素材、训练检查点、本机任务状态或旧工作台截图。源码安装支持准备离线材料，详见 [运行环境说明](launcher/runtime-README.md)。启动器在线自动更新仍关闭，升级通过发行页下载。

当前开发证据覆盖单卡、小参数的保存、停止、恢复和初步收敛趋势。正式案例画质、真实双卡和新用户不同机器上的完整流程等待实测反馈，详见 [短程验收](docs/ME_SHORT_RUN_ACCEPTANCE.md)。

## 工作流程

1. 在设置中创建或切换项目，在工作区导入 SRC 和 DST 视频。
2. 提取视频帧，再提取带 DFL 元数据的 aligned 人脸。
3. 使用人物分组、质量审查、相似图片、姿态图谱和隔离恢复工具整理数据。
4. 按需编辑 XSeg 标注、训练辅助 XSeg 模型并应用遮罩。
5. 在“模型训练”新建 ME，选择名称、训练预设和设备，再检查完整参数。默认 128 / batch 4；GPU 索引可填 `0` 或 `0,1`，多卡时 batch 至少等于卡数。
6. 在“总览”查看真实预览和损失曲线；可以保存、手动备份、刷新预览和生成评估快照。训练过程还会定期生成校验过的自动备份。
7. 用“安全停止”保存并退出，随后选择同一模型续训；也可启用“达到目标后自动保存并停止”。两个可比较的快照可用于质量诊断。
8. 在“模型应用”使用 ME 合成 DST 帧，导出 MP4；也可导出 ONNX 格式 DFM。

项目素材分别保存在 `workspace` 或 `workspaces/<项目标识>`。ME 模型保存为 `model/<模型名>/me.pt`，并带有 `metadata.json`、预览和损失记录。检查点包含网络、判别器、优化器、AMP scaler、迭代与更新计数、随机数、采样状态和困难样本回放。

ME 训练由独立于 Web 服务的监督进程承载；Web 服务重启后会依据持久状态重连训练、日志和控制，并继续占用原模型/GPU 锁。“训练心跳”区分正常、检测中和疑似停滞，疑似停滞不会自动杀死训练。自动备份默认每 1000 次迭代保存一次并保留最近 3 代，手动备份不受这项轮换影响；备份包含检查点、元数据和现有损失历史。在“工作区”的模型恢复点中可查看和选择校验过的备份，恢复前会保留当前版本，运行中的任务不可恢复模型。

ME 向导提供“默认训练”“RG + FP16”“预训练”“后期微调”快速配置，以及网络结构、优化与训练、显存与精度、重建与风格损失、GAN 与 TrueFace、采样与增强、困难样本回放的完整参数分组。参数仍可逐项调整；FP16 训练需要 CUDA，TrueFace 仅适用于 DF。预训练需要指定专用人脸集，SRC 和 DST 共用该数据，GAN、TrueFace 和风格损失在此阶段不生效。

选择已有模型会恢复检查点配置。调整续训参数必须启用“允许续训配置变更”，向导会显示差异；网络结构保持固定。切换优化器类型或已启用 GAN 的判别器形状需“重置优化器”，更换数据集或进入/退出预训练阶段需“重置采样状态”。预训练转微调可继续同一模型，启用配置变更、选择“后期微调”，再重置采样状态并指定 SRC/DST；新模型可通过“从现有模型初始化”只复制网络，建立新的训练进度。

“数据与运行计划”支持当前项目工作区内的自定义 SRC/DST 目录、PAK 或 ZIP 文件。目标迭代默认只估算进度；启用自动停止后才作为总迭代上限。“仅检查采样”可在 CPU 上输出增强与遮罩检查图，不执行训练或覆盖检查点；CPU 检查时关闭 FP16。姿态评测现支持目录、PAK、ZIP、自定义及预训练人脸集，以数据集指纹和包内成员名固定样本；只有默认的非打包 SRC/DST aligned 目录会跳转到对应姿态图谱。

## 配套工具

保留视频提帧/封装、S3FD 切脸、2D/3D FAN、错脸筛选、排序、姿态筛选、PAK/ZIP 打包解包、FaceEnhancer、重设尺寸、DFL 元数据和 XSeg 工具，以及 Web 的人物选择、审核、修复、标注、相似度、回收恢复、诊断、日志和任务管理。详细映射见 [迁移说明](MIGRATION.md)。

ME 训练直接读取带 DFL 元数据的 aligned JPG/JPEG，以及 PAK/ZIP 打包人脸和人物子目录。通用 XSeg 推理权重随源码包和便携包提供，位于 `_internal/model_generic_xseg`，可直接使用“内置遮罩应用”；Git 克隆安装会按固定散列获取相同文件。权重来自官方 DFL 通用模型，保留 WF 人脸类型并转换全部张量供 PyTorch 加载；它不是项目 XSeg 的完整训练状态，专用遮罩仍可在项目内标注和训练。来源、许可证和转换记录见 [来源与许可](docs/SOURCES.md)。

aligned 预览与清洗默认使用 WebUI 的 SRC/DST 人脸浏览器，可检查定位点、标注和遮罩，并隔离或恢复素材，无需安装 XnViewMP。日常相似图审查默认使用“工具实验室 → 数据审计 → 相似组清洗”，无需另装 VisiPics；候选由本地 DCT、色彩和边缘描述子生成，需人工复核并保留代表图。每次仅分析当前 aligned 的前 500 张，暂不支持切换批次或跨批查重。

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
| `launcher` | 本地运行环境检查、修复和 Windows 启动器 |
| `legacy-cli` | 传统命令菜单和工具入口 |
| `tools` | 运行检查、依赖准备、构建和打包 |
| `workspace` / `workspaces` | 忽略的用户素材、模型、输出和任务记录 |

本项目保留 GPL-3.0 许可和来源信息，见 [来源与许可](docs/SOURCES.md)。
