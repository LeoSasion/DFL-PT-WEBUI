# 新仓库迁移说明

本目录是重新初始化的独立 Git 仓库，默认分支 `main`。没有复制旧 `.git`、旧项目更新源、TF/Python 3.7 环境或用户数据；旧仓库和先前 ME 原型均不再作为运行依赖。

## 后端

换脸训练只保留 ME，支持 DF/LIAE 与 u/d/t/c 选项、RG、CUDA AMP、GAN、DF TrueFace、预训练/微调、增强、颜色迁移、并行采样、困难样本回放及 CPU 优化器状态。`me.py` 提供训练、预测和旧网络权重导入，`main.py` 提供统一工具入口，`me_backend/web_bridge.py` 接受 Web 的固定参数与控制文件。训练和导入支持互斥的 `--config FILE` / `--config-json JSON`，内联内容必须是 JSON 对象，未知配置字段会拒绝；旧文件配置仍可使用。保存和安全停止有原子确认记录；界面只在匹配本次 close 请求且进程成功退出后显示“已安全停止”。

ME 使用整份 `me.pt` 原生检查点，包含生成器、判别器、优化器、AMP scaler、迭代/更新计数、RNG、数据和回放状态。已有 version 1 原生检查点可加载，当前后端已核对旧 2002 次迭代模型的加载和有限预测。RG 通过 activation checkpointing 重算激活，不改变权重拓扑；AMP 保留 FP32 主权重和优化器状态。

旧 TF 的分散 `.npy` 权重通过 `me.py import-tf` 或 Web 的“导入旧 TF ME 网络权重”显式转换，逐张量验证名称、形状、类型和有限值，并记录来源 SHA-256。必须提供文件或内联配置之一，填写准确原结构与旧文件前缀；Web 接受本机旧目录的绝对路径。导入只向尚不存在的新模型目录写入，迭代归零、优化器重新建立；旧判别器、优化器、RNG 和数据状态不导入。它不是旧训练器的完整续训，也不需要旧 TF 运行时。模型发现同时检查对应名称的 `metadata.json` 与 `me.pt`。

原生续训严格核对配置；JSON 可以是部分配置，但实际改动必须加 `--allow-config-change`。网络结构字段不能在续训中改变。更换优化器类型或已存在判别器的形状需 `--reset-optimizer`；重置时保留网络及兼容的判别器权重，优化器更新计数重新开始，显示迭代数保留。更换数据集需 `--reset-data-state`。Web 切换预训练阶段时同样要求重置采样状态，并清空困难样本回放。预训练转微调可继续同一模型；“从现有模型初始化”则只复制当前项目来源模型的网络权重，新模型的优化器与迭代从零开始。命令及配置见 [ME 后端说明](_internal/DeepFaceLab/ME_README.md)。

项目 XSeg 使用 `XSeg_data.dat`、`XSeg_256.pth` 和 `XSeg_256_opt.pth`，作为固定的辅助模型实例。内置通用模型随包保存在 `_internal/model_generic_xseg`，仅含转换后的推理权重、WF 类型元数据和来源许可；没有训练样本、历史或优化器。`tools/convert-generic-xseg.py` 按原始 SHA 校验、受限 NumPy 反序列化和严格张量映射转换官方权重，转换无需 TensorFlow。兼容加载器也能识别完整的旧命名 `.npy` 张量；重命名本身不做数据转换。ME 合成通过专用推理适配器接入原有 Merger；DFM 由 ME 原生 PyTorch 网络导出。

## Web 训练配置

ME 向导已接入完整后端选项，按网络结构、优化与训练、显存与精度、重建与风格损失、GAN 与 TrueFace、采样与增强、困难样本回放分组。提供“默认训练”“RG + FP16”“预训练”“后期微调”预设，仍可逐项修改。选择已保存模型会恢复配置并固定网络结构；启用“允许续训配置变更”后可编辑其余训练参数，参数页与运行前复核页都会列出差异。Web 本地服务验证配置，再以单独的 `--config-json` 参数传入后端。

“数据与运行计划”允许指定当前项目工作区内的 SRC/DST 文件夹或 PAK/ZIP 文件，并可选择专用预训练人脸集。预训练时 SRC/DST 共用此数据，GAN、TrueFace 和风格损失暂时禁用。打包及自定义数据、预训练阶段目前没有姿态评测快照，训练、预览、保存和恢复仍可使用。

目标迭代默认用于进度估算；只有开启“达到目标后自动保存并停止”才形成总迭代上限。“仅检查采样”生成 `debug-src.png` / `debug-dst.png`，不执行训练或覆盖 `me.pt`，可关闭 FP16 并使用 CPU。GPU 索引支持 `0` 或 `0,1`，要求索引不重复且 batch 至少等于所选 GPU 数量；FP16 训练需要 CUDA。

## 工具映射

| 能力 | 新仓库实现 |
| --- | --- |
| SRC/DST 视频提帧和成片封装 | `mainscripts/VideoEd.py`、本地 FFmpeg、Web 视频工具 |
| 自动/手动切脸、2D/3D landmarks、调试预览 | `mainscripts/Extractor.py`、PyTorch S3FD/FAN |
| 人脸增强、重设尺寸 | `FacesetEnhancer.py`、`FacesetResizer.py` |
| 排序、姿态筛选、错脸识别 | `Sorter.py`、`yaw_image_filter.py`、`ErrFaceFilter.py` |
| PAK/ZIP、人物子目录、元数据导入导出 | `samplelib/PackedFaceset.py`、`mainscripts/Util.py` |
| XSeg 标注、训练、应用、删除和导出 | Web 编辑器、`XSegEditor`、`Model_XSeg`、`XSegUtil.py` |
| 人物分组、质量/姿态/相似度审核、对齐修复、恢复 | `webui/python/dfl_asset_tool.py` 和相应 Web 面板 |
| ME 完整配置、预训练/微调、初始化、采样检查、预览、保存、评估、续训 | `me_backend`、`me.py`、Web 向导和固定 `train.me` 注册项 |
| 旧 TF ME 网络权重严格导入至新模型 | `checkpoint_import.py`、`me.py import-tf`、固定 `model.import_me_tf` 注册项 |
| ME 合成、DFM、MP4、合成结果审核 | `model_adapter.py`、`Merger.py`、`export.py`、Web 工具 |
| 本地项目、日志、ConPTY、GPU 遥测、任务恢复 | `webui/server`、`webui/scripts/local-manager.mjs` |
| Windows 入口、依赖修复、打包 | `launcher`、`tools`、传统工具菜单 |

旧 `SAEHD`、`AMP`、`Q384/Q512` 等换脸入口、CP37/TF 依赖安装和旧远端更新流程已移除。其旧验收报告保留在原仓库，没有作为新版本的测试证据复制。新的 ME RG 由 PyTorch 实现，不依赖旧 TF 运行时。

旧错脸分类器转换为 JSON 树数据，使用 NumPy 推理，并用原模型概率对照验证。权重加载会验证全部键及形状，不能以缺失权重的随机网络假装加载成功。PAK/ZIP 保留元数据并能往返解包；`absdiff` 改为 NumPy 分块计算。

## 外部材料边界

aligned 浏览和日常相似组清洗默认使用 WebUI，无需安装 XnViewMP 或 VisiPics。相似组目前只分析前 500 张，暂不支持切换批次或跨批查重；EbSynth 关键帧传播尚无 Web 实现，作为后续原生功能候选，不新增安装流程。菜单保留已有外部程序的可选适配入口。官方通用 XSeg 推理权重已加入源码包和便携包。

旧 latent CLI 本身是占位实验，保留其尚未支持提示。下一版源码的图像编辑通过“工具实验室 → 图像工具”接入奇智 API，提供本机 Key 配置、素材确认、后台查询和结果另存，详见 [图像服务](docs/IMAGE_SERVICE.md)；0.1.1-preview 发行包尚不含这项功能。高级 ME 参数与阶段转换已接入 WebUI。多 GPU 调度和界面选择均已实现，但本机只有一张物理 GPU，真实多卡仍待验收；长期真人质量未验收。具体证据见 [ME 训练验证](docs/ME_TRAINING_VALIDATION.md)，首轮迁移记录见 [验证记录](docs/VALIDATION.md)。

## 环境和版本管理

Python、Node、FFmpeg 和辅助权重都是本目录的独立文件。大体积运行材料及模型不提交 Git；源码包与完整运行包应分开，完整运行包必须包含 Python base 与 venv。路径迁移时会修复 venv 的 base 路径，不能只复制 `.venv`。

公开源码与 preview 位于 [DFL-PT-WEBUI](https://github.com/LeoSasion/DFL-PT-WEBUI)。启动器只检查、修复和启动本地项目，在线自动更新关闭。
