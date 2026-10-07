# 来源与许可

## 2026-10-06 非训练候选对照

本轮新增适配和评测工具在本项目独立实现；H3CE 自有 AGPL 源码未复制。以下官方网络源码和权重仅存本机 ignored 对照缓存，固定原提交、SHA 与许可。尚未通过发行准入，不因本机跑通就携带到公共源码/便携包；现有 verified generic XSeg 发行要求保持。

| 候选 | 官方来源 | 对照记录 |
|---|---|---|
| YOLO11/12/26 face | [akanametov/yolo-face](https://github.com/akanametov/yolo-face)，Ultralytics AGPL-3.0；具体发布资产散列锁定 | `tools/vision-model-candidates.json`，`docs/VISION_DETECTOR_EXTRACTION.md` |
| TUFA、ORFormer、Regression | [TUFA](https://github.com/Jiahao-UTS/TUFA)、[ORFormer](https://github.com/ben0919/ORFormer)、[Regression](https://github.com/ca-joe-yang/regression-without-softarg)；分别记录 GPL-2.0、未见根许可、Apache-2.0 与独立权重条款状态 | `docs/VISION_LANDMARK_EVALUATION.md` |
| AdaFace、MagFace | [AdaFace](https://github.com/mk-minchul/AdaFace) MIT、[MagFace](https://github.com/IrvingMeng/MagFace) Apache-2.0；权重与示例来源另记 | `docs/VISION_IDENTITY_EVALUATION.md` |
| SwinIR、Real-ESRGAN、GFPGAN | [SwinIR](https://github.com/JingyunLiang/SwinIR)、[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[GFPGAN](https://github.com/TencentARC/GFPGAN)；官方网络与权重保留原 Apache-2.0/BSD-3-Clause/Apache-2.0 记录 | `docs/NON_TRAINING_UPGRADE_RESULTS_20261006.md` |
| BiSeNet、SAM2.1 | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) MIT、[SAM2](https://github.com/facebookresearch/sam2) Apache-2.0 | `docs/VISION_MASK_EVALUATION.md` |
| LPIPS、DISTS | [LPIPS](https://github.com/richzhang/PerceptualSimilarity) BSD-2-Clause、[DISTS](https://github.com/dingkeyan93/DISTS) MIT；ImageNet 主干权重固定到官方 Torchvision 文件 | `docs/VISION_METRIC_EVALUATION.md` |

代码许可与权重/训练数据条款分别记录；上述表是来源说明，不是所有资产均可分发的结论。评测图像和源项目绝对磁盘路径不进入公共实现或发行资产。

## 现有发行来源

- 工作台、Windows 启动器、传统工具入口和本地 Python 数据工具来自本机 DFL-WEBUI，参考提交 `85fac83490e06478a266661e1acf7714e82e9aff`，已在本目录改为 ME/PyTorch 运行链。
- 辅助 PyTorch 核心来自 [Jandown/DeepFaceLab-PyTorch](https://github.com/Jandown/DeepFaceLab-PyTorch)，参考提交 `cbbe9792893a4d960a6fe5bf6747d6d26785c32a`。
- ME 的网络和损失以旧项目 ME 的 DF/LIAE、非 RG 和 RG 源码为参考；独立后端沿用先前本机验证的 PyTorch 原型，并加入原生激活重算、对抗训练、训练状态、数据处理、显式网络权重导入、Web 控制、推理适配和 DFM 导出。RG 保留相同网络拓扑。
- 架构与数值验证参考 [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)，提交 `e4b7543ffa1d73b26fce1e31852727f658ba490c`。TensorFlow 仅用于先前独立的验证过程，不是新程序依赖。

根目录与后端保留 GPL-3.0 LICENSE，上游文件中的作者与许可声明保留。新代码遵循项目现有许可。

Node、Python、PyTorch、FFmpeg 和第三方 Python/JavaScript 包适用各自许可；运行材料在发布包中需带原分发物的许可文件。SFace 的 Apache-2.0 许可在 `tools/licenses/SFace-Apache-2.0.txt`，准备脚本会将许可同步到运行材料目录。

本预览版的辅助资产来源：

| 资产 | 固定公开来源与说明 |
|---|---|
| S3FD、2DFAN、3DFAN | [iperov/DeepFaceLab 的固定提交](https://github.com/iperov/DeepFaceLab/tree/e4b7543ffa1d73b26fce1e31852727f658ba490c/facelib)。该提交根目录附 GPL-3.0，未发现这三份权重的单独许可证；发布保留原始文件、项目 GPL 和本来源记录。旧 FaceEnhancer 源码、权重及发行项已于 2026-10-07 移除，历史审计保留。 |
| MambaIRv2、Real-ESRGAN、TransNetV2 | 固定上游源码、官方原权重或官方权重数据转换、SHA、来源和许可证分别记录在 `release/vision-assets.json` 的 `mambairv2`、`realesrgan-x4plus`、`transnetv2` 组；本地受控资源包逐项校验，不把源码许可证推定为单独权重再分发授权。Mamba 只引入 PyTorch 推理；TransNet 官方检查点以数据读取转换，无 TensorFlow 网络运行时。 |
| 通用 XSeg | [iperov 固定 README](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/README.md) 链接的 [官方 Mega 目录](https://mega.nz/folder/Po0nGQrA#dbbttiNWojCt8jzD4xYaPw)中 `DeepFaceLab_DirectX12_build_05_04_2022.exe` 的 `_internal/model_generic_xseg/XSeg_256.npy`，WF、584205 次迭代。官方包附 GPL-3.0，模型目录没有单独许可证；原包许可副本为 `tools/licenses/GenericXSeg-DeepFaceLab-GPL-3.0.txt`，原摘要及精确来源在 `release/generic-xseg*`。 |
| SFace | [OpenCV Zoo 的固定模型目录](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface)，Apache-2.0 文本随源码和运行材料保留。 |
| NotoSans-Medium.ttf | 字体文件保留原 2015 Google 版权信息；[Noto Fonts 固定官方 LICENSE](https://github.com/googlefonts/noto-fonts/blob/3858576a0eec798d752be1ce0f9b121705091122/LICENSE) 为 SIL OFL 1.1，副本在 `tools/licenses/NotoFonts-OFL-1.1.txt`。 |
| FFmpeg 9.0.1 essentials | [GyanD 官方发行包](https://github.com/GyanD/codexffmpeg/releases/tag/9.0.1)，保留 LICENSE、README、doc 和 presets；发行 README 指定对应 [FFmpeg 源码提交 bf1b838f2a](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a)。 |
| Python 3.12.14 | [python-build-standalone 20260814 固定发行](https://github.com/astral-sh/python-build-standalone/releases/tag/20260814)，发行附带许可保留在 `_internal/python_base`。 |
| Node.js 24.19.0 | [Node.js 固定官方分发](https://nodejs.org/dist/v24.19.0/)，保留原 LICENSE。 |
| WebView2 SDK | 原生启动器使用 Microsoft WebView2；发布保留对应 SDK 的 LICENSE 与 NOTICE。 |

后端仍保留 Jandown 来源说明。其固定版 README 要求遵守上游许可、保留来源，并包含不鼓励商用的表述；本项目不将这段表述改写为单独权重许可，原说明见 [固定版 README](https://github.com/Jandown/DeepFaceLab-PyTorch/blob/cbbe9792893a4d960a6fe5bf6747d6d26785c32a/README.md)。

`docs/demo-assets/fictional-identities` 中三张演示图为人工生成的虚构成年人物及演示预览，来源说明随图保留；演示预览不是附带训练检查点的输出。旧真实素材截图、本机部署绑定、QA 文件及个人工作区不随公开分发。

通用 XSeg 原件 SHA-256 为 `5eb6c5b67a84ca4bbdbddb036011b488c03abde0c1e45ad700d290666a43faa5`；转换后 `XSeg_256.pth` 为 `26e45677ef3136e0327f0fd51e452cbea81a703ee7fea9121b01ba58da65c385`。转换严格验证全部 222 个张量的名称、形状及有限值，按当前 PyTorch 卷积布局复制，并验证保存后的数值完全一致；没有重新训练。包内 `XSeg_data.dat` 只写入可信的 WF 类型元数据，原始训练状态、样本及优化器均不导入。两类发行包均包含此权重；额外的同名下载资产供 Git 克隆安装使用。

检测/FAN 权重和 SFace/FFmpeg 的固定来源及 SHA-256 在 `tools/prepare-vision-runtime.ps1`，新生产视觉资源与可选修复/场景资源在 `release/vision-assets.json`；通用 XSeg 在 `release/generic-xseg.json`；Node 和 Python/PyTorch 的安装版本在启动器配置和根 `requirements.txt`。保留或转换 NumPy 张量序列化不意味着使用 TensorFlow 执行网络。

最佳训练人脸选集的唯一新增质量候选来自 [Efficient-FIQA 官方固定提交](https://github.com/sunwei925/Efficient-FIQA/tree/68f4c9eb90faa6a474c635f0f34df64304397595)，只读取 EdgeNeXt-XXS 学生检查点，不引入教师或其他 FIQA。原源码、测试预处理、Apache-2.0 文本和检查点逐项固定 SHA，见 `webui/python/vision_fiqa.py` 与显式准备脚本 `tools/prepare-efficient-fiqa.py`。同一项目 Python 的推理子进程读取原始 timm 1.0.19 wheel（含其许可证），不升级 TUFA 的全局 timm 0.4.12；严格加载权重并以 FP32 执行。资源只进入本地忽略缓存，推理不下载，公开权重再分发尚未准入。是否代替既有综合质量分由 `docs/EFFICIENT_FIQA_COMPARISON_20261007.md` 的预锁比较协议决定；图像质量分不是身份或训练效果的准确率。
