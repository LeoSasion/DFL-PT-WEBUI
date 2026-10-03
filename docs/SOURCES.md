# 来源与许可

- 工作台、Windows 启动器、传统工具入口和本地 Python 数据工具来自本机 DFL-WEBUI，参考提交 `85fac83490e06478a266661e1acf7714e82e9aff`，已在本目录改为 ME/PyTorch 运行链。
- 辅助 PyTorch 核心来自 [Jandown/DeepFaceLab-PyTorch](https://github.com/Jandown/DeepFaceLab-PyTorch)，参考提交 `cbbe9792893a4d960a6fe5bf6747d6d26785c32a`。
- ME 的网络和损失以旧项目 ME 的 DF/LIAE、非 RG 和 RG 源码为参考；独立后端沿用先前本机验证的 PyTorch 原型，并加入原生激活重算、对抗训练、训练状态、数据处理、显式网络权重导入、Web 控制、推理适配和 DFM 导出。RG 保留相同网络拓扑。
- 架构与数值验证参考 [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)，提交 `e4b7543ffa1d73b26fce1e31852727f658ba490c`。TensorFlow 仅用于先前独立的验证过程，不是新程序依赖。

根目录与后端保留 GPL-3.0 LICENSE，上游文件中的作者与许可声明保留。新代码遵循项目现有许可。

Node、Python、PyTorch、FFmpeg 和第三方 Python/JavaScript 包适用各自许可；运行材料在发布包中需带原分发物的许可文件。SFace 的 Apache-2.0 许可在 `tools/licenses/SFace-Apache-2.0.txt`，准备脚本会将许可同步到运行材料目录。

本预览版的辅助资产来源：

| 资产 | 固定公开来源与说明 |
|---|---|
| S3FD、2DFAN、3DFAN、FaceEnhancer | [iperov/DeepFaceLab 的固定提交](https://github.com/iperov/DeepFaceLab/tree/e4b7543ffa1d73b26fce1e31852727f658ba490c/facelib)。该提交根目录附 GPL-3.0，未发现这四份权重的单独许可证；发布保留原始文件、项目 GPL 和本来源记录。 |
| SFace | [OpenCV Zoo 的固定模型目录](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface)，Apache-2.0 文本随源码和运行材料保留。 |
| NotoSans-Medium.ttf | 字体文件保留原 2015 Google 版权信息；[Noto Fonts 固定官方 LICENSE](https://github.com/googlefonts/noto-fonts/blob/3858576a0eec798d752be1ce0f9b121705091122/LICENSE) 为 SIL OFL 1.1，副本在 `tools/licenses/NotoFonts-OFL-1.1.txt`。 |
| FFmpeg 9.0.1 essentials | [GyanD 官方发行包](https://github.com/GyanD/codexffmpeg/releases/tag/9.0.1)，保留 LICENSE、README、doc 和 presets；发行 README 指定对应 [FFmpeg 源码提交 bf1b838f2a](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a)。 |
| Python 3.12.14 | [python-build-standalone 20260814 固定发行](https://github.com/astral-sh/python-build-standalone/releases/tag/20260814)，发行附带许可保留在 `_internal/python_base`。 |
| Node.js 24.19.0 | [Node.js 固定官方分发](https://nodejs.org/dist/v24.19.0/)，保留原 LICENSE。 |
| WebView2 SDK | 原生启动器使用 Microsoft WebView2；发布保留对应 SDK 的 LICENSE 与 NOTICE。 |

后端仍保留 Jandown 来源说明。其固定版 README 要求遵守上游许可、保留来源，并包含不鼓励商用的表述；本项目不将这段表述改写为单独权重许可，原说明见 [固定版 README](https://github.com/Jandown/DeepFaceLab-PyTorch/blob/cbbe9792893a4d960a6fe5bf6747d6d26785c32a/README.md)。

`docs/demo-assets/fictional-identities` 中三张演示图为人工生成的虚构成年人物及演示预览，来源说明随图保留；演示预览不是附带训练检查点的输出。旧真实素材截图、本机部署绑定、QA 文件及个人工作区不随公开分发。

检测/FAN/FaceEnhancer 权重和 SFace/FFmpeg 的固定来源及 SHA-256 在 `tools/prepare-vision-runtime.ps1`；Node 和 Python/PyTorch 的安装版本在启动器配置和根 `requirements.txt`。权重沿用原始序列化格式，不意味着使用 TensorFlow 执行网络。
