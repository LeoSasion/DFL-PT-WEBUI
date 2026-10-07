# 后续旧代码、算法与依赖迭代审计

日期：2026-10-07。范围是上一轮质量方案应用之后仍可迭代的实现。此次只补充审计文档，没有修改运行代码、安装包、发布包或用户素材，没有训练。静态调用链发现不等于已发生用户故障；未运行会删除、重命名或覆盖用户数据的旧命令。

基线见 [质量方案应用记录](QUALITY_PIPELINE_APPLIED_20261006.md)。继续遵循 PyTorch 唯一神经网络运行时、ME 唯一换脸训练架构、generic XSeg 原始发行资源与可恢复数据工作流。D:/H3CE_v2 只读参考，不复制其自有应用源码、不建立运行时依赖。

## 优先补齐的实际路径

| 顺序 | 发现与代码证据 | 建议迭代与验收 |
|---|---|---|
| 1 | **旧排序会清空上一轮回收图。** `src/dst.sort_faces` 经 registry:1709/1751 调用 `Sorter.final_process`；[Sorter.py:818](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Sorter.py:818) 的 826–827 删除旧 `aligned_trash` 全部图片，839–855 两轮原地重命名，失败只打印。新 SSIM 隔离没有覆盖此路径。 | 独立批次归档、源/目标 SHA 清单、预览、碰撞前检、提交回执与恢复。验证已有回收图、取消、目标重名、失败重试、磁盘写失败和半途恢复；不能直接把旧排序包装成可恢复操作。 |
| 2 | **元数据仍直接反序列化 pickle。** WebUI annotation → asset helper inspect → `DFLIMG.load` → [DFLJPG.py:150](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/DFLIMG/DFLJPG.py:150)。[PackedFaceset.py:174](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/samplelib/PackedFaceset.py:174)/196、[Util.py:54](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Util.py:54) 也直接 loads。ME 旧权重导入的受限加载器只保护那个导入入口。 | 统一受限加载器、基础类型 schema、深度/字节/数组规模限制；保留历史 NumPy 元数据的明确白名单或独立转换方案。旧 DFL APP15/PAK 格式需兼容，不一次性改成 JSON 并使历史素材失效。验证正常旧样本、异常 global/opcode、截断、超限和完整回读；不执行攻击载荷。 |
| 3 | **新默认提取的干净安装与资源发行尚未闭环。** [DetectorCandidates.py:38](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/facelib/DetectorCandidates.py:38) 依赖 ultralytics/torchvision；[LandmarkCandidates.py:64](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/facelib/LandmarkCandidates.py:64) 依赖 torchvision/timm。包只列在评测 requirements；[install-source.ps1:260](E:/DFL-PT-WEBUI/launcher/install-source.ps1:260) 与 setup-runtime:32 只装 torch+root requirements。新资产保存在 ignored `workspace/.vision-models`，现有 release 清单尚未纳入。 | 拆生产视觉、可选修复和评测依赖；补固定资产安装入口、来源许可/哈希与源码/便携包一致性。全新隔离目录与离线 wheelhouse 真正运行默认提取和可选功能；缺资源明确禁用，不自动下载或静默回退。本机成功不等于新发行包成功。 |
| 4 | **取消清理只覆盖了部分入口。** 新 process-tree 已用于修复/遮罩；[asset-manager.mjs:325](E:/DFL-PT-WEBUI/webui/server/asset-manager.mjs:325) 和 [video-tool-manager.mjs:89](E:/DFL-PT-WEBUI/webui/server/video-tool-manager.mjs:89) 仍 child.kill 后立即完成取消/超时；资产工具还写“已安全终止”。 | 将现有已验证的进程树终止与等待退出契约接入剩余 Python/FFmpeg helper。验证解释器子进程、取消、超时、超量响应、服务关闭与无法确认退出；无法确认时保留占用，不能只靠 kill 返回值宣称成功。 |
| 5 | **传统菜单的项目选择不一致。** [_internal/setenv.bat:19](E:/DFL-PT-WEBUI/_internal/setenv.bat:19) 每次固定 WORKSPACE 为根 workspace，WebUI [paths.mjs:14](E:/DFL-PT-WEBUI/webui/server/paths.mjs:14) 根据 activeId 选择 workspaces/<id>。旧菜单 BAT 又反复 call setenv。 | 显示和统一解析实际项目；过渡期明确标识“默认项目”。两项目隔离夹具验证提取、排序、合成和恢复目标。静态确认路径差异，未运行用户命令验证实际误操作。旧 S3FD/FAN/外部查看器/Qt 入口应明确兼容身份和原生替代，不直接删掉已使用的菜单。 |
| 6 | **复核遮罩尚未被合成直接读取。** [me_backend/data.py:33](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/me_backend/data.py:33) 读取 embedded xseg_mask，因此可用于训练。合成 [Merger.py:169](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Merger.py:169)/186 仅保留 source_landmarks；[MergeMasked.py:79](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/merger/MergeMasked.py:79) 仍重新推理 XSeg-dst。 | 新增明确的“已复核 DST 遮罩”模式，携带对应 aligned 图/仿射/来源；验证回投、同帧多人、裁边、缺遮罩、源图变化及现有模式等价。上轮交付的是可选择 DST 人脸集副本与配置预填，不能表述为合成已直接采用 BiSeNet 复核遮罩。 |

## 下一轮最有价值的质量算法

| 顺序 | 当前实现 | 建议比较与验收 |
|---|---|---|
| 7 | **颜色迁移统计与确定性。** 合成 RCT/LCT/MKL/IDT/SOT/MIX 仍调用 [color_transfer.py:8](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/core/imagelib/color_transfer.py:8)。SOT:41/IDT:110 用全局随机；RCT:172–186 将 mask 外置零后整图统计，而不是仅对有效前景采样；190–196 没有低方差保护。 | 先正确前景统计、低方差/有限值守卫、可复现随机状态，再对照稳健 Lab 统计或带轨迹状态的迁移。审计中的小数组两次 SOT 输出最大差约 0.08066，仅证明随机路径差异，不证明实际短片闪动；平图试验仍为有限值，不声称已出现 NaN。用相同预测脸和真实短片评价肤色、光照、眼嘴及时间稳定。ME data_augmentation 已有样本种子等处理，不混为一谈。 |
| 8 | **PTS、断轨与几何时间稳定。** [Merger.py:193](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Merger.py:193) 按图片顺序建帧；203–231 由相邻位置差估计运动，没有 Δt/切镜/缺口/换人判断。[FrameInfo.py:3](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/merger/FrameInfo.py:3) 无时间或轨迹字段。 | 先绑定已有整数 PTS/time_base，双向唯一关联，切镜/缺口/歧义断轨；再比较有限窗口或局部拟合的中心/尺度稳定，不平滑眼睛开闭、嘴形。H3CE 的 face_geometry_tracks/stable_head 契约可借鉴，常数重新标定。保留原值、使用值和回退原因，验证 VFR、跳帧、交叉、切镜与表情。 |
| 9 | **“最佳人脸”筛选与覆盖。** registry:168–175 仍推荐 final-fast；[Sorter.py:471](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Sorter.py:471) 快速路径将源框面积当 sharpness；blur 采用旧 CPBD，motion-blur 用 Laplacian variance。sort_best:626–627 固定 128 yaw 桶，target_count ≤64 可令每桶配额为 0，658–668 将全部图列入 trash；为静态发现，未运行破坏性命令。 | 首先修小数量/缺元数据与数据提交问题；再比较当前指标和 1–2 个质量候选的人工排序一致性，结合姿态/表情覆盖，展示分项。侧脸/闭眼不能只因 IQA 或正脸偏好被淘汰。上一轮代表图评分已用于相似图复核，尚未代替这条旧排序路径。 |
| 10 | **另一条仍在用的 FaceEnhancer。** [FacesetEnhancer.py:97](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/FacesetEnhancer.py:97) 与合成 superResolution 使用原 DFL FaceEnhancer.npy，当前是 PyTorch 执行。139–158 删除先前 enhanced 输出、默认询问覆盖 aligned；117–119 复制整份旧元数据。上轮源帧 SwinIR 没有替换此链。 | 以旧模型为基线，在真正 aligned/ME 预测脸输入上比较 SwinIR、Real-ESRGAN，单独验证输入域、身份、真实细节、输出尺寸及视频稳定；不沿用源帧评测赢家。工具先改独立批次副本/来源记录，新审核 sidecar 随内容变化失效或重算，不直接照抄为新预测证据。 |
| 11 | **切镜检测。** [video-tool-manager.mjs:177](E:/DFL-PT-WEBUI/webui/server/video-tool-manager.mjs:177) 固定 scene 阈值，showinfo 秒数解析后取三位小数；cuts 截到 501 时最后尾段可能未保留。 | 以现有 FFmpeg 为基线，比较 PySceneDetect AdaptiveDetector 与 TransNetV2 官方 PyTorch 推理候选。绑定源帧索引和整数 PTS、SHA；验证闪光/快摇/渐变/硬切、负时间起点、长片截断提示和尾段。候选尚未在本项目实测，无质量赢家。 |
| 12 | **合成边缘融合。** [MergeMasked.py:97](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/merger/MergeMasked.py:97) 使用固定像素侵蚀/膨胀、高斯羽化及 alpha/seamlessClone。 | 比较距离场羽化、边缘引导 matte 或多频段融合中的 1–2 项。按脸尺寸归一参数，保护细发/眼镜/真实遮挡，检查边缘与颜色时间稳定；保留现有模式，不能预先宣布某个算法普遍更好。 |

所有模型或视觉算法候选沿用用户要求：质量作为选择依据，GPT-6 Astra low 做盲评复核，保留不可判定项；人工参考不足时不制造准确率/AP/NME/IoU。此次未开展新增模型评测。

## 依赖与维护层的具体整理

实际 `.venv` 为 Python 3.12.14、torch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 2.2.6；`pip check` 输出 `No broken requirements found`。Node 24.19.0、FFmpeg 9.0.1 已是项目本地基线。此检查只证明包声明关系无冲突，不证明安装闭环、无安全问题或模型质量。

- **完整依赖锁与分层**：root/评测 requirements 主要锁直接包，BasicSR 1.4.2/GFPGAN 1.3.8 metadata 仍带未锁 tb-nightly、lmdb、yapf 等；本机 tb-nightly 为 2.21.0a20251023。release/package-config.json:12 整棵纳入 site-packages。建议生产/修复/评测分层、传递依赖 hash 锁、离线 wheelhouse 与 Python SBOM，避免研究工具链被无差别发行。
- **旧兼容层**：timm 0.4.12、BasicSR 1.4.2 值得隔离或收敛。[vision_restoration.py:49](E:/DFL-PT-WEBUI/webui/python/vision_restoration.py:49) 仍用 functional_tensor alias。TUFA/SwinIR 固定官方源依赖旧 timm API；升级之前必须同权重数值和视觉复核，不能直接删 shim。
- **统一 OpenCV 声明**：root 使用 GUI opencv-python，backend requirements-me 用 headless；当前只装 GUI，未发现混装。两份 requirements 不应直接叠装；桌面/CPU 夹具各自明确一种 cv2 wheel。
- **更新测试基线**：[webui/tests/requirements.txt:1](E:/DFL-PT-WEBUI/webui/tests/requirements.txt:1) 仍为 Python 3.10/NumPy 1/Pillow 9 并把生产误称 CPython3.7/TensorFlow；改为当前最小 CPU 测试环境或明确另一个兼容矩阵。
- **待移除或拆为验证 extra**：h5py 在仓库代码中只找到 requirements 声明；删除需干净安装与全支持命令验证。onnxruntime 主要用于导出回读和审计，可分层；onnx 仍用于 SFace 解析/DFM 导出，不能按名字删除。
- **前端版本一致性**：WebUI Vite 6.4.3、launcher Vite 6.4.2，可统一受支持的修复版本并查官方公告；未完成新的漏洞扫描，未据版本差异声称存在漏洞。WebUI 的 ^ 依赖已有 frozen pnpm lock，并非每次启动漂移。
- **剩余 GUI 与旧封装**：Qt 当前固定 PySide6，可在真实 XSeg 打开/绘制/保存验证后缩小 PyQt5 fallback。ffmpeg-python 0.2.0 虽旧，VideoEd 的 cut-video 等仍调用，应先统一底层调用再删除。启动时仍自动构建，Vite/plugin-react 暂不能只移入 devDependencies 后从便携包去掉。
- **按可达性清理代码**：main.py:6–7 在解析命令前初始化 nn，nn:37–48 导入全部网络模块，nn:10 全局忽略 FutureWarning；可改按需初始化和局部告警。core.leras、DeepFakeArchi、RMSprop 仍被 ME/XSeg/FAN/FaceEnhancer 调用，不是仅因名字旧就能删除的 TF 残留。后段三引号包住的旧 VGGFace/enhance/get_transform_mat 是历史字符串，应与有效实现区分。
- **旧元数据恢复事务**：[Util.py:43](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/mainscripts/Util.py:43) 的恢复先直接加载 meta.dat，69–77 逐图缩放/覆盖，81 删除该文件。应与旧排序一起增加原件备份、源 SHA、可预览差异、整批 staging/逐项回执与失败恢复；保留旧 68、新 98、手绘和 XSeg 的兼容证据，不把恢复等同于不可逆重编码。
- **多进程生命周期**：[SubprocessorBase.py:223](E:/DFL-PT-WEBUI/_internal/DeepFaceLab/core/joblib/SubprocessorBase.py:223) 初始化等待没有独立 deadline，50–52 terminate 后无界 join；MPFunc/MPClassFuncOnDemand 的 RPC get 也缺 deadline。补存活检查、有限等待、Queue 关闭与结构化失败，验收 worker 初始化崩溃/静默退出/调用端失联和真实少量提取、合成输出等价；避免仅换成另一个池而丢失取消/恢复语义。
- **模块边界**：app-server、command-registry、dfl_asset_tool 已汇集大量业务。先抽公共 process runner、project/environment 契约，再按 assets/video/quality/jobs 拆分，保留统一身份/工作区/关闭守卫。Node environment、BAT setenv、C# 环境读取仍有旧路径与用户目录策略差异；用同一命令的环境等价测试确认后收敛，暂不引入新框架。

## 官方依据与下一步顺序

- pickle 对非可信数据有执行语义风险：[Python 文档](https://docs.python.org/3/library/pickle.html)。此次只确认加载路径，没有发现或运行攻击样本。
- torch 2.9.1/torchvision 0.24.1/CUDA12.8 是官方配套组合：[PyTorch 版本安装表](https://pytorch.org/get-started/previous-versions/)。NumPy2 已使用，迁移仍需注意 ABI/标量提升：[NumPy 迁移文档](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html)。
- Node24 仍处官方 LTS 支持周期：[Node 发布计划](https://github.com/nodejs/Release)。无需仅凭“旧代码”假定运行时过时。
- OpenCV 官方要求同一环境只安装一种 wheel：[OpenCV 安装说明](https://pypi.org/project/opencv-python/)。BasicSR PyPI 1.4.2 的发布记录为 2022-08-30：[官方项目记录](https://pypi.org/project/basicsr/)，包旧本身不等于模型输出差。
- 切镜候选的能力与 PyTorch 实现：[AdaptiveDetector 文档](https://www.scenedetect.com/docs/latest/api/detectors.html#adaptivedetector)、[TransNetV2 官方仓库](https://github.com/soCzech/TransNetV2)。不引用官方他库分数作为本项目得分。

建议先落实 1–6 的数据、运行与安装契约，再做 7–8 的颜色/几何时间稳定，随后对照 9–12 的算法。依赖减负、死代码与模块拆分伴随推进，不以大规模改名或整体重写为目标。若后续实际改动 ME 桥接，按现有一小时内短训范围验证；此次没有触发或开展训练验收。
