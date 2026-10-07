# 镜头检测比较与原始时间戳验收

生产默认保留 FFmpeg。GPT-6 Astra low 在隐藏算法身份的第二轮评分中给 FFmpeg、PySceneDetect AdaptiveDetector 各 5/5，TransNetV2 4/5；前两者在本次可见素材上并列。耗时只作诊断记录，没有参与选择。这些主观分数不等于准确率百分比或泛化 F1。

## 比较材料与盲评

使用已授权只读的 `workspaces/real-interview-me-acceptance/data_dst.mp4`（4 秒，100 帧，SHA256 `ef4846a61fe2915480c7c3b1e156dba708d6adb96df1ce665112d4cf6723be5d`）和 Big Buck Bunny 的公开 30 秒编辑动画片段（720 帧，SHA256 `84abe3ddf1938f0a6053cf6cc7874434177cdce927ba07d603cdd84c4c67a6ab`）。后者是动画，不能冒充真人场景验证。

Big Buck Bunny 归属 `(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org`，影片为 [CC BY 3.0](https://wiki.creativecommons.org/wiki/Big_Buck_Bunny)。官方下载域返回 HTTP 403，本次二进制来自固定 GitHub 镜像提交 `915c4b2aba75614b20dec3852375b394bb305f10`，这一事实及下载 URL、SHA、大小保存在 `.launcher-install/locked-runtime-20261007/scene-comparison/BigBuckBunny480p30s.mp4.source.json`。没有把镜像称为官方二进制来源，也没有发布影片。

`tools/scene-benchmark.py` 先锁定原媒体 SHA，再运行 FFmpeg、Adaptive、TransNetV2，最后复核原 SHA。每份盲包随机分配 A/B/C，提供两张完整时间顺序概览及六张候选切点的相邻原帧图。评分者先看概览，再看候选；揭盲键在评分锁定前保持隐藏。

第二轮已锁分、再揭盲，键 SHA256 `ee24fde71920329e4c97f0979a3f5fa521fb981b4c638dd43061254a67be6f35`：

| 算法 | 可见且确认的硬切覆盖 | 可见额外误报 | 第一新帧位置 | Astra 主观总分 |
| --- | --- | --- | --- | --- |
| FFmpeg（B） | 采访 1，动画 3 | 0 | 四个确认切点均一致 | 5/5 |
| AdaptiveDetector（A） | 采访 1，动画 3 | 0 | 四个确认切点均一致 | 5/5 |
| TransNetV2（C） | 采访 1，动画 3 | 动画渐亮位置 1 | 四个确认切点均一致 | 4/5 |

第二轮完整证据在 `.launcher-install/locked-runtime-20261007/scene-comparison/blind-v2/`：`blind/locked-scores.json`、`blind/ground-truth-annotations.json`、原始每算法结果、隐藏键及八张 PNG。概览每六帧取一帧，候选局部才显示连续原帧，因此不能排除概览间的短镜头，也不能把“未漏掉确认切点”改写成完整召回率。采访和动画的素材覆盖仍有限。

第一轮证据原样保留在 `blind-v1/`。它发现 TransNet 的切点约定提前一帧，评分为 3.5/5（其余两者 5/5）。核对固定官方 `inference/transnetv2.py` 的 `predictions_to_scenes` 后，适配层改为阈值活跃段结束后第一帧，即官方 1→0 边；所有过渡原帧仍归入前一段，连续输出没有漏帧。第一轮未被改写为成功；第二轮是新的图包、随机别名和评分。

## 固定模型与运行时

- FFmpeg：项目本地 9.0.1，`lavfi.scene_score`，默认阈值 0.32。
- Adaptive：官方 PySceneDetect 0.7.1，默认比率阈值 3.0，15 帧最短段、前后窗口 2、最小内容值 15。不同算法的阈值单位独立。
- TransNetV2：官方 `soCzech/TransNetV2` 提交 `85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed` 的 PyTorch 源码和 Git LFS checkpoint。源文件、转换代码、许可证和原始 checkpoint 均留存并锁 SHA。

官方仓库没有可直接取得的官方 `.pth`。`tools/convert-transnetv2-checkpoint.py` 使用独立的纯 Python checkpoint 文件读取器及固定官方名字/轴映射进行一次性转换，没有导入或安装 TensorFlow 神经网络运行时。84 个 FP32 原张量与 checkpoint 字节一致、逆置换逐值一致，90 个 PyTorch state 项严格加载和保存回读；转换权重 SHA256 `bb135534c7f734d421d84c66d2124e8f3971e5ce134962a23866383e1ea0c4c6`。维护转换器不会成为生产依赖。

实际 PyTorch FP32 推理已在上述两段影片及合成 VFR 上完成；没有声称完成 TensorFlow/PyTorch 前向等价。该未验证项保存在 `identity.json`，没有用第三方 `.pth` 假冒官方转换结果。场景运行时由可选 `scene` 哈希配置提供，推理不下载模型；资源通过受控 `transnetv2` 组安装。

## 时间、完整尾段和失败处理

`webui/python/scene_detection.py` 输出 schemaVersion 2：媒体名/SHA/bytes、源帧数、第一 PTS、整数 terminal PTS/timeBase、全部切点、最多 500 个显示段及总数/截断标志。每个切点保留 `sourceFrameIndex/pts/timeBase`，显示段有 `startFrameIndex/endFrameIndexExclusive/startPts/endPts/timeBase`，秒值相对第一源 PTS，未四舍五入。超出 500 段时保留前 499 个边界并让最后段覆盖完整尾部，全部切点仍供时序处理使用。

合成 VFR 功能样本起始 PTS 2000、timeBase 1/1000，36 帧，已知硬切第一新帧 18（PTS 3080），尾部 4120。第一轮 FFmpeg/Adaptive 返回 18，旧 TransNet 适配返回 17，原记录保留；修正后的适配按官方过渡结束约定处理。合成样本只用于接口和时间验收，不构造视觉质量赢家。

媒体 SHA 在探测前、探测后和检测后复核。缺失原始 PTS、非递增 PTS、无法证明末帧整数时长、HDR/宽色域未经显式转换、媒体变化、无效切点或非有限分数均失败，不能发布猜测结果。标准输出只有 JSON，诊断写 stderr。`--capabilities` 明确反映可选依赖/受控模型资源可用性。

`tools/tests/test_scene_detection_contract.py` 的五项测试验证非整数/越界切点拒绝、VFR 偏移与精确尾部、500 段仍完整覆盖尾部、探测/推理期间媒体变化拒绝。真实 VFR 证据在 `.validation/scene-vfr-20261007/`，候选推理、盲评和测试均没有修改原素材。
