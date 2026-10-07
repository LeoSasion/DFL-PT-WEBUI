# 合成质量与旧算法升级记录

日期：2026-10-07。范围是既有 ME 预测之后的合成、颜色、几何、边缘融合和合成会话，不改变 ME 网络、桥接、训练配置或检查点。H3CE 的时序测试契约仅作为只读参考；未复制其应用源码，没有增加运行时依赖。

## 已接入的配置

| Web 参数 | 可选值 / 范围 | 默认及兼容 |
|---|---|---|
| maskMode | 10 `reviewed-dst`；11 `learned-prd*reviewed-dst` | 旧 0–9 语义保留。新模式缺复核绑定立即拒绝，不回退 XSeg。 |
| colorTransfer | 新 `robust-lab`、`lab-quantile` | WebUI 引导式合成默认 `robust-lab`，其余模式仍可选择；原生旧配置默认不变。RCT 修复前景统计。采用依据与低置信限制见末节。 |
| randomSeed | 0..4294967295 整数 | 0。每轨局部 seed，不修改全局 NumPy/SciPy 随机状态。 |
| geometryMode / geometryStrength | `off` / `bounded-center-scale`，0..100 | `off` / 0。必须有整数 PTS 和逐帧 SHA 绑定的提帧清单。 |
| blendMode / blendWidth | `legacy` / `distance` / `multiband`，0..20% | `legacy` / 3%。Astra 盲评无可靠优胜，保持原融合默认。 |
| superResolutionModel | `mambairv2` / `realesrgan-x4plus` | 最终预测脸增强默认 `mambairv2`，强度默认 0；两模型输入域评测、选择及限制见 [全流程最终报告](ALL_PIPELINE_UPGRADE_20261007.md)。不再保留旧 FaceEnhancer 或 SwinIR 预测脸增强入口，两者均采用正确 BGR float32 [0,1] 域（is_tanh=False）。源帧修复中的 SwinIR 是独立流程。 |

## 复核遮罩与来源

合成模式 10/11 只接受已发布 DST 副本目录的 `mask-assist-provenance.json`。每个将合成的 aligned 文件必须是 assisted 条目，并具有 sourceFrame 绑定。训练用途副本可存在未辅助图片或无原帧绑定，但这种副本不能隐式满足新合成模式；应使用有原帧的完整复核批次或选择旧模式。

预检核对回执状态、aligned SHA、原帧 SHA/尺寸、原始与对齐 68 点摘要、当前仿射和 aligned 画布、68 点回投一致性、embedded mask 大小及有限值。在每次预测前重新核对原帧、aligned 和回执 SHA。遮罩从实际 aligned 坐标经逆仿射回源帧，再进入当前预测坐标；没有重新运行 XSeg，没有转换 98 点拓扑，没有把缺遮罩当作成功。

`merge.audit.json` 保存配置、种子规则、原帧与 aligned SHA、整数 PTS、轨迹、原/使用中心与尺度、窗口贡献帧、回退原因、真实 Δt、速度/曝光与运动模糊量、最终图及遮罩 SHA。合成异常标记 failed；缺输出标记 incomplete。

## 颜色、时序与融合

RCT 只对有效前景采样；空前景返回原颜色，低方差通道只平移均值，保护有限值。SOT、IDT、MIX 的随机方向来自局部 RNG，可用 seed 重现。新 Lab 候选分别采用前景 median/IQR 的有界缩放、前景分位数映射；补齐 none 基线并锁分揭盲后，前者用于 WebUI 引导式默认，后者保留可选。

时间几何核对 `frames.timeline.json` 的源 SHA、逐帧 SHA、原整数 PTS/timeBase/sourceFrameIndex。双向唯一几何关联不等同身份识别；切镜标记、分段、源索引缺口、大 PTS 缺口、漏脸、重叠重复或歧义均断轨。额外使用保守缩略图强变化守卫，阈值记录在报告且不当作切镜真值。中心/尺度采用真实时间上的短窗口局部线性拟合，每帧位移不超过脸尺度 2.5%、log-scale 修正不超过 0.025，再乘用户强度。原 68 点完全不改；通过整体仿射修正预测坐标，保留眼嘴表情和旋转。

没有可信时间清单时保持旧几何且关闭运动估计，明确记录 missing-timeline，不猜 FPS；用户请求几何稳定时直接提示重新提帧。运动量基于已关联同轨中心的 pixels/second 和源帧 durationPts，缺 duration 使用该轨相邻真实时间间隔并记录。当前时序常数经过已知运动、VFR、跳帧、交叉、漏脸和切镜的合成契约测试，不据此宣称真实复杂短片质量已验证。

距离场候选按脸尺寸归一羽化宽度，只收缩现有前景，不填遮挡洞。多频候选按 Laplacian 金字塔混合，并保持遮罩零区域与原图逐像素相等；不允许羽化或多频传播污染眼镜和遮挡外部区域。旧侵蚀、膨胀、高斯和 seamless 模式保留。

## 受限会话读取与恢复

新会话只写 schema 2 基础数据，不保存 Python Frame/config 对象。旧会话经 `core.safe_pickle` 的 session profile 仅将固定 Frame/FrameInfo/MergerConfigMasked 与路径构造记录解释成惰性数据；不导入、调用或构造原类，再经过明确 schema 校验和受信构造器恢复。未知类被拒绝且原件保留；旧文件在新会话保存前备份为 SHA 命名 `.bak`。保存采用独立 pending 文件、fsync 与原子替换。

会话恢复总是采用新预检的 FrameInfo。原帧/aligned SHA 或模型迭代变化会重算，不使用历史缓存遮罩或仿射；旧会话缺 SHA 也重算。复核遮罩、几何强度和启用的超分模型需与当前预检方案一致，不能由会话跳过严格检查。旧 FaceEnhancer/SwinIR 会话若启用了 SR，会明确提示选择新模型并保留原件；SR=0 的旧会话保留其余配置，避免隐式复用旧增强链。

## 实际验证与限制

- `tools/tests/test_merge_quality_upgrade.py`：25 项通过，覆盖前景背景隔离、空/低方差颜色、局部 RNG、两候选、遮挡洞、不同画布融合、真实不均匀 PTS 运动、有限稳定、断轨/歧义、SHA/仿射变更拒绝、数据会话与旧会话惰性转换/备份。
- `workspace/.vision-evaluation/merge-quality-20261007-v1/acceptance.json`：公开采访原视频只读，52 秒处截取 0.32 秒 FFV1/RGB8 测试片，8 帧实际 PTS 提取，8 张实际 YOLO26/TUFA WF JPEG，原已保存 interview ME 检查点独立副本；RCT、两 Lab、两融合、有限几何及复核遮罩共 7 组 CLI 实际预测与合成，每组 8 张图和 mask，均 complete。原检查点与测试原帧 SHA 保留，未训练。原/clip/checkpoint SHA 和每组输出均记录。
- `legacy-parity.json`：两个实际 ME 预测帧在 learned mask / color none / geometry off / legacy blend 下，与仓库 HEAD 原合成代码输出逐字节一致。一次直接适配器 QA 缺 nn 主进程初始化而失败，修正验证脚本初始化后完成；不把该脚本错误归因于合成算法。
- `blind/manifest.json`：初轮 3 组随机 A/B/C 色彩/融合/几何参考，24 张 PNG 与 3 张时序 storyboard 均有 SHA，GPT-6 Astra low 已先锁分后揭盲，结果保留于同目录 `locked-scores.json` / `review.md`。融合三候选没有可靠视觉分离，几何亦未取得足够时间稳定证据，因此维持 `legacy` / `off`；初轮颜色缺 none 基线，由下述追加评测补齐。

这一个 0.32 秒采访片仅证明实际链路和有限样例的颜色比较，不能代表长片时间稳定、强遮挡、眼镜、多人交叉、HEAD、颜色标定或最终身份保真。依据追加评分只更新 WebUI 引导式颜色默认；融合与几何保持原默认，其他颜色方案继续可选。通用 XSeg 资源和用户模型保持原样。

## 当前无颜色迁移默认的追加比较

最初三颜色候选只包含 RCT、robust-lab、Lab 分位数，无法据候选间排序直接替换当时 WebUI 的 `colorTransfer=none` 默认。为补齐真实基线，`tools/verify_color_default.py` 复用同一隔离 ME 检查点和原八帧输入，执行 none 的实际 CLI 合成；overlay、mask4、workers1、seed121、blur16 与原颜色比较完全一致。八张合成图、八张遮罩与 audit 均完整，检查点前后 SHA 一致，没有训练。

新范围为 `workspace/.vision-evaluation/merge-quality-20261007-v1/color-default-v2`。旧三方案的每张输出及源帧 SHA 经原 audit 复核后重用，与 none 构成四候选 A–D 新随机别名；八张逐帧盲图和一个按时间排序的 storyboard 都有 SHA，独立映射只存放在 blind 目录之外。GPT-6 Astra low 仅读取该 blind 目录的 manifest、八张逐帧图和 storyboard，已将 `locked-scores.json` 状态写为 `locked-before-reveal` 后，再由主任务读取外部 `blind-key.json` 揭盲。旧结果、低分与失败记录均保留。

| 已锁别名 | 揭盲方案 | 整体视觉自然度（0–5） | 解释 |
|---|---|---:|---|
| A | robust-lab | 2.0 | 对比度相对温和，略优于 B；肤色过渡与融合仍明显有缺陷。 |
| B | RCT | 1.9 | 与 A 接近，局部明暗和嘴唇较浓；A/B 是低置信近似并列。 |
| C | none | 1.5 | 颜色偏棕、低对比且平坦，眼嘴定义更弱。 |
| D | lab-quantile | 1.0 | 眉眼、鼻孔和口腔过黑，块状明暗与边缘条带突出。 |

八帧均记录 `A > B > C > D`，高度相关的八帧不作为八个独立统计样本。A/B 的 0.1 差值是主观细分，不表示统计精度。所有方案都存在明显额头多边形边界、面颊横带、过平滑纹理与周围皮肤不连续，不能把相对第一名写成最终视觉验收通过。时序稳定、身份、强遮挡和眼镜均不可据此定论，也没有准确率/AP/NME/IoU。

锁分揭盲文件 SHA256：`blind/manifest.json` 为 `7596a1ce7d7ea0555f24e932b90e482a039c054a14dc15b9f3220e167c552eb2`；`blind/locked-scores.json` 为 `ba49c7e4f87faccc3e1256326d2cfc575d371aab2617e873543797d8ba343b2a`；外部 `blind-key.json` 为 `5b2c4e3fd3369eda1047ab5efc6026a2319b4c732c638b3a05c9ebce4b4ab27f`。审核文字见同盲评目录 `review.md`；这些文件与原 acceptance 一起保留。

基于这次含当前 none 默认的比较，主任务已将 WebUI 引导式 `colorTransfer` 默认更新为 `robust-lab`；原生旧配置默认保持不变，RCT、关闭和 Lab 分位数继续可选。融合无可靠优胜，继续默认 `legacy`；几何仍默认 `off`。实际生产应用与超分两模型最终结论统一见 [全流程最终报告](ALL_PIPELINE_UPGRADE_20261007.md)。
