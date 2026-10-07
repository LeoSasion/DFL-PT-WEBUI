# 质量方案生产应用记录

日期：2026-10-06。响应用户“应用最佳方案，更新我们的各个环节”，在第一轮质量对照之后将有证据支持的流程接入本地工作台。本文记录实际行为和证据边界。第一轮原始分数、失败与验证计数保留在 [历史结果](NON_TRAINING_UPGRADE_RESULTS_20261006.md)，实施原则见 [当前方案](NON_TRAINING_UPGRADE_PLAN_20261006.md)。本轮没有训练、改动 ME 桥接或发布新发行包。

## 已应用的生产流程

| 环节 | 当前应用 | 原始数据与兼容约束 |
|---|---|---|
| 新人脸提取 | WebUI 默认 YOLO26s-face、TUFA、JPEG100；旧检测器可明确选择 | CLI FAN 默认保留；缺固定资产/依赖提前失败，不下载或静默回退；既有 aligned 不迁移 |
| 二维关键点 | 原生 TUFA68 使用现有 DFL 对齐；原生 TUFA98 独立审核 | 98 不截成 68，不写入旧 68 点字段，不自动生成深度或遮罩；保留来源与仿射矩阵 |
| HEAD | 真实 FAN3D 权重与原有 68 XY/HEAD 几何，另存 TUFA98 | FAN 接口没有输出深度；真实对照中三张 HEAD 的 source68/矩阵和原 FAN HEAD 完全一致 |
| 相似图 | 描述子召回后同尺度 SSIM 逐对复核，组内所有配对都过门槛 | complete-linkage 防止链式合并；每轮最多 500 图，配对比较每批最多 250 图；隔离可恢复 |
| 人物分组 | 原锁定 SFace 权重由 PyTorch 执行，保留原阈值与人工复核 | AdaFace/MagFace 已实跑，但没有已确认身份真值和可信质量赢家；不移植阈值或改默认 |
| 源帧修复 | 可选 SwinIR-L PSNR，新同尺寸 RGB8 PNG；Real-ESRGAN 可明确选择 | 单模型，不串联；原帧/元数据不覆盖，缺资源禁用或失败；不能保证身份与真实细节 |
| 修复后提取 | PNG 重提取至独立 `data_<side>/aligned_restored/<taskId>` | `source_map` 恢复原帧文件名、记录输入/输出 SHA；包括奇数宽度原画布，历史 aligned 保留 |
| 辅助遮罩 | BiSeNet 草稿、原图/边界/遮罩复核，再发布完整 aligned 副本 | 眼、口、耳等类别保留；不伪造人工 polygons；未选图片也独立复制；generic XSeg 保留 |
| 人脸集选择 | ME 训练可指定 SRC/DST 副本；合成可指定 DST 副本 | 必须位于当前 workspace 内并经内容/路径前检；拒绝外部、遍历或链接逃逸；不自动开始训练 |
| 抽帧与导出 | 原整数 PTS/time_base 清单、FFV1 RGB8/NUT 母版与 MP4 播放版 | 不虚构 VFR；“母版与播放版”产生四个视频与四份 `.media.json`；CRF0 仅说明 YUV 编码域 |

工作台将这些步骤集中到“质量方案”，按批次选择源帧修复或辅助遮罩。已保存遮罩副本的发布回执可以从历史重新读取，恢复训练入口；DST 回执另有合成配置入口。模型可用性、可读状态、批次范围和选择数量显示明确，加载失败可重试。预检仍负责核对实际资产和输入，界面可用提示不能替代严格内容校验。

2026-10-07 调用链复查补充：ME 训练读取副本内的 embedded `xseg_mask`；当前合成器只从所选 DST aligned 取源关键点，其 XSeg-dst 模式仍对原帧重新推理。因此合成入口和 DST 副本选择已经交付，但“合成直接采用已复核 BiSeNet 遮罩”尚未交付。相关旧代码、依赖和后续验收见 [迭代审计](LEGACY_ITERATION_AUDIT_20261007.md)。

## 质量选择依据与不能推断的结论

第一轮复现 H3CE 实际使用模型并对每类再运行两个候选；H3CE 没有的人物描述子、语义遮罩以 DFL 当前路径为参照。只读 H3CE 的授权难例与固定参考，没有复制 H3CE 自有 AGPL 应用源码、导入其虚拟环境或把其私有媒体纳入公开提交。

[Astra 盲评记录](VISION_ASTRA_VISUAL_REVIEW_20261006.md) 的关键点视觉指数为 TUFA98 80.7、Regression98 79.7、ORFormer98 77.6、TUFA68 73.4、FAN68 40.6；检测本组 YOLO26s 框覆盖/位置较好。数值与参考只适用于已锁定样本和实际预处理。TUFA98/Regression98 差值及旧检测轮廓差值的条件区间跨零，不由此声称通用检测/关键点赢家。采用 TUFA 原生 68 的生产方案另经过真实几何与独立视觉复查；没有将 98 点“兼容转换”为旧 68 点。

Astra low 审查和 H3CE 的人工探索性参考均不是专家 ground truth。视觉等级、轮廓距离、功能回读和专家准确率分别陈述；没有标准框真值时不报 AP/Recall，没有专家点真值时不报 NME，没有像素真值时不报 IoU/Dice，没有已确认人物标签时不报 FAR/TAR。旧 FFHQ 输入上 generic XSeg 的低分不能淘汰其 WF 生产路径。SwinIR 的退化恢复与清晰保持工程分也不能证明真实细节、身份保真或视频稳定。

新生产关键点和真实 DFL WF 遮罩又分别交给 GPT-6 Astra low 查阅实际输出，两个记录均为 `accept_with_review`。关键点点标会覆盖细边缘，HEAD 面部较小，眼镜/闭嘴等区域有不可判断项；12 张 WF 遮罩存在相近或近重复场景，缺足够真实遮挡，不外推为 12 个独立场景的泛化结果。遮罩结论只支持进入人工复核流程，不自动认定全部可用于训练/合成。

## 已完成的实际运行证据

证据均位于 ignored 本地目录。目录中的原始媒体、审核 PNG、哈希和失败产物保持私有，不加入源码发行包。

| 证据 | 实际范围与可以确认的结果 |
|---|---|
| `workspace/.vision-evaluation/production-landmarks-20261006-v2/acceptance.json` | 3 个独立源图，TUFA whole_face/FAN HEAD/TUFA HEAD 各 3 个完整 CLI 输出，另 1 个 PNG 来源映射容器副本；DFL68/原生98、仿射回投、侧车 SHA 和旧 HEAD 完全等价通过 |
| `workspace/.vision-evaluation/production-landmarks-20261006-v2/astra-production-review.json` | 实际查看 6 张 audit 与对应 6 张 aligned JPEG，独立视觉检查；不提供专家准确率 |
| `workspace/.vision-evaluation/mask-assist/wf-twelve-v1/functional-summary.json`、`draft/report.json` | 12 张真实 DFL WF 输入的 BiSeNet 草稿；源像素/元数据保留，人工 polygons 未伪造，IoU/Dice 为空；点窗口覆盖是预测几何信号 |
| `workspace/.vision-evaluation/mask-assist/wf-twelve-v1/astra-production-review.json` | 查看全部 12 张生产预览，眼嘴前景和边界人工审查；`accept_with_review`，真实遮挡维度不足保留 null |
| `workspace/.vision-evaluation/restoration/restored-extraction-20261006/evidence.json` | 官方示例经真实 SwinIR PNG 修复及完整重提取，发布 1 张 512 aligned；259×194 源画布完整保留，68/98/矩阵回读、原 JPEG 帧名与 SHA 来源对应，原帧/原 aligned 保留 |

关键点生产 v2 的 PNG 仅用于来源映射容器副本检查，不能当作实际模型修复证据；真正 SwinIR 链另有上表的 `restored-extraction` 记录。其同画布验证与 immutable manifest 保持严格，没有放宽校验来接受裁边。

历史失败保留：关键点错误 RGB 和错误解码两种探索运行仍为 rejected；生产关键点 v1 的验收脚本误用 float64 重建旧 float32 几何所致失败仍保留。真实恢复重提取首次遇到奇数列被旧 `cut_odd_image` 裁掉，严格发布检查拒绝，失败写入原 pending 目录；随后仅让已校验 `source_map` 的输入保留完整画布，普通提取旧策略未变，修复后重新实跑并发布。旧失败不改为成功，原产物不覆盖。

## 当前验证状态与最终补录

以下计数是已完成运行的阶段记录，不相加、不冒充最新总数：

- 第一轮历史回归：Node 324 项/322 通过/2 既有跳过，Python 79 项全通过，原 UI 与媒体检查；原始记录保留在历史报告。
- 应用阶段 Python 最终全量 190 项全通过、25 个子案例通过；包括奇数画布、真实 FFmpeg、模型资产和新原生流程。locale/torchvision 的五条上游弃用警告保留，没有隐式下载。
- 应用阶段先完成 Node 339 项/337 通过/2 既有显式跳过；最终全量为 351 项/349 通过/2 既有显式跳过/0 失败。Windows 文件符号链接权限和平台专属项保持原跳过原因。最终日志为 ignored `.validation/quality-all-verified.out`。
- 最新获准 Playwright/headless Edge 界面 QA 使用独立 `127.0.0.1:4275`，桌面 1440×1000 与紧凑 1100×800；内置浏览器因已有 Windows 时钟断言故障使用获准替代。页面身份、非空、框架错误层、控制台、真实控件交互与截图检查通过，无 HTTP 错误、浏览器控制台错误或警告。
- 交互覆盖：推荐 YOLO26s/TUFA/JPEG100 → 实际异步 SwinIR 与 259×194 PNG → 修复记录预填；三图遮罩预览 → 复核勾选守卫 → 幂等完整副本与回执恢复 → 训练人脸集预填；原生 98 点图层实际显示 98 个点；四视频/四记录导出前检；页面往返历史恢复和紧凑视窗无横向溢出。
- 截图和机器记录位于 `C:/Users/Administrator/.codex/visualizations/2026/10/05/01a10c3e-92d7-7d23-90e3-7b5087ba3aa3/quality-applied-20261006`，`checks.json` 记录十组通过项。脚本在独立系统临时目录；没有使用 HTML 报告代替功能。测试只修改隔离素材副本，原工作台服务和用户工作区未被重启或写入。截图中的界面与原生元数据是真的，模型效果的准确率仍由前述独立视觉协议限定。

已通过 Vite 构建、版本和 dist 新鲜度检查。共用 Windows 终止器在 taskkill 前后读取 PID 与创建时间，确认启动器和解释器均退出；取消、超时和响应超限三条实际异常路径共六个 PID 的退出已验证。终止不能确认时保留占用状态并报告失败，不把未知状态写成成功取消。排队/运行前取消、关闭等待清理和关闭期间拒绝新操作均已回归。已有 CPU aligned 审计可与写独立源帧副本的修复共存；GPU 人物分析和训练继续互斥。

推荐双输出在独立离线 VFR 夹具实际执行四次编码，生成两个 NUT、两个 MP4 和四份记录。四个输出的 PTS 最大误差均为零，两个 NUT 的 RGB 回读一致；带音轨母版原包/解码时间检查通过。MP4 仍记录 `rgbReadbackExact:false`，不写成 RGB 无损。验证记录保存在 `C:/Users/Administrator/AppData/Local/Temp/dflsn-webui-tests-quality-20261006/repo/workspace/result*.media.json`，实际执行日志在 ignored `.validation/quality-encode-verified.out`。通用 XSeg 的现有权重再次与发行锁定 SHA 核对一致。

阶段通过不等于已经发布，也不认证样本外的最终视觉质量。

## 仍未交付或仍需新范围的事项

- 几何时间稳定、双向唯一关联及切镜/缺口/歧义断轨、真实短片眼嘴表情保护、颜色与遮罩边界时序验收仍待实现/复核。真实 PTS 清单已经接入，但它不等于时间平滑已经实现。
- SAM2 视频传播、跨帧辅助标注与人工修正闭环仍是后续能力；当前 SAM 候选实跑不等于生产视频功能。
- 人物身份真值、模型专属阈值校准、遮挡真值及人工像素参考不足；保留 SFace/generic XSeg 和复核流程，不制造质量赢家。
- 媒体证据限 FFmpeg 解码 RGB8。HDR/高位深未支持输入明确拒绝；RGB 母版不恢复源视频已有损失，也不认证源色准。音轨按原包或 PCM 转换后的实际协议验证，不泛称全链路 AAC/RGB 无损。
- 本轮不训练，不启动长期训练、旧 12 小时目标或双 GPU 验收。未来若改动 ME 桥接或开展训练验收，须遵守 [一小时内短训范围](ME_SHORT_RUN_ACCEPTANCE.md)；正式视觉质量/身份保真另需明确任务范围。
- 新候选权重、源码许可与依赖的正式发行准入、源码/便携包资源一致性、离线可用性与回退仍另行验证。现有 generic XSeg 发行资源保持锁定；本机 ignored 资产已可运行不能写成已加入公共包。代码未据此自动提交、推送或发布。
