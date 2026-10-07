# 全流程升级应用与验收

日期：2026-10-07。承接用户批准的 [旧代码审计](LEGACY_ITERATION_AUDIT_20261007.md) 全部整改项。实现仅在本仓库；D:/H3CE_v2 作为只读经验参考，未引入其应用源码或运行依赖。保持 PyTorch、ME、原通用 XSeg 和本地可恢复工作流。本轮没有 ME 训练、训练桥改动或双 GPU 验收。

## 已采用的方案

| 环节 | 当前选择与行为 | 依据与实际限制 |
|---|---|---|
| 检测 / 标注 / 相似复核 | 保持上轮 YOLO26s / TUFA 68 与独立 98 点审查、SFace PyTorch / SSIM；现在纳入固定资源安装与干净环境验收 | 延续上轮评分；没有新的标注准确率真值 |
| 最佳训练人脸 | 原生默认 Best Training Faceset：YOLO/TUFA 坏图检查 → 既有综合质量底线 → TUFA98 几何状态、SFace、可靠时间的全量多样性选择 | 数量是上限，低增益提前停止，人工确认同人参考。唯一 Efficient-FIQA 候选在 64 项预锁盲评未显示明显收益，保留既有综合分。旧 `quality-coverage` 降为明确标注的旧批次排序，原恢复记录保留 |
| Aligned 人脸修复 | **MambaIRv2 Large x4 默认，Real-ESRGAN x4plus 可选；两者整幅 FP32** | 最终 v2 三张 actual aligned 盲评整体 3.5 并列，保真维度 Mamba 4.5 / Real 4。此为官方 classic-SR Large 权重，不能冒充官方真实退化修复专用权重 |
| ME 预测脸增强 | **MambaIRv2 默认，Real-ESRGAN 可选，强度默认 0** | 最终 v2 三张 actual ME 预测脸盲评 Mamba 2 / Real 1.5；Real 更锐却有未经输入支持的眼鼻牙重描。按保真优先选择，不能宣称最终身份准确性 |
| 旧增强链 | 旧 `FaceEnhancer.py` / `.npy`、懒导出、调用和发行项移除；此链不提供 SwinIR | 用户指定替代方案。上轮源帧修复是独立的输入域和资源组；其原有 SwinIR 可选项不是 FaceEnhancer 替代链 |
| 颜色迁移 | 引导式合成默认 `robust-lab`；RCT / 关闭 / Lab 分位数可选 | 补充当前关闭基线的 8 帧真实 CLI 合成后，再由 Astra 盲评，稳健 Lab > RCT > 关闭 > 分位数；前两者区分置信度低，所有样例仍有明显融合缺陷 |
| 镜头检测 | FFmpeg 默认，Adaptive 与官方 TransNetV2 PyTorch 为可选候选 | 两段有限素材盲评 FFmpeg=Adaptive > TransNet。修正 TransNet first-new-frame 约定后再盲评；渐亮误报仍保留。旧 v1 早一帧失败没有改写为通过 |
| 时间与几何 | 源 SHA / 原帧 SHA / 整数 PTS / timeBase，唯一关联，镜头/缺口/歧义断轨；中心尺度稳定默认关闭 | 真实 VFR、分段、边界和有限运动契约验证；只对整体仿射修正，不平滑眼睛或嘴形。0.32 秒采访不足以证明长片稳定 |
| 边缘融合 | 原融合默认，距离场 / 多频候选保留 | 盲评没有可靠优胜者，不强行替换。遮罩零区保持原图，保留遮挡洞 |
| 已复核 DST 遮罩 | 新模式 10 直接读取复核遮罩，11 与预测脸遮罩相乘 | 必须绑定 aligned 与原帧 SHA、68 点摘要、画布及仿射；缺失/变化明确拒绝。训练可用副本不自动等同合成可用 |

详细分项、评分锁与局限见 [人脸增强替代](FACE_RESTORATION_REPLACEMENT_20261007.md)、[清晰度覆盖](FACE_QUALITY_COVERAGE_UPGRADE_20261007.md)、[合成升级](MERGE_QUALITY_UPGRADE_20261007.md)、[切镜比较](SCENE_MODEL_COMPARISON_20261007.md)。评分均为当前样例的视觉复核，不是准确率、AP、NME、IoU 或最终训练质量。

后续用户将“最佳人脸”收敛为训练集选择，当前实现与证据见 [Best Training Faceset](BEST_TRAINING_FACESET_20261007.md)、[原生操作流程](BEST_TRAINING_FACESET_WEBUI.md)、[唯一 FIQA 对照](EFFICIENT_FIQA_COMPARISON_20261007.md)。原 12 项整改的 v3/v4 发行包属于这项新需求之前的基线，不包含新选集模块。

## 数据、生命周期与依赖

排序与元数据工具采用独立批次、独立原件备份、SHA 计划、staging、预览、回执及恢复，不清空上一轮回收图。小目标数量也保留有效样本；质量筛选范围外保持原名和原字节。实际 8 张 aligned 副本已完成预览、提交、恢复，恢复后逐字节相等。

DFL APP15、PAK、权重、旧模型与合成会话改为有界数据解释器，只接受明确的基础数据及 NumPy 数据构造记录，不执行 pickle 的 import/call。旧格式兼容路径保留，未知对象拒绝并保留原件。原通用 XSeg 的 222 个数组及实际 ME 检查点的 56 个张量安全回读、CPU XSeg 推理和 SHA 不变证据已记录。

Python/FFmpeg helpers 统一等待进程树终止及流关闭后再完成取消。无法确认退出时保留当前服务的工作区占用与 staging；这不是服务异常退出后的跨重启持久锁保证。Subprocessor、RPC 与辅助 generator 增加有限初始化/响应/保存退出等待、私有响应管道、失联处理和 owner-only close；Merger 在 finally 逆序关闭 RPC 后释放模型。真实原生合成和真实 Windows venv 启动器/解释器停止验证均已运行。

Node、BAT 与 C# 使用相同 active project 环境；用户目录与继承 PATH 保留，Qt 固定 PySide6。原生 XSeg 编辑器在 offscreen 平台实际加载、使用 Qt 鼠标事件绘制、保存及像素/DFL 回读；不是人工可见窗口操作。`main --help` 不再提前加载神经网络；全局 FutureWarning 屏蔽删除。固定参数 subprocess 替代 ffmpeg-python，真实裁剪含音轨，失败保留旧成功输出；h5py 与 onnxruntime 移出生产层，OpenCV 单一变体。

生产、修复、验证、评测、场景五档传递依赖 SHA 锁和离线 wheelhouse/SBOM 已生成，生产 61、修复 78、验证 66、评测 89、场景 64 个包。固定生产资源、可选修复与场景共 9 组，逐项来源/许可/大小/哈希校验；缺资源明确不可用，不自动下载或静默回退。保留官方旧 API 所需的局部兼容，不把升级版本号当作质量证据。资源包为本地受控审核包，没有公开发布。

## 实际验证记录

- 新增及回归视觉/质量/合成/扫描/切镜/源绑定：90 passed，含 3 subtests；旧 torch/LPIPS API 两条告警如实保留。
- 辅助工具 / VFR 媒体 / 完整增强副本 / 合成 / Mamba / 切镜：77 passed，2 项训练相关用例未运行；补充尾 PTS 篡改拒绝后媒体 18 passed、增强副本 5 passed。
- 安全数据与 worker：最后 51 passed；此前组合含旧 ME 导入及合成 131 passed。质量标定完成后质量/事务另 38 passed、相似契约 19 passed；这些套件有重叠，不能相加当独立测试总数。
- `tools/verify_face_restoration.py`：最终整幅 FP32 的 Mamba 与 Real 各生成完整 8 图独立 aligned 副本（仅修复第 2 张），各完成 8 帧真实 ME 预测+原生合成及 8 mask；原检查点、aligned 和帧 SHA 全部不变。证据 `workspace/.vision-evaluation/merge-quality-20261007-v1/restoration-acceptance-v3/acceptance.json`。两链补充回归 48 passed。旧 v1/v2 证据与第一次脚本误读 provenance 字段的诊断保留。
- Mamba FP32 扫描：10 个边界/布局用例对照固定官方参考；完整 Large 网络 16×16 输入对照最大误差 1.10e-5。最初 TF32 开启时误差 0.001089 未通过，关闭 TF32 后通过，不降低原门槛；固定源码与权重字节不变。完整 512 aligned 与实际 128 ME 预测脸已另行实际推理。
- Node 隔离最终全套 361（359 passed / 2 skip，0 fail），结果记录于 `.validation/all-upgrade-20261007/node-verified.out.log`。最终模型默认更改另做合成/配置定向回归及实际界面验证；此前失败和中间运行日志保留。
- 用户已许可 Playwright：内置浏览器先前因 Windows 时钟断言重复退出。独立端口 4275、隔离素材完成 8 组界面操作：两模型及真实资源预检、质量范围/预览、恢复入口、ME 默认与复核 mask、真实场景检测、保存原帧区间/提帧/shot 传播、1100×800 布局及控制台健康。最终零 page/console error、零 4xx 请求；向导重开时旧步骤触发错误预检的实际问题已修复。桌面与紧凑截图/QA JSON 位于本机 visualizations 的 `all-upgrade-20261007`。

干净源码安装实际默认提取、BiSeNet / 修复、干净 Mamba CPU strict Large 推理、离线依赖检查以及便携包完整解包后的真实提取/遮罩/Qt 证据由发行验收文档记录。曾发现全局排除 tests 误删 NumPy 的生产 runtime sentinel，失败证据保留，已修正并重新验收。最终源码/便携包构建不自动发布、不更改用户媒体和模型。

本轮只证明这些功能与有限样例的评分依据。长片时序质量、更多身份/强遮挡、最终训练画质与双 GPU 仍按原开发范围保留未验证状态；原历史审计及失败门槛没有被重写。

## 后续 Best Training Faceset 验证

新核心及必要历史回归 64 passed，Efficient-FIQA 官方数值一致性、主进程依赖隔离和 worker 关闭 3 passed。Node 完整隔离回归 365 passed、2 个既有 Windows 文件 symlink skip；最终差量与两项独立复核问题另做定向测试，不把重叠套件相加。损坏内嵌 XSeg 不再卡住整批，身份参考变化使旧选集发布失效；两项均已独立复核。

用户许可的 Playwright 在独立端口 4295、真实模型与隔离 16 图副本完成 10 组实际操作：全量分析、未确认身份复核、参考变化阻止旧结果、重新选择、全量分类查看、预演、三目录发布、只预填 selected、撤回保留及原件 SHA 校验。结果 selected12/review1/rejected3；1440×1000、1100×800 均可操作，无 page/console error 或 warning、无 4xx/5xx。截图与 QA JSON 位于本机 visualizations 的 `best-training-faceset-20261007`，不进入发布包。没有启动训练，也没有在用户原 aligned 上执行筛选。
