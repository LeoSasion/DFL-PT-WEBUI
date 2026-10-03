# ME 训练能力验证（2026-10-02）

## 当前验收范围（2026-10-04）

用户已将开发验收调整为小参数、小模型的短时验证，**禁止长时间训练，包含保存、停止和续训在内的训练验收总墙钟预算最多一小时**。当前只观察损失/固定样本指标的收敛趋势，保留有限数值、实际训练、保存、停止、续训与预测的功能证据；开发阶段降低画质要求，正式案例、身份自然度和成片视觉质量以后另行处理。当前流程见 [ME 短时开发验收](ME_SHORT_RUN_ACCEPTANCE.md)。

双 GPU 暂不测试，保留现有初步实现和“真实双卡未验证”的状态，等待开源用户反馈后再决定后续验证。**12 小时真人长训、最终视频人工画质、DeepFaceLive 正式案例及真实双卡验收均不是当前开发或发布门槛。** 不为补足这些历史目标延长短训或重启原访谈模型。

下文是历史功能验证及当时的验收范围。既有 `longRunEvidenceQualified=false`、视觉未通过和双卡 `partial/skipped` 等结果继续保留；范围变更不把历史失败改成通过，也不把历史长训当作本轮短时验收。历史长训工具的 12 小时阈值保持原义，相关报告只作可选参考。

## 历史验证记录

本记录对应 PyTorch ME 后端和高级 WebUI；前半部分记录已完成的高级向导基线，后文追加独立训练、备份、统一评测、性能及验收工具的后续工程证据。使用的 aligned 数据均为人工合成素材。完整能力、配置命令、严格恢复、预训练转微调和 TF 导入说明见 [ME 后端说明](../_internal/DeepFaceLab/ME_README.md)。后端与发布证据在本机忽略目录 `.runtime/verification`；前一轮浏览器脚本、截图及 JSON 在 `%TEMP%/dfl-pt-qa`，未提交人工数据或临时产物。

## 高级 WebUI 验收

完整 45 项配置、分组预设、严格续训差异、阶段转换、原生初始化和旧 TF 导入已接入。Chrome / Playwright 实际访问 `http://127.0.0.1:4173`；Browser 插件未提供，使用已有 Playwright，无新依赖安装。桌面 1440×1000、窄窗口 900×1000、手机 390×844 通过布局检查，页面非空、身份正确、无框架错误层，目标操作后检查真实任务和检查点，页面/控制台错误为 0。中文与英文表单均已核对。

| 检查 | 已核实结果 | 证据 |
| --- | --- | --- |
| Python 全套 | 236 passed、17 现有 warnings、0 failed/skip，含 3 项实际 CUDA | `me-advanced-release-gates.log` |
| 隔离 Web 全套 | 194 项：193 passed、0 failed、1 项 Windows 符号链接 EPERM 跳过 | `me-advanced-release-gates-continuation.log` |
| 发布检查 | Ed25519 2 passed、Sites 4 passed；runtime/native smoke 0 failed，50 个注册 CLI 参数通过；生产构建、freshness、loopback 5 个引用资源有效 | `me-advanced-cli-smoke.log`、`me-advanced-release-gates-continuation.log` |
| Web 采样检查 | 输出 SRC/DST 增强与遮罩网格和 Web 预览，不创建检查点；之后同名新模型能正式训练 | `%TEMP%/dfl-pt-qa/me-advanced-webui.json` |
| Web 联合训练与续训 | GPU 0、DF-ud/64、ae/e/d/mask 32/16/16/16、batch 2；RG/FP16、CPU 优化器、GAN/TrueFace、眼嘴优先、CT mix、每侧 worker 1、回放一起启用；SRC PAK/DST ZIP 均含中文和空格路径。save ACK 4，close 9 → 新进程续训 close 14；lr 5e-5 → 2e-5；exit 0/safe-stop | 同上，`me-advanced-resume-diff.png` |
| 严格错误处理 | CPU+FP16、重复 GPU、LIAE+TrueFace、结构修改、缺优化器重置、预训练阶段切换缺数据重置均拒绝；拒绝前后检查点字节一致 | 同上 |
| 预训练转微调 | 专用 8 张 aligned、预训练自动保存停止在 3；允许配置变更并重置数据状态，微调停止在 5、pretrain=false/random_warp=false | 同上，`me-pretrain-confirm.png` |
| 原生权重初始化 | 选择已保存来源，结构自动带出并锁定；新模型从独立进度训练到 2 | 同上 |
| 旧 TF Web 导入 | counted tensor records、中文路径及前缀；导入新目录并被模型发现，iteration 0 → 实际 GPU 续训 2 | 同上，`me-tf-import-confirm.png` |
| 导入后预测 | 加载已训练 2 次的导入模型，CPU CLI 生成 swap/两侧 mask/aligned composite；原始预测形状为 1×3×64×64 与两张 1×1×64×64 mask，全部有限 | `%TEMP%/dfl-pt-qa/me-imported-prediction.json`、对应输出目录 |
| 最终生产页面 | 刷新新构建，桌面与手机首屏、模型字段间距、焦点、折叠组无效输入展开定位、横向溢出检查通过；page/error/warning 均为 0 | `%TEMP%/dfl-pt-qa/me-final-render.json`、`me-final-fields-desktop.png`、`me-final-fields-mobile.png` |

自动停止达到上限后，“继续训练”默认清除旧的自动停止开关；不会立刻因已达到上限而退出。结构在续训和原生初始化时保持只读，确认页显示实际服务端差异。任务结束后已恢复 default 项目，未留下运行中验证任务；原 `C:/Users/Administrator/Documents/ChatGPT/DFL-WEBUI` 工作树未修改。

发布门槛分两段完成：首段完整 Python 全套通过后，native smoke 因清单未收录新导入命令的 `--source` 失败。修复 `tools/smoke_test.py`，同时检查 `web-train` 和 `import-tf --help`，保留未知参数拒绝。后段使用 `-SkipLauncherTests -SkipPythonTests`，不重复未变更且已通过的 Python 后端，继续运行 smoke、Web、构建、Sites、freshness 和 HTTP smoke，全部通过。Launcher 源码未变更，本轮未重跑其独立测试。

## 前一阶段后端证据

| 检查 | 已核实结果 | 证据文件 |
| --- | --- | --- |
| Python 后端全套 | 231 passed、17 warnings、0 failed/skip；3 项 CUDA 实际通过 | `me-release-gates.log` |
| 隔离 Web 测试 | 172 passed、0 failed、1 项已知 Windows 跳过 | `me-release-gates.log` |
| Runtime / vision / 签名 / 版本 / i18n | smoke 0 failed，Ed25519 2 passed，其余通过 | `me-release-gates.log` |
| 已有前端构建检查 | dist freshness 通过，loopback HTTP smoke 的 5 个引用资源有效 | `me-release-gates.log` |
| 单 GPU 联合训练 | DF-ud，RG/FP16/GAN/TrueFace、CPU optimizer/dropout、yaw、增强、颜色迁移、worker、replay 联合启用；停止 12 → 续训 14 iteration，21 optimizer_updates | `me-capabilities-live.log`、`me-capabilities-live/result.json`、`all-features.json`、`train.log`、`resume.log`、ACK/control/model |
| 实际 WebUI | GPU 0、默认 liae-ud/FP32、128/batch 4；save ACK 23、close ACK 32、跨进程续训后 close ACK 44；两个 job exit 0/safe-stop、requestedAt 匹配、预览有效、page/console errors 0，default 项目恢复 | `me-complete-webui.json`、`me-complete-webui.log` |
| 传统 main.py CLI | 人工小模型、RG/CPU，新建到 1，第二次自动 resume 并由 target-iterations 保存到 2，成功退出 | `me-native-cli.log` |
| 旧原生模型兼容 | 当前代码加载旧 liae-ud/128 检查点，iteration/optimizer_updates=2002，CPU 预测有限 | `me-compatibility.json`、`me-compatibility.log` |
| 新 DF 模型 DFM | 已训练 DF-ud/RG/FP16/GAN/TrueFace 模型导出，ONNX Runtime CPU dynamic batch=2；目标掩码/swap/源掩码 maxabs 为 2.98e-7 / 8.94e-8 / 2.38e-7 | `me-compatibility.json`、`me-compatibility.log` |

全套覆盖全部 32 种架构组合、RG 重算/梯度、真实对抗更新、状态精确恢复、配置变更、旧字段兼容、损坏状态拒绝、TF 权重导入、ONNX 对照，以及 CUDA AMP 溢出重试/失败恢复/scaler 检查。此前 222 项全套、19/23 项局部边界记录不能累计为最终总数。

本轮 gate 使用 `-SkipLauncherTests -SkipBuild`，没有重跑 Launcher Pester/Node，也没有重新构建前端。启动器和前端源码未变更，已有构建的 freshness 和 HTTP smoke 重新通过。首次启动器/build、合成/MP4 和历史 Web 记录见 [首轮集成验证](VALIDATION.md)，不计为本轮重跑。

## RG 受控单卡短基准

NVIDIA RTX PRO 6000 Blackwell、liae-ud、128、ae/e/d/mask 256/64/64/22、batch 4、FP32，关闭 GAN/TrueFace，`cudnn.deterministic=True`。两次使用相同初始权重，训练 5 步，统计后 3 步中位耗时。

| 模式 | CUDA 峰值 allocated bytes | 中位单步耗时 |
| --- | ---: | ---: |
| 非 RG | 1,826,172,416 | 54.9368 ms |
| RG | 1,564,939,776 | 66.5125 ms |

该配置显存峰值降低 14.3049%，中位耗时增加 21.0709%；五步后权重 SHA-256 一致、最大绝对差为 0。证据：`me-rg-benchmark.json`。它只证明此配置的短程收益和代价，不保证其他架构、精度、batch、对抗配置或 GPU 得到相同比例。

## 历史能力边界与可选后续

RG 保持拓扑和权重键；AMP 保持 FP32 主权重/优化器状态。原生 version 1 检查点有兼容路径。TF 导入只迁移网络：卷积 HWIO→OIHW，Dense 保留 DFL [input, output]；旧 RG/非 RG 同拓扑，但不恢复旧优化器、判别器、RNG、数据状态或迭代，不宣称旧训练器逐步等价。

多 GPU 调度和界面选择已实现，真实双卡 scatter/reduce、稳定性、吞吐和内存分配未验收；当前不再安排双卡测试，等待开源反馈。长期真人训练、最终换脸质量和 DeepFaceLive 正式案例仍未验收，属于日后另行确定范围的可选后续，不阻塞当前开发或发布。姿态评测现支持目录、PAK、ZIP、自定义和预训练数据集；样本以数据集指纹与包内成员名固定，跨快照比较仍需同一 manifest。Web 目标迭代默认只估算进度，启用“达到目标后自动保存并停止”才传入实际停止阈值。

## 后续工程与本机证据（2026-10-02）

| 能力 | 历史已验证范围 | 可选现场验证（当前不执行、不作为发布门槛） |
| --- | --- | --- |
| 独立 ME 训练监督与 Web 重连 | 自动测试覆盖监督进程在 Web runner 释放后继续运行、日志偏移续读、Web 服务重新附着、模型/GPU 锁保留及安全停止 ACK 身份；训练心跳区分正常、未知和疑似停滞，不自动强杀 | 真人数据连续长训期间的服务重启及长时间无进度故障注入 |
| 备份与恢复 | 自动备份默认每 1000 iteration 建立校验清单并保留最近 3 代；手动备份不参与轮换。Python 与 Web 测试覆盖校验、恢复前留存当前版本和运行时恢复锁。旧版备份按内容重验 | 真人长训中的多代备份空间占用、人工选择恢复点后的质量复核 |
| 统一姿态评测 | 人工 DFL aligned 目录、人物子目录、PAK、ZIP 和预训练同集的内容身份、固定成员、严格变更拒绝与实际 `read_aligned` 读取通过；引导式 Web 预检为自定义 PAK/ZIP 和预训练建立 manifest。非默认素材不提供错误的默认姿态图谱跳转 | 真人 SRC/DST 姿态覆盖及跨快照质量解释 |
| 阶段性能画像 | 同一已解码 batch 的 RG × FP16 × CPU 优化器八种组合在本机单卡短测，含阶段同步计时、allocated/reserved 峰值、优化器状态字节数和可追溯输入指纹；数值与条件见 [ME 性能对比](ME_PERFORMANCE_PROFILE.md) | 目标真人数据、分辨率和 batch 下的吞吐与画质取舍 |
| 长训、DFM 与视频技术审计 | [历史长训审计工具](ME_LONGRUN_ACCEPTANCE.md) 的 5 项人工合成素材测试通过：连续 loss history、离线间隔不计入有效时长、固定样本有限预测、临时 DFM 的 ONNX/PyTorch 对照，以及合成 MP4 的完整解码。短测报告不会标为真人长训或视觉质量合格 | 旧 12 小时真人有效训练、成片视觉检查与 DeepFaceLive 正式案例要求已移出当前范围；如以后明确开展正式案例，再单独制定验收 |
| 双卡 | [暂缓的 ME 双卡验收](ME_DUAL_GPU_ACCEPTANCE.md) 保留固定 batch、实际 scatter/reduce hook、数值门槛、保存/续训和双卡必需模式。本机 `torch.cuda.device_count()` 为 1，历史单卡基线与联合功能短测已完成；双卡结果明确为 partial/skipped | 当前不做真实双卡验证，保留初步代码并等待开源反馈 |

真实 GPU Web 端到端重连验收使用 `me-verification` 项目里的人工 PAK/ZIP 人脸集与 `goal-reconnect-1790946671936` 模型：Web 预检建立评测 manifest，训练至 iteration 12 后以同一 Python PID 跨 `local-manager restart` 继续到 51；重连事件、健康心跳、评测快照 `iter-00000053-9144d2d8`、保存 ACK、iteration 77 手动备份、安全停止于 107、续训并自动停止于 109 均通过。随后通过模型恢复 API 回退到 77，再用自动创建的手动回滚恢复至 109；检查点和 109 条损失记录均与最终迭代一致。本轮修复了 Windows 虚拟环境启动器与真实 Trainer PID 不同、启动器提前退出后的跟踪，以及心跳文件读取与原子替换的瞬时共享冲突。忽略目录 `workspaces/me-verification/qa-07a49b681798/goal-reconnect-result.json` 保存了各项机器记录；测试结束已切回默认项目。

对该模型运行短程[长训技术审计](ME_LONGRUN_ACCEPTANCE.md)，检查点、连续 loss、固定 PAK/ZIP 样本、预测及临时 DFM 的 ONNX/PyTorch 数值对照均为 passed；报告在同一忽略目录的 `goal-longrun-audit.json`，`technicalPassed=true`、`longRunEvidenceQualified=false`、`visualQualityAccepted=false`、`deepFaceLiveAccepted=false`。没有把这 109 次人工素材迭代解释为真人长训或画质合格。

另对 `pipeline-verification` 既有人工合成夹具的同一 `web-gpu-128` ME 检查点（iteration 2002，SHA256 `83cf0c0d3990804b59791c657fddcab843e99c082574bc5d49ae99083f694242`）做了隔离离线复跑：固定引导参数下的 CPU 单帧合成、双 MP4 编码和 DFM 导出均以退出码 0 完成。RGB 合成帧、`result.mp4` 和 DFM 与原产物的 SHA256 完全相同；mask 帧最大相差 1 个 8-bit 像素值（归一化后为 1/255），因此 `result_mask.mp4` 并非逐字节一致。现有和新导出的 DFM 都用同一固定 aligned 图像做动态 batch=2 的 ONNX Runtime/PyTorch 对照，最大绝对误差为 `1.79e-6`；两个新 MP4 均经 ffprobe 确认为 128×128 H.264、时长约 0.042 秒，并通过 FFmpeg 全片解码。输入/输出哈希、参数、退出日志和复跑脚本保存在忽略目录 `workspaces/pipeline-verification/acceptance-20261002/` 的 `evidence.json` 等文件中。旧 metadata 缺少后来新增的 21 个默认配置字段；修复长训审计的兼容比较后，该旧模型的检查点、固定样本、预测、DFM 和视频检查均通过。旧项目未保存 `loss-history.jsonl`，所以完整长训技术结论仍为未通过，详情见同目录 `legacy-audit.json`。该夹具只有 1 帧人工合成素材；这只验证同一模型的技术通路，不代表真人长训、人工视觉质量或 DeepFaceLive 实机验收。

在运行中的中文 WebUI 复查了模型恢复点入口：1440×900 桌面可查看迭代 109/77 恢复点并打开恢复确认，按 Escape 关闭确认；390×844 手机视口的抽屉无横向溢出，控制台与页面错误为 0。截图和运行结果保存在忽略目录 `workspaces/me-verification/qa-07a49b681798/ui-recovery-*`；测试后已切回默认项目，未实际再次恢复模型。

整合后的 `tools/verify-release.ps1` 发布门禁通过：Python **252 passed、2 skipped**（真实双卡项），隔离 WebUI **209 passed、1 项 Windows symlink 权限跳过**，launcher bridge 7 passed、Pester 43 passed，依赖和路由 smoke、i18n、生产构建、Sites 及 dist smoke 均通过。长训工具兼容修复后的针对性测试 6 passed。

历史工作已找到旧参考工作区中此前选定的真人访谈素材，并在本仓库启动独立的 [ME 真人访谈阶段验收记录](ME_REAL_MEDIA_ACCEPTANCE.md)：真人训练、Web 重连、备份、早期视频与官方 DeepFaceLive 模型加载均有现场证据。旧 12 小时有效训练、最终视频人工质量检查和正式 DeepFaceLive 现场要求保留为历史未通过/未验证结论，已移出当前开发及发布范围；不继续为这些目标运行长训。

## 复现入口

以下为历史完整回归入口。当前开发只运行变更所需的检查，并遵守一小时短训预算及暂缓双卡的范围；Python 全套包含真实 CUDA 训练及依硬件启用的双卡测试，会使用 GPU，不能将整套命令自动视为本轮必跑项：

```powershell
.\.venv\Scripts\python.exe -m pytest _internal\DeepFaceLab\tests -q
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py train --help
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py import-tf --help
powershell -NoProfile -ExecutionPolicy Bypass -File tools\verify-release.ps1 -SkipLauncherTests
```

启动器或前端变更后，应执行覆盖相关变更的检查，并保持一小时短训预算及双卡暂缓范围。先前独立 TF 数值对照见 [历史 ME 验证](../_internal/DeepFaceLab/ME_VALIDATION.md)，仅覆盖当时列出的原型架构、梯度和损失。
