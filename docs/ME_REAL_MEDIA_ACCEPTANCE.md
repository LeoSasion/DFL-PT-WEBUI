# ME 真人访谈阶段验收（2026-10-02）

## 当前范围（2026-10-04）

本页保留此前真人访谈训练、停止、技术审计与画质诊断的历史证据。当前用户要求**禁止长时间训练，开发验收使用小参数、小模型，总墙钟预算最多一小时（含保存、停止和续训），只观察收敛趋势**；具体流程见 [ME 短时开发验收](ME_SHORT_RUN_ACCEPTANCE.md)。开发阶段降低画质要求，正式案例与成片视觉质量以后另行确定范围；双 GPU 暂不测试，保留初步代码并等待开源反馈。

历史的 **12 小时门槛未通过、人工视觉未通过、旧监督退出未确认** 等结论和原始报告字段保持不变。它们描述当时的验收结果，**不再阻塞当前开发或发布**，也不会因新范围而改成通过。新的短时验收应在独立小模型上建立自己的证据，不重启本页冻结的 99214 轮模型，不执行文中的拟议长期续训或正式画质验收。

下文复现命令和后续建议均属历史/可选参考，不是当前任务的自动执行清单；如以后明确开展正式案例，再单独制定时长、素材和视觉标准。

## 历史现场记录

用户选择在 2026-10-02 16:33:34 UTC 提前结束训练并做阶段验收。Trainer 在第 **99214** 轮完成关闭保存，审计计入 **2.68517 小时**有效训练。隔离模型的固定样本、DFM、官方 DeepFaceLive 组件/后台/带兼容修正的 Qt GUI、100 帧合成、双 MP4 与六项技术审计检查均通过；**12 小时长训门槛未通过，整体验收未通过**。旧 sidecar 的状态写入失败导致监督退出未确认；DFL 离线合成可见贴片边界，人工视觉验收未通过。详见下文阶段报告。所有媒体、检查点、日志与临时验证脚本均位于忽略的 `workspaces/real-interview-me-acceptance/`，没有加入 Git。

素材来自旧参考工作区已选定的 `interview-practice` 项目。原片是 W!ZARD Radio Media 的[访谈视频](https://commons.wikimedia.org/wiki/File:Interview_with_Rebecca_Ferguson_-_Mission-_Impossible_-_Fallout.webm)；Wikimedia Commons 页面列出 CC BY 3.0，同时明确标记外部来源许可尚待复核。本轮仅做本地技术验收，不发布生成媒体。原有工作区保持不变；复制进新项目的 67 张 SRC、62 张 DST DFL aligned 人脸均由当前 PyTorch ME 读取成功。另复制了 100 帧目标序列与 4 秒目标视频；复制前后源视频和目标视频 SHA256 均相同。详细出处、归属及哈希保存在忽略目录的 `qa-20261002/source-provenance.json`。

Web 端启动了 `interview-me-128`：LIAE-UD、128 px、batch 4、FP16、GPU 0，任务 ID `20261002135224-ff15f576`。训练从 2026-10-02 13:52 UTC 开始；启动时心跳健康，迭代 8。随后在真人样本训练期间请求保存，迭代 927 的 ACK 成功；重启 Web 管理器后，Trainer PID 仍为 44624，迭代从 920 增至 961，任务重新附着。迭代 9099 的手动备份与自动轮换备份均已创建。机器记录分别为 `qa-20261002/start-training.json`、`live-reconnect.json` 和模型 `backups/`。

另将真实自动备份 `automatic/iter-00035000-1790952302119209100` 复制到隔离模型，先核对检查点、连续历史、固定样本预测与临时 DFM，再通过正式恢复函数从 9099 恢复至 35000。恢复动作自动生成 9099 的手动回滚备份；再从该回滚备份恢复至 9099，也通过迭代和历史末条校验。最后恢复回 35000，并在 GPU 0 上真实续训一轮至 **35001**：检查点和历史末条均为 35001，源备份哈希未变。全过程未写在线模型，证据见 `qa-20261002/automatic-backup-35000-audit.json`、`restore-rehearsal-result.json` 和 `restore-resume-one-step.log`。

原计划由独立监控在 `observedActiveHours` 达到 12 小时后请求关闭；该口径将超过 300 秒的离线间隔排除在有效训练时长之外。用户在不足 12 小时时改变目标，监控没有触发时长关闭。用户发出的 `close` 请求及当时 **2.68515 小时**的历史快照保存在不可变的 `qa-20261002/stage-stop-request.json`。Web 请求时间、Trainer `close` 完成 ACK 与 `finished` 心跳均指向本次请求和第 **99214** 轮，Trainer PID 44624 已退出。监控对这次“监控之外的提前停止”记录了异常状态，未将其伪装成 12 小时监控达标；记录见 `qa-20261002/longrun-monitor-status.json`。

运行中 Windows 曾对监督进程的状态文件重命名返回 `EPERM`。旧 sidecar 的写入队列随后停更，Web 健康状态显示“未知”，但 Trainer PID 44624 的迭代、心跳及自动备份持续推进；监督状态停更后，Web 手动保存仍在迭代 31377 收到匹配的 ACK，见 `qa-20261002/save-under-stale-supervisor.json`。提前关闭时 Trainer 的 ACK/心跳与最终检查点一致，但旧 sidecar 以 **exit 1** 结束且状态缺少 token；Web 在加载保守回收逻辑后将任务终结为 `cancelled`、`safe-stop-timeout`、exit 1。阶段报告因此保留 `supervisionUnconfirmed=true`，**没有声称旧监督进程安全退出**。相关代码加入了短暂文件重试、失败后写入队列恢复和保守终态回收。Windows 隔离集成测试还用实际文件锁制造状态文件重命名失败，验证管理器重建后附着到同一模拟 Trainer、解锁后的心跳与匹配 close ACK 以及终态元数据落盘；该测试不将旧任务追认为成功，也不替代真实训练的长时监督证据。

修复后的 sidecar 又在**隔离项目**从冻结的 99214 轮检查点实际续训至 **99267**（+53 轮）：save ACK 99261，close ACK 与 `finished` 心跳均为 99267；Web 任务为 `cancelled`/`safe-stop`/exit 0，sidecar 为 `exited`/exit 0 且 token 一致，日志无 `EPERM`。原项目模型、素材哈希和时间戳未变。证据见 `qa-20261002/new-sidecar-20261002165104-f311423f/status.json`。这验证修复后的短程保存/关闭，不会把旧任务的异常追认为成功，也不会补足 12 小时长训门槛。

真实人脸的固定样本评测在迭代 3391 生成快照 `iter-00003391-adf2b7e7`。阶段验收复用了同一 manifest `828f8e83c0f67ab09c157446` 的 **37 个样本**，与 99214 轮快照逐项比较：SRC `maskedMse` 均值 **0.003395→0.001406**，DST **0.002729→0.001204**；眼口误差下降、清晰度比值上升。指标描述训练进展，**不能替代成片视觉判断**。阶段对比见 `qa-20261002/stage-20261002T164657Z-75e865de/evaluation-comparison.json`。迭代 9099 的不可变手动备份曾用于早期端到端检查；当时有效训练仅 **0.22249 小时**，长训结论为 `false`，见 `qa-20261002/early-audit.json`。

在相同 37 个固定样本上，迭代 37000 的不可变自动备份与 3391/9099 快照身份一致；SRC `maskedMse` 均值依次为 0.003395/0.002010/0.001461，DST 为 0.002729/0.002014/0.000997。该中途 DST 数值小于 99214 的阶段数值，说明指标并非随迭代单调改善；不能用单一均值证明画质提升。中途证据见 `qa-20261002/midpoint-37000-immutable/midpoint-eval.json`。冻结的 9099 备份还曾完成隔离接线干跑：固定评测、DFM 导出、官方组件、100 帧合成、双 MP4 与六项技术审计检查均通过，12 小时门槛如实失败；见 `qa-20261002/dryrun-stage-wiring-20261002T144858Z-15555474/status.json`。

同一早期备份经 CPU 合成得到 100 帧真人访谈片段，再编码为 4 秒、1920×1080、H.264/AAC 视频和 H.264 遮罩视频，两者均通过 FFmpeg 全片解码。第 50/90 帧目视可见换脸区域模糊和边界，因此**早期成片未通过画质验收**；阶段最终模型也已重新合成并抽查，结论见下文。早期产物与日志为 `qa-20261002/early-result.mp4`、`early-result-mask.mp4`、`early-merge.log`、`early-encode-*.log`。

官方 [DeepFaceLive](https://github.com/iperov/DeepFaceLive) 源码提交 `fc7b787bda2b8c186e142c52857458eea3a935ed` 的 `DFMModelInitializer` 与 `DFMModel.convert` 已在本机实际加载 ME 导出的 DFM。隔离的 ONNX Runtime GPU 1.23.2 使用 CUDAExecutionProvider 处理迭代 9099 真人模型 DFM 与 62 张不同的真实 aligned 输入；所有输出形状正确、数值有限，最近一次模型调用中位耗时 4.07 ms、P95 7.03 ms。报告见 `qa-20261002/early-real-deepfacelive-cuda.json`。该数字只覆盖 DFM 模型调用，不包含 DeepFaceLive GUI、检测、对齐、合成、视频输入输出或相机，因此不能标为完整实时链路通过。此前人工模型的官方 DFMModel CPU/CUDA 连续帧报告见 `workspaces/pipeline-verification/acceptance-20261002/official-deepfacelive-*.json`。

另以官方组件直接处理真人目标 MP4 的相邻两帧：`VideoFilePlayer` 解码，`YoloV5Face` 每帧检出 1 脸，`FaceMesh` 给出 468 点，`FLandmarks2D` 对齐至 256×256，早期真人 DFM 换脸后由官方 `FaceMergerWorker` 合成 1080p 帧。两帧输出哈希不同且数值有限，证据和复现脚本见 `qa-20261002/official-deepfacelive-components.json` 与 `official-deepfacelive-component-video-smoke.py`。这项早期测试直接调用组件；阶段最终模型另完成了相同对齐参数的双帧、后台与 GUI 验证。

进一步运行了官方 **8 个后台 Worker 的真实多进程链**：`FileSource` → 检测 → 标点 → 对齐 → `FaceSwapDFM` → 调整 → 合成 → `StreamOutput`，CPU 模式处理完整 4 秒目标视频并保存连续的 100 张 1920×1080 JPEG。100 帧均可解码且哈希不同，Worker 与 FFmpeg 均正常退出；早期 9099 次及中途 37000 次检查点的默认对齐报告分别见 `qa-20261002/official-backend-host-runs/20261002T145535Z-60856/report.json`、`20261002T150022Z-58636/report.json`。这些早期运行证明后台链路技术可运行，没有证明实时帧率；阶段模型的匹配对齐报告见下文。

中途 DFM 的默认 DeepFaceLive 对齐曾产生严重的黑块和面部错位。排查发现当前训练模型是 **full_face (`f`)**，真人 DFL aligned 原图是 `whole_face`，训练读取时会转为 full_face；DeepFaceLive 默认 `FaceAligner` 覆盖值 2.2、内部垂直偏移 -0.08 则给 DFM 输入了显著不同的裁切。对同一原视频第 49 帧，覆盖值 **1.6**、DeepFaceLive 面板 Y 偏移 **0.08**（内部有效偏移 0）与训练裁切的归一化像素 MSE 为 **0.00183**，默认设置约 **0.05707**。在完全相同的 128×128 输入上，PyTorch 检查点与导出的 ONNX DFM 最大绝对误差仅 **1.52e-6**，排除了本次导出权重/张量路径损坏。诊断原图、推理图和报告见 `qa-20261002/dflive-alignment-diagnostic/`、`dfm-discrepancy-37000/`；先前把默认对齐下的黑块主要归因于模型输出的中途报告，已由 `midpoint-37000-alignment-correction.json` 明确更正。

按覆盖值 1.6、面板 Y 偏移 0.08 重跑迭代 37000 的官方后台链，**100/100 帧**连续、可解码、哈希各异，8 个 Worker 均退出；报告见 `qa-20261002/official-backend-host-runs/20261002T151558Z-55392/report.json`。相同第 49、90 帧的大块黑区和严重错位消失，面部完整性明显改善。同参数的 9099 次早期 DFM 也完成 100 帧，报告见 `20261002T152747Z-64656/report.json`，可用于公平比较训练前后。37000 次的 100 帧已编码为带原音轨的 `qa-20261002/midpoint-37000-matched-deepfacelive.mp4`：1920×1080、4 秒、100 帧，FFmpeg 全片解码无错误。该参数是针对本次素材与 `full_face` 模型的实测起点，其他模型应以其训练裁切作对照调节，不能照搬为通用默认值。阶段 99214 模型已用相同参数重新完成双帧、后台和 Qt GUI 测试；视觉评价见下文。

原片约 286 秒，其中约 219 个整秒画面是同一女性，本次训练目标集却只取了 4 秒片段中的 62 张脸。另在忽略目录 `qa-20261002/enhanced-dst-v2/` **独立准备**了 195 张从原片跨时段选取的候选帧，并用仓库原生提取器在 CPU 上生成 195 张 `whole_face`、256×256、68 点元数据的人脸；195 张均通过 DFL 元数据与 ME 实际读取/增强路径，掩码、像素均有限。按同一 atlas，姿态占用格从本次目标集的 8/117 增至 14/117，清晰度达到 0.24 阈值的样本从 2/62 增至 77/195。初筛还保留少量稀缺侧脸和仰头姿态，但该粗姿态估算不等同于 DFL atlas。v2 指纹 `531e8b9b6d17d20f8e7516948ea4270762e0b6c8332ea863879f61e5cdb28a71` 与本次目标集不同，**未参与本次 2.68517 小时训练**，以保持固定样本和历史可比；证据为 `enhanced-dst-v2/manifest.json`、`validation-summary.json`、`atlas.json` 和 `me-reader-validation.json`。如要改善当前可见的合成边界，应在独立项目中用 v2 目标集与新的固定评测 manifest 验证，不能把两阶段指标混为同一数据集。

官方 DeepFaceLive 源码提交已另存到忽略目录 `qa-20261002/official-deepfacelive-source/`，以固定验收依赖。对其原版 Qt GUI，在隔离的 Python 3.10、上游指定的 PyQt6/Qt 6.5.1 与 CPU ONNX Runtime 1.15.1 下，`QSliderCSWNumber` 和 `QXFixedLayeredImages` 向仅接受整数的 Qt API 传入浮点数，导致回调异常及原生进程终止；记录在 `qa-20261002/official-qt-app-runs/20261002T154246Z-71516/unhandled.txt`。因此**原版 GUI 在该环境未通过**，这不是 DFM 加载失败。仅在忽略的 QA 脚本 `official-deepfacelive-qt-app-smoke.py` 进程内对这两个控件值取整、不改上游源码后，真实 `DeepFaceLiveApp`、Qt 窗口/预览、8 个处理 Worker 和视频文件源启动。阶段 **99214 DFM** 在相同兼容修正下完成 **100 张不同且可解码的 1080p 合成帧**；Qt 事件循环及所有 Worker 正常退出，DFM 哈希未变、回调异常为 0。阶段报告见 `qa-20261002/stage-20261002T164657Z-75e865de/official-deepfacelive-qt-gui-100frames.json`，截图见同目录 `official-qt-app-runs/20261002T164815Z-65328/gui-offscreen.png`。该次成功仅证明**带两处运行时兼容修正**的 CPU 图形应用链路；当时尚未验证相机或 GPU 实时帧率。隔离运行环境的包版本保存在 `qa-20261002/official-qt-python310-freeze.txt`。若临时虚拟环境消失，可重建 Python 3.10 环境并用 `DFL_QT_PYTHON` 指向其 `python.exe`。

阶段验收独立入口 `qa-20261002/stage-acceptance.py` 在复制检查点前核对用户停止请求、Web 终态、close ACK、`finished` 心跳，以及 Trainer/sidecar/监控 PID 均已退出；它不发送训练控制，也不改在线模型。可复现命令为 `& '.venv/Scripts/python.exe' 'workspaces/real-interview-me-acceptance/qa-20261002/stage-acceptance.py'`。已完成的一次运行目录为 `qa-20261002/stage-20261002T164657Z-75e865de/`，总结果见其中 `status.json`：`stage-technical-passed-supervision-unconfirmed`、`technicalPassed=true`、`durationThresholdPassed=false`、`longRunEvidenceQualified=false`、`supervisionUnconfirmed=true`、`overallAccepted=false`。审计仍以 **12 小时**为阈值，六项检查（检查点、连续历史、固定样本、预测、DFM、视频）均为 `passed`，但有效时长 **2.68517 小时**，故审计命令按设计返回 1；详见 `audit.json`。原 `qa-20261002/final-acceptance.py` 的 12 小时守卫没有放宽，本次未运行，不能将阶段结果写成长训完成。

阶段模型导出 DFM 约 **329.7 MB**；官方 DeepFaceLive 组件在相邻两帧成功，8 个后台 Worker 和带兼容修正的 Qt GUI 各保存 **100/100** 张不同、可解码的 1080p 帧，相关 Worker 均退出。DFL 离线合成也生成完整 100 帧，`result.mp4`（4 秒、1920×1080、H.264/AAC）与 `result-mask.mp4` 均通过全片解码。阶段目录中的 `official-deepfacelive-components.json`、`official-deepfacelive-backend-100frames.json`、`official-deepfacelive-qt-gui-100frames.json`、`result.mp4` 和 `result-mask.mp4` 是对应证据。另将官方后台的 100 帧编码为带原音轨的 `deepfacelive-result.mp4`，经 FFprobe 确认 100 帧、4 秒、1920×1080、H.264/AAC，FFmpeg 全片解码无错误；它是方便人工观看的补充产物，不替代脚本的逐帧哈希核验。

2026-10-03 又对同一冻结 DFM 运行官方 DeepFaceLive **8 个后台 Worker 的 GPU 多进程链**：检测、标点和 DFM 用 CUDA，合成可选 NVIDIA OpenCL，输入仍为原 4 秒、100 帧的真人目标视频。QA 启动脚本只在自身及子进程加入已有 CUDA DLL 路径；模型和训练数据均未改动。最初 DFM 加载超时的实际原因是 QA 审计强制禁止所有 CPU 节点，而该 DFM 的动态尺寸计算由 ONNX Runtime 分配给 CPU。逐节点 profile 确认这些节点仅为 `Gather`、`Mul`、`Unsqueeze`、`Div`、`Concat`，输入输出全是至多 6 个元素的 `int64` 形状数据；QA 现在仅在显式 `--allow-cpu-support-nodes` 时允许这类有界辅助节点，其他 CPU 节点仍使验收失败。仅用 CUDA 且 CPU 合成的冷启动实时运行保存 **43/100** 帧；CUDA 加 OpenCL 合成的冷启动运行保存 **25/100** 帧，均因丢帧失败，报告分别见 `qa-20261002/official-gpu-backend-runs/20261003T145133Z-66484/report.json`、`20261003T145809Z-29220/report.json`。

随后在**同一批 Worker 内先做不计分的非实时热身**，关闭序列保存，处理完源视频后回到第 0 帧并等待各处理队列排空；热身源日志和切换输出路径时上游重发的第 0 帧单独保存。计分阶段才以实时模式重新播放原 100 帧，逐帧验证源时间戳、连续文件名、图像尺寸和不同哈希，要求输出 ≥24 fps、从 `VideoFilePlayer` 取得帧的时间戳至保存 1080p JPEG 的延迟 P95 ≤160 ms。此延迟包含后台 Worker 链与 JPEG 写入，不包含取得该帧之前的解码时间。三次参数完全相同的热身 OpenCL 运行都保存 **100/100** 张连续且不同的 1080p 帧，每个神经模型在计分阶段各实际调用 100 次；热身阶段采集的 CUDA 节点 profile、计分阶段的 CUDA 优先运行记录、辅助 CPU 节点白名单、模型/视频哈希和 Worker 退出检查均通过。三次输出帧率为 **24.892 / 24.978 / 25.008 fps**；P95 延迟为 **183.783 / 105.950 / 107.658 ms**，因此依次为**未通过 / 通过 / 通过**。报告见 `qa-20261002/official-gpu-backend-runs/20261003T150706Z-38164/report.json`、`20261003T150852Z-45256/report.json`、`20261003T151001Z-64348/report.json`。这证明带热身的官方后台链在本机两次达到本次实时门槛，也保留一次延迟超限的波动；这些后台运行本身**不证明冷启动实时达标、相机输入或未经修改的官方 Qt GUI 的 GPU 运行**，更不改变 12 小时与人工画质验收结论。

之后用同一冻结的 **99214 DFM** 和原 4 秒目标视频，运行了官方 `DeepFaceLiveApp` 的**离屏 Qt 窗口与 GPU 多进程链**。忽略目录 `qa-20261002/official-deepfacelive-gpu-qt-app-smoke.py` 仅在 QA 进程内对上述两个 Qt 整数参数作兼容修正，并使用 Python 3.12、PyQt6/Qt 6.6.1、ONNX Runtime GPU 1.23.2；官方源码保持在提交 `fc7b787`。同一组 Worker 先做不计分的非实时 100 帧热身，再以实时视频文件源计分：连续 **100/100** 张不同、可解码的 1080p JPEG，源帧序号与输出逐项匹配；检测、FaceMesh 和 DFM 在计分阶段各记录 100 次以 CUDA 为首选提供程序的调用，有界 CPU 形状辅助节点单列放行，合成选择 NVIDIA OpenCL。CUDA 内核轨迹只抽样了热身起始的 8 次调用，计分调用未逐次采集内核轨迹；OpenCL 也只验证了设备选择，未单独采集逐帧内核轨迹。窗口/预览已构建并截图、未观察到 Qt 未处理回调异常，全部已知 Worker PID 在结束时退出，DFM 与视频 SHA-256 均未变化。计分输出 **24.963 fps**，源帧时间戳至 JPEG 落盘延迟 P95 **106.665 ms**，达到与后台相同的 24 fps / 160 ms 门槛；GPU 显存峰值比空载基线高约 **3109 MiB**。机器报告、截图和启动日志分别在 `qa-20261002/official-gpu-qt-app-runs/20261003T165849Z-55364/report.json`、同目录 `gui-offscreen.png` 与上层 `launch-20261004T005847.log`。此结论仅覆盖**带两处进程内兼容修正、离屏、热身后、视频文件输入**的 GUI；原版未经修正的 GUI、摄像头和冷启动实时表现仍未通过验证，也不能据此宣布视觉质量或 12 小时长训合格。

复现使用忽略目录中的独立 `qa-20261002/gpu-gui-py312-venv`（Python 3.12.14、`onnxruntime-gpu==1.23.2`、`opencv-python==4.12.0.88`、`numpy==2.2.6`），以及项目 `.venv/Lib/site-packages/torch/lib` 中现有的 CUDA/cuDNN/NVRTC DLL。在 `E:\DFL-PT-WEBUI` 仓库根目录运行以下 PowerShell 命令；它会新建独立报告目录，不覆盖阶段模型或上述报告：

```powershell
& 'workspaces/real-interview-me-acceptance/qa-20261002/gpu-gui-py312-venv/Scripts/python.exe' `
  'workspaces/real-interview-me-acceptance/qa-20261002/official-deepfacelive-gpu-backend-smoke.py' `
  --model 'workspaces/real-interview-me-acceptance/qa-20261002/stage-20261002T164657Z-75e865de/model.dfm' `
  --stage-run 'workspaces/real-interview-me-acceptance/qa-20261002/stage-20261002T164657Z-75e865de' `
  --torch-dll-dir '.venv/Lib/site-packages/torch/lib' --ffmpeg-dir '_internal/ffmpeg' `
  --target-frames 100 --max-seconds 240 --min-fps 24 --max-p95-latency-ms 160 `
  --allow-cpu-support-nodes --merger-opencl-index 0 --warmup-pass
```

带 Qt 窗口的 GPU 对照可从同一仓库根目录复现；脚本固定校验官方源码、DFM 与视频 SHA-256，在独立目录热身和计分，不触碰在线训练项目：

```powershell
& 'workspaces/real-interview-me-acceptance/qa-20261002/gpu-gui-py312-venv/Scripts/python.exe' `
  'workspaces/real-interview-me-acceptance/qa-20261002/official-deepfacelive-gpu-qt-app-smoke.py' `
  --target-frames 100 --max-seconds 240 --cuda-index 0 --merger-opencl-index 0 `
  --allow-cpu-support-nodes
```

人工抽查阶段的第 50、90 帧：官方 DeepFaceLive 输出不再出现默认错位造成的黑块，但脸部仍偏软、身份和时间连续性需要用户观看短片判断；**DFL 离线 `merged/00050.png` 与 `00090.png` 在额头、脸颊可见明显矩形贴片和肤色边界，视觉质量不通过**。这与技术审计通过并不矛盾：审计验证模型与媒体可用、数值有限、格式完整，不为身份自然度背书。后续如追求成片，应先调合成遮罩、颜色和过渡，再考虑在独立项目用扩展 DST 数据进行对照续训；发布生成片段前还需复核素材来源页标记的许可与署名要求。未经修改的上游 DeepFaceLive GUI 与相机输入仍未验证；GPU 后台与带兼容修正的 GUI 实时结果如上单列。

阶段验收后，在冻结的同一检查点上另做了**不重训、不覆盖原结果**的合成参数对照。项目专用候选 `mode=overlay, maskMode=4, erodeMask=20, blurMask=80, colorTransfer=rct` 对完整 100 帧重新合成；62 张有人脸的帧在额头边界跳变指标上全部下降，均值 **0.2303→0.0289**。候选 100 帧与遮罩均可解码，带原音轨的 `candidate-result.mp4` 为 4 秒、100 帧 H.264/AAC，FFmpeg 全片解码通过；模型与原输入哈希不变。抽查帧的硬边明显减轻，但脸仍偏软、身份感不足，**视觉验收结论仍不变**。参数含义、逐帧指标、复现命令和候选成片见忽略目录 `qa-20261002/merge-visual-tuning-20261003/REPORT.md`；这套设置只适用于本段素材的对照，不修改全局合成默认值。

为判断剩余问题是否来自贴合参数，同一冻结检查点又通过 CPU `raw-predict` 模式隔离输出第 50、90 帧未经遮罩和调色的 128×128 预测脸。第 50 帧原片睁眼、张嘴，预测脸已出现半垂眼和不一致的嘴形；第 90 帧也有表情偏差。官方 DeepFaceLive Qt 输出在相近序列帧可见类似眼口问题。由此只能确认**这两帧的表情偏差至少部分存在于合成前**：继续调遮罩、羽化和颜色可改善接缝，无法修复模型预测的眼口几何。双帧原图、哈希和 CPU 命令见上述忽略报告；尚无完整逐帧主观画质验收，视觉不通过的结论不变。

又从已验证的 195 张跨时段目标候选脸中，按原视频秒数划出 **138 张拟议续训图、10 张人工挑选的表情/姿态挑战帧和 47 张时间邻近隔离图**。留出帧与拟议续训图至少相隔 4 秒，其文件哈希也与原 62 张目标训练图不同。冻结的第 99214 轮模型未续训；用独立评测目录对这 10 张未见目标帧做 CPU 推理后，目标侧重建遮罩 MSE 均值为 **0.011518**，而原有 22 张训练内固定评测图为 **0.001204**。对照图中，睁大眼会被预测成半垂眼，极端仰头帧有明显伪影；这给出了合成前表情问题的更多直接证据。挑战帧是有意挑选的难例，两个均值不能视为随机总体估计，也不能单独评价身份自然度。拆分规则、输入哈希、独立 manifest/快照、逐帧图和可复现脚本见忽略目录 `qa-20261002/expression-coverage-20261003/REPORT.md`；通用只读复评命令见[离线 ME 评测](ME_OFFLINE_EVALUATION.md)。后续若获准在**新项目**续训，只能将核验过的 138 张拟议训练图放入训练目录，保留 10 张挑战帧和 47 张隔离图不参与训练；这项续训尚未启动。

2026-10-04 用该离线 CLI 对同一冻结检查点和显式挑战 manifest 再做一次独立 CPU 只读运行，退出码为 0。忽略目录 `qa-20261002/expression-coverage-20261003/offline-cli-20261003/` 保存了 manifest `7e8a9bd0e3d8ebe796c003b7` 和快照 `iter-00099214-e2bcb865`：1 张 SRC、10 张 DST，数据指纹与 manifest 一致；`summary.json` 的生成器权重 SHA-256 为 `8c6ca677fd4f0bf5bbc57f3a10b56356e9a34e5957178e7e23b5174855427836`，`attestation.json` 的完整检查点 SHA-256 为 `d61c30628ced73ad2698b2d5f90c589efc6652f99035f11d1b68d6faf13ae43d`。这是同一失败画质样本的工具链复验，不改变上述视觉结论。

另在隔离的 `web-holdout-bridge-20261003/` 中用 Web 新增的成对独立评测路径配置做只读复验，快照 `iter-00099214-055fdda1` 同样包含 1 张 SRC 和 10 张 DST，目标侧遮罩 MSE **0.011518**、眼口 MSE **0.024771**，与原挑战集快照逐项一致；生成器权重摘要及完整检查点哈希也与上述离线运行一致。该验证没有启动训练任务或改写当前项目的 `.webui` 评测索引。

快照内容校验和 CLI 原子发布上线后，又对原显式 manifest 与同一冻结检查点做了一次 CPU 只读复评，生成 `iter-00099214-2aa809a3f9924dab566688fd`。Node 读取器将新快照识别为 `summary-verified`，54 个 WebP 变体全部通过哈希核对；与上述旧快照的 11 个样本相比，64 个数值指标逐项相同（最大绝对差 0）。模型完整文件 SHA-256 前后仍为 `d61c30628ced73ad2698b2d5f90c589efc6652f99035f11d1b68d6faf13ae43d`，显式 manifest 字节未改动，发布后无残留暂存目录。复现命令、文件哈希及逐项验证见忽略目录 `qa-20261002/expression-coverage-20261003/offline-cli-20261003/REPLAY_20261004.md`；`summary-verified` 仅表示摘要及提供的图片相互一致，不代表对检查点来源作了签名认证，也不改变视觉质量未通过的结论。

同一留出集又从原图副本制成 PAK 和 ZIP，在各自独立评测目录运行冻结模型。两种打包格式都保留 11 个样本 ID，与目录快照的 64 个数值指标和 54 张输出图哈希逐项完全一致；Node 读取器逐张验证通过，检查点及原 manifest 哈希不变。打包准备、两次 CLI 命令和机器报告见忽略目录 `qa-20261002/packed-holdout-20261004/REPORT.md`。这补足真实素材的载体等价性证据，不改变长训与视觉验收未通过的结论。
