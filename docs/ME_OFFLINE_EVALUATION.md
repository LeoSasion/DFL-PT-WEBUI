# 离线只读 ME 评测

`webui/scripts/evaluate-me-checkpoint.mjs` 可以对已有 `me.pt` 做独立推理评测，不启动训练，也不保存检查点。SRC/DST 可是带 DFL 元数据的 aligned 目录、PAK 或 ZIP，包括当前 Web 项目工作区以外的绝对路径。命令沿用 Web 的姿态 probe、manifest 和快照格式。

先在 PowerShell 中设置自己的路径，再从仓库根目录执行：

```powershell
$frozenMePt = (Resolve-Path '<已有模型目录>\me.pt').Path
$srcFaceset = (Resolve-Path '<SRC aligned 目录或 PAK/ZIP 文件>').Path
$dstFaceset = (Resolve-Path '<DST aligned 目录或 PAK/ZIP 文件>').Path
$outputRoot = '<新建的独立评测输出目录>'
& '_internal/node/bin/node.exe' webui/scripts/evaluate-me-checkpoint.mjs --model $frozenMePt --src $srcFaceset --dst $dstFaceset --name '<模型名称>' --evaluation-root $outputRoot --create-manifest --device cpu --timeout-seconds 3600
```

输出位于 `$outputRoot/<model-key>/`，包括 `manifests/<manifest-id>.json`、`snapshots/<snapshot-id>/summary.json`、逐张 WebP 对照图和 `attestation.json`。`summary.json` 的 `modelWeightsDigestVersion` / `modelWeightsSha256` 标识生成器权重，能区分迭代和配置相同但权重不同的模型；`attestation.json` 则绑定完整检查点文件 SHA-256、迭代、SRC/DST 路径与数据集指纹。脚本先在该模型评测目录的 `_pending-cli-*` 中推理，待检查点与数据集复核、attestation 写入后才原子发布整个快照；进程意外终止时可能留下未发布的暂存目录，但 Web 快照列表不会把它当作完成结果。输出目录不得包含检查点或输入数据，也不得位于它们内部。

旧快照 ID 的随机后缀为 8 位十六进制，新快照的内容锚定后缀为 24 位。界面显示的 `summary-verified` 只验证摘要与提供的图片相互一致，不认证 `attestation.json` 或检查点的历史来源；需要证明使用了某个冻结文件时，应把 `--checkpoint-sha256` 与可信的外部哈希记录核对。

固定样本复评时，将 `--create-manifest` 换成 `--manifest '<已有 manifest 的绝对路径>'`。该文件必须位于本次 `$outputRoot/<model-key>/manifests/`；使用手工构建且非模型名称派生的 key 时另加 `--model-key '<manifest 中的 key>'`。可加 `--checkpoint-sha256 '<64 位小写十六进制摘要>'`，要求精确使用预期的冻结检查点。评测缺省超时为 120 秒，`--timeout-seconds` 可设为 1–3600。

显式 manifest 的 `createdAt` 可省略；它不参与 `manifestId` 的内容哈希，Web 评测读取器也接受省略该字段的有效 manifest。

`--create-manifest` 使用与 Web 相同的姿态分箱及确定性抽样，每侧最多 180 张；它不保证包含指定的每一张挑战帧。人工挑选的留出集应使用已经校验样本和内容 ID 的显式 manifest，避免抽样规则改变其成员。不同 manifest 或数据集指纹的指标不能当作同一固定样本上的前后对比。

2026-10-04 对冻结的真人 ME 检查点与同一组 1 张 SRC、10 张 DST 留出脸，分别通过普通目录、PAK、ZIP 做了完整 CPU 推理。三种载体使用各自的数据集指纹与 manifest ID，但样本 ID 相同；PAK、ZIP 各自的 64 个数值指标与目录版逐项相同（最大绝对差 0），54 张 WebP 输出的哈希也逐张一致。Node 读取器对两份打包快照返回 `summary-verified`，模型 SHA-256 前后不变。原始输入未打包或删除；准备脚本、命令和报告见忽略目录 `workspaces/real-interview-me-acceptance/qa-20261002/packed-holdout-20261004/REPORT.md`。这证明该固定样本的格式等价性，不推断其他数据集或视觉质量已通过。

Web 新建 ME 训练任务的参数页还可同时填写“独立评测 SRC 人脸集”和“独立评测 DST 人脸集”。这两项只决定训练过程中的质量快照样本，训练的 SRC/DST 输入仍由原字段指定；都留空时维持原有训练内固定样本行为。Web 路径必须位于当前项目 workspace，可为 aligned 目录、PAK 或 ZIP；单填一侧、使用训练集的相同或嵌套物理路径会在预检中拒绝。复制到另一目录的重复图片仍可能混入，作为真正留出集使用前应自行核对样本哈希和来源时段。Web 不会从已停止的模型自动补跑快照；此时使用上面的只读 CLI。
