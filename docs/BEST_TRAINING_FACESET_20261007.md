# Best Training Faceset 开发验收（2026-10-07）

训练子集采用“先保证够好，再最大化多样性”。`targetCount` 是数量上限。它先保存全量输入清单、原始 SHA 与 sidecar 指纹，每次最多分析 500 张；只有全部批次完成才能进行一次全局选择。不会把各 500 张的独立排序拼接成所谓全局多样性。双图特征计算块最多 250 张。

旧 `quality-coverage` 排序的事务和有限盲评证据仍保留在 [FACE_QUALITY_COVERAGE_UPGRADE_20261007.md](FACE_QUALITY_COVERAGE_UPGRADE_20261007.md)。本功能是新的独立训练子集计划，不覆盖该历史验收。

## 质量、身份与覆盖边界

生产质量分精确复用既有 `face_quality_review.quality_review` 的综合分：前景 Tenengrad 0.55、曝光 0.20、已有源框分辨率 0.15、对齐一致性 0.10；不可用分项从分母中省略。调用使用既有 XSeg mask 的 `bounded_image_metrics`，与 [Efficient-FIQA 对照](EFFICIENT_FIQA_COMPARISON_20261007.md) 的 current baseline 一致。`score = aggregate * 100`，不是准确率、身份概率或百分位。模型标识仍为 `foreground_tenengrad`，provenance 保留各分项、方法、当前输入 SHA 与显示尺度。

`minimumAcceptable=5/100` 是未校准的保守开发策略，不能称为从小样本拟合得到的质量阈值。严重诊断使用现有 aligned 像素上的 YOLO26s-face、关键点形状与画布边界、极端曝光和 raw foreground Tenengrad `<.008`。明显多脸、清晰检测与关键点严重不重合、关键点大比例出框、坏 JPEG/不可读或被安全数据 pickle 读取器拒绝的元数据会进入 `rejected`。弱检测或新旧关键点不够一致进入 `review`，以免把困难侧脸直接认定错误。坏文件原始字节也能作为 opaque 副本保留；发布阶段不反序列化 APP15。

没有原始帧时仍能分析 aligned。YOLO 重检测和必要时新 TUFA68/98 诊断都绑定当前 aligned 的 SHA，不改写图像或标注。这验证的是裁片内证据，不是原始帧检测框标注准确性。有原生98点时先检查已有 affine、画布和点数契约；缺少时用当前 aligned 上的新 TUFA 诊断补充。

姿态来自既有 68 点 PnP，以该几何实现返回的 signed yaw/pitch 角度分桶。眼口覆盖来自 WFLW98 几何纵横比：眼 `<.16`、嘴 `<.10/.25` 形成覆盖桶。这些不是经人工标签验证的闭眼、张嘴或遮挡分类。左右正负方向沿用 PnP 几何约定，不能当作已验证的摄像机方向标注。

身份仅把严格 PyTorch SFace 的共享五点 112 crop 描述子与用户明确确认的 1–32 张同一人物参考比较。没有参考时全部可读合格样本进入 `review`；相似度不足仍是身份待复核，不能直接称为错误身份。不会把最大群当真身份。多姿态参考和相似度阈值 `.50` 也是透明开发策略；相互冲突或不可用的参考阻止自动选择。

唯一 Efficient-FIQA EdgeNeXt-XXS 候选未达到预先锁定的明显收益门槛；因此生产保持上述综合分。`status` 返回它的 `comparison-only/default:false` 状态；评测计划允许分析与全局对照，但 Python `publish`（包含 dry-run）拒绝将它发布为生产训练子集。不会自动加载 FIQA，亦未增加其他 FIQA 模型。

## 全局选择与保留

先去除完全相同 decoded-pixel SHA 的重复；只有原始帧 SHA、提取来源 SHA、整数 PTS 与有理 timeBase 全部绑定时才减少 `.35s` 内同覆盖且高相似的连续副本。比较始终对已保留代表，避免把相似链整段折叠。没有可靠时间清单时不会猜测连续帧。重复项明确标记 `low-value-duplicate`，不是检测或身份错误，并记录所保留代表。

每个合格覆盖桶先保留质量代表，优先补全不同 yaw/pitch 与眼口几何。之后跨全量清单按离已有集合的最小距离增加样本：姿态 `.45`、眼口桶差异 `.30`、SFace 描述子变化 `.15`、亮度 `.10`。这是可解释特征距离，不是训练收益预测。默认 `minimumNewDiversity=.035`，只有新增距离达到该门槛才继续；门槛可在 **0.005–0.30** 范围调整，不能设 0 强行凑上限。已合格但没有足够新增覆盖的剩余图保留在 `review/adequate-reserve`。

理由最多三条，记录质量分与保底、signed yaw 和眼口几何、该桶候选数及稀缺覆盖。没有可靠 PTS 时不会使用“减少连续画面”的理由。低一点质量但已达保底的稀缺大侧脸、眼口几何样本不会因为纯质量 TopN 而被同类高分正脸挤掉。

最终只创建新的 `best-training-<id>/selected`、`review`、`rejected`。所有图片保留 byte SHA、decoded pixel SHA、APP15 SHA、可用 sidecar SHA 与理由。回执包含全量分类，不删除或重命名源件；训练入口只应使用 committed 回执对应的 `selected`。dry-run 不创建输出目录。撤回把三个输出目录移到该批次的 `withdrawn` 中，继续保留所有副本和源件。

发布的中间失败会保留暂存回执；若目录重命名已提交而计划元数据保存失败，重试仅在完整回执、计划 SHA、类别和所有输出 SHA 一致时接纳已有提交，不覆盖文件。撤回若部分重命名失败会回滚，并可重试。修改过、缺失或额外加入的输出文件会阻止撤回；已撤回批次不能再次作为生产发布复用。

## Python CLI 与 JSON 接口

入口为 `webui/python/best_training_faceset.py`，仅最后一行 stdout 是 JSON；进度为 stderr 的 `DFL_PROGRESS {stage,current,total,detail}`。

```text
status
init --input <aligned/PAK/ZIP> --plan-root <root> --target-count N
     [--reference <member>] [--frames <frame-dir>]
     [--identity-threshold .50] [--minimum-diversity .035]
     [--quality-model foreground_tenengrad|efficient-fiqa]
analyze --plan <plan-32hex> --offset 0 --limit 500 --device cpu
inspect --plan <plan-32hex> --offset 0 --limit 120 [--status selected|review|rejected]
confirm --plan <plan-32hex> --reference <member> [--reference <member>]
finalize --plan <plan-32hex>
publish --plan <plan-32hex> --output-root <root> [--dry-run]
recover --receipt <batch/receipt.json> [--dry-run]
```

状态包含 `total/completedCount/globalReady/nextOffset`、可见 `selectedRange`、`items` 与 `selection/publication`。每个 item 的 `position` 始终是原始全量 inventory index；分类分页的 `selectedRange.total` 是该分类总数，顶层 `total` 仍为全量。`confirm` 可复用已完成的特征，并清除旧决策。已发布计划不允许改参考；撤回后的 `inspect.state` 为 `withdrawn`。

## 验收证据与局限

针对本功能的 26 个测试覆盖：500+3 全局累积与早期 finalize 拒绝、无时间近似样本不凑上限、低分稀缺覆盖、未知身份/冲突参考、可靠 PTS 的非传递重复、floor 仅复核、FIQA 对照发布守卫、0–100 aggregate 等价、fresh TUFA98 当前像素绑定、worker finally close、坏 JPEG/无元数据/恶意 pickle opaque 保留、损坏内嵌 XSeg 遮罩不阻塞其他图片、缺失或异常68点、原图及 sidecar 变更拒绝、全量 byte 精确发布/撤回、输出新增文件冲突、原始帧变更以及发布与撤回故障重试。合并现有质量排序与安全 pickle/事务测试为 **64 passed**；只扩大本轮必要回归范围，未运行 ME 训练。

真实小数据验收位于忽略目录 `workspace/.vision-evaluation/best-training-faceset-20261007-v1/report.json`，脚本为 `tools/verify-best-training-faceset.py`。从已有同一个 SRC 访谈的 67 张 aligned 中按位置 `0,6,...,66` 只读抽取 12 张非邻样本及 sidecar 到隔离副本，真实 CPU YOLO26/TUFA68+98/SFace 与综合分执行约 **9.48 秒**。先分析两个 6 张窗口、拒绝半量 finalize、无参考全部 review，再由开发验收指定同侧参考，最后全局选择、dry-run、byte 精确发布与撤回。12 张有 8 个覆盖桶，并有其他特征新增距离，因此有限集合保留全部 12；这不要求实际数据必然减少数量。

原件 SHA 与隔离输入 SHA 均不变，输出与撤回副本的 JPEG、像素、APP15 与可用 sidecar 字节证据保留。没有原始帧时间清单参与本轮，未作连续帧去重声明。另一个受控测试让 500 张不同像素 SHA、无时间、近似正脸覆盖的样本与 3 张质量较低但合格的稀缺覆盖共同参与，cap400 时 **selected4、review499、rejected0**；这是不凑配额的功能证据，不是视觉准确率。

这 12 张来自同一 SRC 访谈，非邻不意味着独立视频样本。身份参考是开发流程指定，不是人工身份真值集；实际用户仍必须本人确认参考。所有覆盖桶和保底阈值均未通过正式标注校准，本轮不能证明普遍质量、身份准确性或最终 ME 训练收益。
