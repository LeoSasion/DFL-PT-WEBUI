# ME 长训、DFM 与成片验收（历史可选审计）

## 当前范围（2026-10-04）

本页归档旧 12 小时规格及其审计工具，供以后正式案例另行明确启用时参考。**当前禁止长时间训练，开发验收改用小参数、小模型，包含保存、停止、续训在内的总墙钟预算最多一小时，只观察收敛趋势。** 当前入口和结果口径见 [ME 短时开发验收](ME_SHORT_RUN_ACCEPTANCE.md)；最终画质和正式案例以后处理，双 GPU 暂缓测试。

本页的 12 小时时长、人工视觉与 DeepFaceLive 现场要求**均不是当前开发或发布门槛**。历史工具保留原阈值和字段语义；原有 `longRunEvidenceQualified=false`、视觉未通过等报告不会追认为成功，也不通过修改旧阈值把长训审计充当新的短时验收。

## 历史审计工具与复现

`tools/me-longrun-audit.py` 用于训练安全停止后的技术验收。它读取原生 `me.pt`、`metadata.json`、`loss-history.jsonl` 和指定的 SRC/DST aligned 人脸集；除首次建立 `audit-samples.json`、可选 JSON 报告外，不修改模型或媒体。再次运行时，它要求样本清单中的数据集指纹和成员名完全一致。换数据阶段请指定新的 `--sample-manifest` 路径，并保留旧清单与报告。

以下是旧 12 小时规格的历史审计示例，仅在以后另行明确开展正式案例时适用；现阶段不为运行此审计启动或延长训练：

```powershell
.\.venv\Scripts\python.exe tools\me-longrun-audit.py `
  --model "workspace\model\my-me" `
  --src "workspace\data_src\aligned" `
  --dst "workspace\data_dst\aligned" `
  --material real --minimum-hours 12 `
  --dfm "workspace\model\my-me\model.dfm" `
  --video "workspace\result\merged.mp4" `
  --json-out "workspace\model\my-me\longrun-audit.json"
```

`--src`、`--dst` 同样接受显式 `.pak`/`.zip`，也接受包含默认 `faceset.pak`/`faceset.zip` 的目录。没有现成 DFM 时省略 `--dfm`，工具会临时导出并进行 ONNX Runtime CPU 数值对照；这只验证导出能力，不会留下一个可供 DeepFaceLive 加载的文件。没有最终视频时省略 `--video`；提供视频时会用内置 ffprobe 核对视频流，再用 FFmpeg 完整解码，不能只凭容器元数据判定可播放。

报告逐项核对：检查点能被 MEEngine 严格加载、元数据和迭代数匹配（旧 metadata 缺失的新增默认配置项按 MEConfig 补齐，显式冲突仍拒绝）；loss history 从 1 到检查点迭代逐项连续，损失和耗时有限，时间戳不倒退；固定 SRC/DST 样本的内容指纹不变、预测张量形状和数值范围有效；DFM 输入/输出名称符合 DeepFaceLive 契约，动态 batch 的 ONNX 输出与 PyTorch 推理在 `rtol=2e-4, atol=2e-6` 内一致。报告给出首末 100 条平均损失供人工观察，但**不以损失下降自动判定换脸质量**。

有效训练时长由 history 时间戳估计，离线间隔超过默认 300 秒不计入；同时报告纯步骤计算时间、墙上跨度、暂停次数。可以用 `--max-gap-seconds` 调整对慢速加载环境的间隔上限。`--minimum-hours 0 --material synthetic` 仅用于人工合成数据的短程技术自测，报告的 `longRunEvidenceQualified` 仍为 `false`。`--material real` 是操作者声明，工具不识别真人身份或素材授权；只有声明真人、有效时长达到至少 12 小时、全部技术检查通过时，该字段才可能为 `true`。即使如此，`visualQualityAccepted` 和 `deepFaceLiveAccepted` 始终为 `false`，需要现场人工验收。

真人成片的人工验收应保存有代表性的正侧脸、表情、遮挡、光照和连续运动片段，检查身份相似、轮廓接缝、遮罩溢出、色差、闪烁与视频音画同步。DeepFaceLive 验收需在实际目标主机加载最终 DFM，确认实时输入输出、掩码、连续帧稳定性、延迟和显存占用，并记录版本、设备和现场结果。ffprobe、ONNX 数值对照或人工合成数据短训均不能替代这两项视觉验收。

DFM 数值对照只说明 ONNX 与 PyTorch 对**同一已对齐输入**给出相近输出。DeepFaceLive 的 `FaceAligner` 覆盖值和偏移还必须匹配训练时实际送进 ME 的脸部裁切；不匹配会造成黑块、错位和模糊，即使 DFM 数值检查全部通过。对真人样本先保存一张训练输入裁切，再从相同原视频帧经 DeepFaceLive 对齐并比较，调到两者接近后才评估 DFM 画质。本仓库访谈样本的实测参数和证据见 [ME 真人访谈长训现场验收](ME_REAL_MEDIA_ACCEPTANCE.md)。

本机自动自测可运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tools\tests\test_me_longrun_audit.py
```
