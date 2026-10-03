# ME 阶段性能对比

`me_backend.profile` 在同一批已解码的 SRC/DST 样本上，分别以相同 seed 新建模型，对 RG、FP16 和优化器 CPU offload 的全部八种组合运行短基准。每步通过 `StepProfiler` 在张量传输、前向、损失、反向、梯度检查、判别器和优化器阶段边界同步 CUDA，并记录阶段时间、PyTorch 进程 CUDA 峰值 allocated/reserved 及优化器状态所在设备的逻辑字节数。阶段 `median_ms` 只取实际执行该阶段的步骤；`executed_steps` 记录执行次数，`mean_ms_per_step` 将未执行的步骤按零耗时计入，便于解读间歇发生的 replay 阶段。数据加载与增强在计时区间外，只执行一次；普通训练不启用 profiler，也不会增加阶段同步。

下表的显存列是每步峰值的**中位数**，用于观察典型步骤。判断配置能否放入显存时，应查看报告的 `max_cuda_peak_allocated_bytes` 及对应的 `max_cuda_peak_allocated_bytes_change_percent`：间歇执行的 replay 或个别步骤的显存峰值可能高于中位数。这个最大值仍只覆盖本次短测的固定批次和当前 PyTorch 进程，应在目标训练配置上留出余量复测。

从 `_internal/DeepFaceLab` 运行，使用已有的人工对齐素材（不会改动图片或模型）：

```powershell
E:/DFL-PT-WEBUI/.venv/Scripts/python.exe -m me_backend.profile `
  --src "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/SRC 人工 aligned" `
  --dst "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/DST 人工 aligned" `
  --device cuda:0 --warmup 2 --steps 5 `
  --output "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/profiling/default128-rg-fp16-cpuopt-20261002.json"
```

若要复用 64px QA 配置与 SRC/DST 路径，可用 `--fixture-manifest E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/fixtures.json` 代替 `--src`、`--dst`；`--config` 可接受自己的 ME JSON 配置。要测 PAK/ZIP，可将它们直接传入 `--src` / `--dst`。报告含基础配置、实际变体选项、输入指纹与批次 SHA-256，确保每个变体使用同一批样本。选择 `--device cpu` 时只比较 RG；FP16 和 CPU offload 需在 CUDA 上比较。

## 本机实测（2026-10-02）

设备为 NVIDIA RTX PRO 6000 Blackwell，PyTorch 2.9.1+cu128；ME 默认 `liae-ud`、128px、ae/e/d/mask 为 256/64/64/22、batch 4。每组先预热 2 步，再测 5 步，表内为中位数。SRC/DST 是人工对齐素材，固定批次 SHA-256 为 `69fcd94ae82ffb83e4c8c861ed792afbea8f5c53ab1c1fe4a766aa30d21c8ba1`。运行前 `nvidia-smi` 显示 LM Studio 等进程占用约 56.9 GiB，故计时会受共享 GPU 负载影响。原始逐步、逐阶段数据在上述忽略目录的 JSON 中。

| RG | FP16 | CPU 优化器 | 单步中位毫秒 | 峰值 allocated MiB | 相对默认耗时 | 相对默认峰值 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 关 | 关 | 关 | 67.94 | 1739.0 | 基线 | 基线 |
| 关 | 关 | 开 | 433.50 | 1108.1 | +538.0% | -36.3% |
| 关 | 开 | 关 | 58.09 | 1577.4 | -14.5% | -9.3% |
| 关 | 开 | 开 | 416.03 | 924.1 | +512.3% | -46.9% |
| 开 | 关 | 关 | 89.55 | 1491.9 | +31.8% | -14.2% |
| 开 | 关 | 开 | 490.65 | 862.3 | +622.2% | -50.4% |
| 开 | 开 | 关 | 75.11 | 1429.6 | +10.6% | -17.8% |
| 开 | 开 | 开 | 433.43 | 780.9 | +537.9% | -55.1% |

本配置优先试 FP16：这里既减少峰值约 9.3%，又使同步计时缩短约 14.5%；仍须在实际训练中确认 loss 有限并能正常保存/续训。RG 适合显存接近上限时逐步启用：单独省约 14.2%，但这组数据慢约 31.8%；它的重算开销主要反映在生成器反向阶段（16.85 → 36.30 ms）。优化器 CPU offload 将约 659 MB 状态从 CUDA 转到 CPU，单独节省约 36.3% 峰值，但优化器阶段由 9.10 增至 380.29 ms，因此只适合作为显存不足时的后续选项。

这些数值是短时、固定批次、强制同步的诊断结果，不是最终吞吐或画质保证。峰值 allocated 是当前 PyTorch 进程的分配量，不包含其他应用占用或驱动预留。较小的 64px/batch1 QA 模型上 RG 单独反而使峰值上升，说明配置建议必须按目标架构、分辨率、batch 和实际设备重测；旧版 128px RG 独立对比见 [ME 训练验证](ME_TRAINING_VALIDATION.md)。

## 真人访谈数据复测（2026-10-03）

用阶段模型 `interview-me-128` 的元数据配置（LIAE-UD、128px、batch 4）和实际训练所用的 67 张 SRC、62 张 DST aligned 人脸，再跑同一八组合。每组新建同 seed 的模型，预热 2 步、同步计时 5 步；只训练进程内的临时模型，没有恢复或改写已停止的 99214 轮检查点。八组复用同一已解码 batch，SHA-256 为 `b66556cf99c20c41d6379d3b0771746668b628451d875bd39123b60398eafdef`；源和目标指纹分别为 `1bdcce22adda741955afe094d8f947f196d2dfb0873bcb0b764ccb69b7ed5fa7`、`c8f31db606b107457eb24df4a06f4d5f9cf8b754e2bfcf982016f51f3616f38b`。原始阶段样本、loss 和配置保存在忽略目录 `workspaces/real-interview-me-acceptance/qa-20261002/real-data-profile-20261003.json`，八组实测 loss 均有限。

| RG | FP16 | CPU 优化器 | 单步中位毫秒 | 峰值 allocated MiB | 相对 FP32 基线耗时 | 相对 FP32 基线显存 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 关 | 关 | 关 | 50.59 | 1739.0 | 基线 | 基线 |
| 关 | 关 | 开 | 316.01 | 1108.1 | +524.6% | -36.3% |
| 关 | 开 | 关 | 40.41 | 1577.4 | -20.1% | -9.3% |
| 关 | 开 | 开 | 311.58 | 924.1 | +515.8% | -46.9% |
| 开 | 关 | 关 | 61.95 | 1491.9 | +22.5% | -14.2% |
| 开 | 关 | 开 | 346.39 | 862.3 | +584.7% | -50.4% |
| 开 | 开 | 关 | 56.50 | 1429.6 | +11.7% | -17.8% |
| 开 | 开 | 开 | 326.74 | 780.9 | +545.8% | -55.1% |

在这批真人数据和当前显存余量下，保留训练采用的 **FP16、不开 RG、优化器留在 GPU** 是这次短测中最快的组合；相对 FP32 基线节省约 9.3% 进程显存，单步中位耗时缩短 20.1%。若目标设备显存逼近上限，先试 RG：单独降低 14.2% 峰值，但生成器反向阶段中位耗时约 12.14→24.85 ms；CPU 优化器单独降低 36.3% 峰值，优化器阶段却约 6.90→271.03 ms，应作为显存不足时的后续选项。这里的数值只比较固定批次的短步，不代表 12 小时吞吐、画质或共享 GPU 下的稳定性能。

复现时从 `_internal/DeepFaceLab` 运行 `me_backend.profile`，将 `--src`、`--dst` 指向 `workspaces/real-interview-me-acceptance/data_src/aligned` 与 `data_dst/aligned`，`--config` 指向同一忽略目录的 `real-data-profile-config.json`，设置 `--device cuda:0 --seed 7 --warmup 2 --steps 5` 并给出新的 `--output`。配置文件是从阶段模型的 `metadata.json` 中提取的 `config`，报告中的源/目标指纹和 batch 哈希应与上述值一致，才可直接比较数值。
