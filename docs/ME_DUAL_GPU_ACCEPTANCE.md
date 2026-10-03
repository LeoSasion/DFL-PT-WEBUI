# ME 双卡验收（当前暂缓）

## 当前范围（2026-10-04）

用户已明确**现阶段无需双 GPU 测试**。保留当前初步实现、已有门槛和历史记录，等待开源用户反馈后再决定是否开展后续验证；本轮不运行双卡 CLI、真实双卡测试或为验收寻找另一台主机。真实双卡训练、梯度同步、稳定性、显存与吞吐仍为未验证，不能宣称已通过。

双卡结果不是当前开发或发布前提。开发训练使用单卡、小参数、小模型，在包含保存、停止和续训的一小时总墙钟预算内观察收敛趋势，见 [ME 短时开发验收](ME_SHORT_RUN_ACCEPTANCE.md)。旧 `partial/skipped` 报告继续保持原义，既不作为失败的发布门槛，也不改成真实双卡通过。

## 历史工具与可选复现

以下说明、命令和双卡签收条件保留作历史参考，仅在以后明确恢复双卡验证范围时使用；当前不执行。

`me_backend.multi_gpu_acceptance` 使用固定的已解码 SRC/DST batch，在主卡上先跑单卡基线，再用 `torch.nn.DataParallel` 在指定的两张卡上训练。双卡预热步的前向 hook 记录两卡实际拿到的样本数，输出张量的反向 hook 记录两卡收到的非零梯度，并检查主卡参数梯度及副卡 CUDA 分配。随后分别测单卡、双卡每步中位时间、图片/秒、各卡 PyTorch CUDA allocated/reserved 峰值，保存检查点、严格重载权重与优化器状态，并再继续一步。验收会检查每步计时为有限正数、吞吐与计时一致，以及主卡和副卡 allocated/reserved 峰值有效。报告记录输入指纹和 batch SHA-256。

从 `_internal/DeepFaceLab` 运行以下 PowerShell 命令。在有两张实际 CUDA 卡的机器上加 `--require-dual`，让缺卡或任何验收失败使命令非零退出：

```powershell
E:/DFL-PT-WEBUI/.venv/Scripts/python.exe -m me_backend.multi_gpu_acceptance `
  --src "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/SRC 人工 aligned" `
  --dst "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/DST 人工 aligned" `
  --devices cuda:0,1 --warmup 2 --steps 5 --require-dual `
  --output "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/profiling/dual-gpu-parity.json"
```

这条默认配置为 `liae-ud`、128px、batch 4、FP32、无随机优化器 dropout/对抗项，用于数值对照。`parity` 模式的门槛为：两卡都参与前向与反向，副卡有 CUDA 分配，吞吐有限且大于零，保存与续训通过；单卡与双卡末步 SRC/DST loss 的绝对差各不超过 `5e-3`，网络权重相对 L2 不超过 `1e-3`、最大绝对差不超过 `1e-3`，预测最大绝对差不超过 `5e-3`。多卡归约顺序与单卡不同，所以要求合理数值接近，不要求逐位相等。吞吐只测量并报告，不强制双卡比单卡快；小 batch 的 DataParallel 开销可能超过并行收益。

两种模式均要求每个实测步骤的 loss 齐全且有限、权重和预测有限，且两卡前向各得到非空样本；单卡 `partial` 基线也检查这些输出。`smoke` 只免除 FP32 数值接近阈值，不免除有限性、真实双卡参与、优化器和 FP16 GradScaler 状态精确重载或续训门槛。无需第二张卡即可运行的判定测试会分别注入空副卡 batch、缺少副卡反向、显存为零、优化器/GradScaler 恢复不一致、loss/权重/预测偏离、早期 loss 为 NaN、缺失实测 loss 以及零/非有限吞吐，确认这些情况不会误报通过。失败报告会以 JSON `null` 记录非有限测量值。这只验证门槛逻辑，不能代替真实双卡训练。

进一步对 RG、FP16、CPU 优化器 offload、GAN、TrueFace 的联合路径做双卡烟测：

```powershell
E:/DFL-PT-WEBUI/.venv/Scripts/python.exe -m me_backend.multi_gpu_acceptance `
  --src "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/SRC 人工 aligned" `
  --dst "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/DST 人工 aligned" `
  --config E:/DFL-PT-WEBUI/docs/examples/me-dual-gpu-stress.json `
  --mode smoke --devices cuda:0,1 --warmup 2 --steps 5 --require-dual `
  --output "E:/DFL-PT-WEBUI/workspaces/me-verification/qa-07a49b681798/profiling/dual-gpu-stress.json"
```

`smoke` 仍记录单卡/双卡损失和权重差异，但由于 AMP 与对抗项有额外数值、随机性差异，不用 FP32 门槛阻止结果；实际两卡梯度、内存、吞吐、检查点和续训仍是硬门槛。可用 `--config` 换成目标训练配置，需确保 batch 至少为 2；数据路径可指向目录、PAK 或 ZIP。

本机 `torch.cuda.device_count()` 为 **1**。已在 GPU 0 运行默认配置的单卡基线（预热 1、实测 2）：单步中位 63.16 ms，约 63.3 图片/秒，GPU 0 allocated 峰值 1,823,494,144 字节；988,578,731 字节检查点重载后权重逐值一致、续训到 iteration 4。联合功能烟测同样完成单卡保存/续训，报告的已执行检查均通过。原始记录分别为忽略目录 `workspaces/me-verification/qa-07a49b681798/profiling/multi-gpu-partial-20261002.json` 与 `multi-gpu-stress-partial-20261002.json`。双卡项明确标记 `partial` / `skipped`，没有宣称已现场通过。

自动测试 `tests/test_me_multi_gpu_acceptance.py` 在单卡机器检查基线和缺卡状态；两项真实双卡测试使用 `@pytest.mark.skipif`，只有 `torch.cuda.device_count() >= 2` 才运行实际 scatter/reduce、数值、显存、联合功能与续训门槛。当双卡硬件可用时，在 `_internal/DeepFaceLab` 执行：

```powershell
E:/DFL-PT-WEBUI/.venv/Scripts/python.exe -m pytest tests/test_me_multi_gpu_acceptance.py -q -rs
```

测试会自行创建人工 aligned 素材，因此复制到另一台双卡机器时可直接运行；上面的 CLI 示例使用本机忽略目录中的 QA 素材，在其他机器上应改为该机器的 SRC/DST 路径。素材仅用于训练链路验收，不评价真人换脸质量。

若要在另一台机器上复现 CLI 的人工素材验收，可从 `_internal/DeepFaceLab` 执行以下命令；`$fixtureRoot` 是新建的临时目录，不依赖本机忽略的 `workspaces/me-verification` 内容：

```powershell
$python = (Resolve-Path ../../.venv/Scripts/python.exe).Path
$fixtureRoot = Join-Path $env:TEMP 'me-dual-gpu-acceptance'
$src = Join-Path $fixtureRoot 'src'
$dst = Join-Path $fixtureRoot 'dst'
& $python -c "from tests.me_fixtures import make_aligned; import sys; make_aligned(sys.argv[1], offset=3); make_aligned(sys.argv[2], offset=11)" $src $dst
& $python -m me_backend.multi_gpu_acceptance --src $src --dst $dst `
  --devices cuda:0,1 --warmup 2 --steps 5 --require-dual `
  --output (Join-Path $fixtureRoot 'parity.json')
& $python -m me_backend.multi_gpu_acceptance --src $src --dst $dst `
  --config ../../docs/examples/me-dual-gpu-stress.json --mode smoke `
  --devices cuda:0,1 --warmup 2 --steps 5 --require-dual `
  --output (Join-Path $fixtureRoot 'stress.json')
```

2026-10-04 已在单卡 Windows 主机用上述生成命令创建素材，并以 `--warmup 1 --steps 1 --require-dual` 验证第一条 CLI 的路径与加载：单卡保存、重载及续训通过；返回 `partial` 且退出码为 2，双卡门槛仍待双卡主机执行。

2026-10-02 本机重跑多卡验收、阶段性能、ME 引擎及网络相关测试共 `80 passed, 2 skipped`；跳过的正是两项需要第二张物理 CUDA 卡的测试。

2026-10-03 复核：`nvidia-smi` 与当前 `.venv` 的 `torch.cuda.device_count()` 均确认只有一张 RTX PRO 6000。当前代码的默认配置和联合配置各重新运行一次 `--require-dual`，两份报告分别为忽略目录 `multi-gpu-partial-20261003.json`、`multi-gpu-stress-partial-20261003.json`：单卡模型权重、主优化器和对抗优化器状态重载及续训均为真；报告仍是 `partial`，命令非零退出，**双卡梯度同步、数值接近、双卡显存和吞吐仍未验证**。共享 GPU 上本次单卡计时明显波动，不应与上次数字直接比较。针对备份、数据格式、性能和多卡的当前回归为 `27 passed, 2 skipped`，跳过的仍是两项真实双卡硬件测试。隔离 Web 全套回归为 `220 passed, 1 skipped`，包括真实的重连、身份与保存 ACK、以及自定义 PAK/ZIP 的评测预检。

在可用双卡主机上，应先确认 PyTorch 报告至少两张物理 CUDA 卡，再在仓库虚拟环境从 `_internal/DeepFaceLab` 执行上述 pytest 命令；必须看到两项原本跳过的测试实际通过，且没有跳过。然后以目标分辨率和 batch 的 aligned SRC/DST 路径分别运行上面的 `parity` 与 `smoke` CLI，保留两份 `status=passed` JSON 和原始吞吐、各卡显存记录。`--require-dual` 会把缺卡或未通过变成非零退出；单卡 `partial` 报告不能作为双卡签收证据。

在另一台 Windows 主机复制仓库后，可从其 `_internal/DeepFaceLab` 目录用相对路径先做设备与自生成素材门槛；这一步无需复制本机忽略的 QA 素材：

```powershell
$python = (Resolve-Path ../../.venv/Scripts/python.exe).Path
& $python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() >= 2"
& $python -m pytest tests/test_me_multi_gpu_acceptance.py -q -rs
```
