# ME 短程开发验收

当前开发验证使用独立的小参数模型，在 **1 小时以内完成训练和安全停止**，检查保存、恢复、有限数值与初步收敛趋势。当前固定预设为 LIAE-UD、64px、ae/e/d/mask 32/16/16/16、batch 2；正式案例和最终画质验收留到后续阶段。不要为通过此检查启动长训。

`tools/me-short-run.py` 启动一个全新的隔离模型，默认总时限 900 秒（15 分钟），包含输出准备、模型加载、训练、保存和退出。`--max-seconds` 只能取 60–3600；到时限前 30 秒发送关闭请求，独立 watchdog 在时限前终止未退出的自有进程树。强制终止、关闭 ACK 不匹配或未确认 `finished` 心跳都不会标为安全完成。GPU 路径只接受 `cuda:0`，使用 FP16；CPU 路径关闭 FP16，不执行双卡。已有非空输出目录会被拒绝。

从仓库根目录运行，SRC/DST 可以是原有数据路径，输出使用新目录：

```powershell
$runRoot = 'workspaces/short-development/run-001'
.\.venv\Scripts\python.exe tools\me-short-run.py `
  --src 'workspace/data_src/aligned' --dst 'workspace/data_dst/aligned' `
  --output $runRoot --device cuda:0 --max-seconds 900
if ($LASTEXITCODE -ne 0) { throw '短时训练未确认安全完成，请检查运行报告' }
$run = Get-Content (Join-Path $runRoot 'short-run-result.json') -Raw | ConvertFrom-Json
$runStart = ([DateTimeOffset]$run.startedAt).ToUniversalTime().ToString('o')
$runStop = ([DateTimeOffset]$run.stoppedAt).ToUniversalTime().ToString('o')
.\.venv\Scripts\python.exe tools\me-short-run-audit.py `
  --model $run.model --expected-model-name short-run `
  --src $run.src --dst $run.dst --material real `
  --run-started-at $runStart --run-stopped-at $runStop `
  --json-out (Join-Path $runRoot 'short-audit.json')
```

`short-run-result.json` 保留运行参数、外层时限、启动器和实际 Trainer 身份、关闭 ACK 与退出证据。Windows 使用隐藏窗口和 kill-on-close Job Object 管理本次创建的进程树。查看趋势后结束本次验证；证据不足时保留 `inconclusive`，不要延长训练来满足旧的时长或画质要求。

`tools/me-short-run-audit.py` 在安全停止后只读检查 `me.pt`、`metadata.json`、完整 `loss-history.jsonl` 及指定的 SRC/DST aligned。工具不启动训练、不导出 DFM、不生成样本清单，只在明确指定 `--json-out` 时写报告。示例：

```powershell
.\.venv\Scripts\python.exe tools\me-short-run-audit.py `
  --model "workspace\model\short-development" `
  --src "workspace\data_src\aligned" `
  --dst "workspace\data_dst\aligned" `
  --material real `
  --run-started-at "2026-10-04T01:00:00+08:00" `
  --run-stopped-at "2026-10-04T01:10:00+08:00" `
  --json-out "workspace\model\short-development\short-audit.json"
```

请记录启动训练进程前的 `--run-started-at` 和安全停止完成后的 `--run-stopped-at`，两者包含加载与保存时间。时间戳是操作者提供的证据，工具不能独立证明进程运行范围；它会验证时间顺序及对 history 和最终 `savedAt` 的包含关系。缺少外层运行时间时，仍可获得技术及趋势结果，但时限验收为 `inconclusive`。墙上跨度包含所有暂停和离线间隔，不能通过忽略长间隔满足上限。`--max-runtime-seconds` 可降低上限，不能高于 3600；`--runtime-allowance-seconds` 是上限以内预留的余量，例如 60 秒余量意味着实测时长必须不超过 3540 秒，没有额外宽限时间。

默认首尾窗口各 20 条，至少需要 40 条连续记录且窗口不重叠。报告列出实际样本数、SRC/DST 的首尾窗口中位数和合并相对降幅。只有两个中位数都不增加、合并中位损失至少下降 1%，才通过初步趋势检查。小幅波动、下降不足或样本不足会得到 `inconclusive`；不会被当作收敛已经证实。可通过 `--window-size` 调整窗口（5–1000 条），通过 `--minimum-relative-drop` 调整正数阈值，需在运行前确定标准。训练随机增强影响损失，因此这个比较只是开发趋势证据，不是统一验证集评估。

工具还严格加载检查点、核对配置/迭代和元数据，按训练桥规则验证模型名称；名称可与保存目录名不同，可用 `--expected-model-name short-run` 明确核对指定模型身份。要求隔离模型的 history 从迭代 1 连续到已保存迭代，拒绝非有限或负数损失/耗时和倒退时间戳。默认每侧按稳定成员排序选择 1 张固定样本（可选 1–3 张），在报告中记录数据集指纹、完整样本哈希和预测形状/范围；可读取松散 JPG、PAK 或 ZIP。检查过程中输入身份变化会失败，报告输出也不能覆盖模型输入或写入 aligned 数据集。

退出码：`0` 表示短程开发检查通过，`1` 表示技术/时限检查失败，`2` 表示证据尚不足。`longRunEvidenceQualified`、`visualQualityAccepted`、`deepFaceLiveAccepted` 始终为 `false`，报告不要求长训、百帧速度或最终视觉效果。

聚焦单元验证不训练模型：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tools\tests\test_me_short_run_audit.py
```

## 2026-10-04 实测结果

本轮使用隔离工作区 `workspaces/me-short-development-20261003T172558Z`，复用真实访谈材料的只读副本（67 张 SRC、62 张 DST），按上述 64px 小参数预设执行一次 900 秒上限训练。没有重新启动原有 99214 轮模型。

- 首轮从北京时间 01:35:47.112326 至 01:50:17.743604，实际 870.63 秒，保存到 **15812** 轮。关闭 ACK 匹配，实际 Trainer PID 15172 的身份和 `finished` 心跳匹配，退出码 0，无强制终止。
- 再次加载该检查点，以目标迭代 `15812 + 2` 续训，3.94 秒内保存到 **15814** 轮并正常退出。首次启动至续训最终退出总跨度 **881.040603 秒（14 分 41 秒）**，包含加载、保存、两次运行之间的间隔和退出。
- 最终只读审计 `shortDevelopmentPassed=true`：检查点、元数据、1–15814 连续历史和每侧 1 张固定样本预测均通过，数值有限。

| 首尾各 20 轮中位损失 | SRC | DST |
|---|---:|---:|
| 首窗口 | 4.506994 | 4.898924 |
| 尾窗口 | 0.596752 | 0.549291 |

两侧均下降，合并相对降幅 **87.82%**。曲线使用不重叠的 200 轮中位数，最后一组 14 轮；它展示训练趋势，不代表正式案例或最终画质合格。

证据位于该工作区的 `run-001/short-run-result.json`、`resume-result.json`、`short-audit.json`、`final-integrity.json` 和 `final-environment.json`；完整命令见工作区 `REPORT.md`，曲线为 `loss-trend.png`。原模型 SHA256 仍为 `d61c30628ced73ad2698b2d5f90c589efc6652f99035f11d1b68d6faf13ae43d`，129 张原始样本及其副本哈希均未变化，审计前后的三个模型输入文件哈希一致，本轮训练进程按 PID 与创建时间核对已退出。

首个只读审计因 PowerShell 自动转换 JSON 时间戳后以本地格式传给 CLI，时限项报 `invalid timestamp`；原失败报告保留为 `short-audit-initial-cli-timestamp-error.json`。按上方示例显式转换 ISO 时间戳后，只读复验通过，没有追加训练。新入口与审计的聚焦验证共 60 项通过。正式画质和真实双卡验证按当前范围延期，历史报告保持原有结论。
