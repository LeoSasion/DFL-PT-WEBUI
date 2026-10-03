# ME PyTorch 后端

本后端已接入 WebUI、ME Merger 和 DFM 导出。以下命令从仓库根 `E:\DFL-PT-WEBUI` 运行，使用根 `.venv`。Web 向导已接入完整的 45 项 ME 配置、预设、严格续训差异、预训练和原生权重初始化；CLI/JSON 保留相同训练入口。

## 新建训练

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\main.py train `
  --training-data-src-dir E:\DFL-PT-WEBUI\workspace\data_src\aligned `
  --training-data-dst-dir E:\DFL-PT-WEBUI\workspace\data_dst\aligned `
  --model-dir E:\DFL-PT-WEBUI\workspace\model --model ME `
  --force-model-name my-me --resolution 128 --batch-size 4 `
  --force-gpu-idxs 0 --steps 1000
```

默认 `liae-ud`，resolution 128，ae/e/d/mask 为 256/64/64/22，batch 4。`--steps` 是本次追加迭代数，`0` 为持续训练；CLI `--target-iterations` 是保存并停止的总迭代目标。Web 向导的目标迭代默认用于进度估算，勾选“达到目标后自动保存并停止”才会执行该总迭代上限。困难样本回放可额外更新优化器，所以 `optimizer_updates` 可以大于 `iteration`。

`me.py train` 和 `me.py web-train` 使用同一训练、元数据和控制桥，直接指定模型实例目录：

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py train `
  --src E:\DFL-PT-WEBUI\workspace\data_src\aligned `
  --dst E:\DFL-PT-WEBUI\workspace\data_dst\aligned `
  --model E:\DFL-PT-WEBUI\workspace\model\advanced-me --name advanced-me `
  --config E:\DFL-PT-WEBUI\advanced-me.json --device cuda:0 --steps 1000
```

JSON 可以是部分配置，其余使用默认值。例如 `advanced-me.json`：

```json
{
  "archi": "df-ud",
  "use_rg": true,
  "use_fp16": true,
  "optimizer_on_cpu": true,
  "gan_power": 0.1,
  "gan_patch_size": 16,
  "gan_dims": 16,
  "true_face_power": 0.01,
  "uniform_yaw": true,
  "data_workers": 2
}
```

独立 CLI 支持 `--device cpu`、`--device cuda`、`--device cuda:0` 和 `--device 'cuda:0,1'`；主 CLI 分别使用 `--cpu-only`、`--force-gpu-idxs 0` 或 `--force-gpu-idxs '0,1'`。PowerShell 的逗号参数需加引号。多卡要求 batch 不小于卡数。多卡调度已实现，尚无真实多卡验收；当前实测为一张 NVIDIA RTX PRO 6000 Blackwell。FP16 训练要求 CUDA，CPU 预测和导出使用 FP32。

模型目录有 `me.pt`、`metadata.json`、`preview.png`、`loss-history.jsonl` 及可选备份和评估。保存、备份、刷新预览、评估和关闭使用 Web 控制 ACK；Ctrl+C 在完整迭代边界保存。Web ME 训练由独立监督进程持有，Web 服务重启后依据持久状态重连日志、控制、预览和任务锁；训练心跳在页面显示正常、检测中或疑似停滞，疑似停滞不会自动终止进程。其他 ConPTY 任务的服务重启语义仍按各任务处理。

自动备份默认每 1000 次迭代建立一个校验过的版本，默认保留最近 3 代；`--backup-every` 和 `--backup-keep` 可调整间隔和代数（保留 2..32 代）。新备份一并保存当前损失历史，恢复旧迭代后再续训不会重复追加历史记录。手动备份不随自动备份轮换；没有历史副本的旧版备份仅在现有历史可连续截取到该迭代时恢复历史。Web“工作区 → 模型恢复点”可选择恢复，必须先停止占用工作区的任务；恢复前会为当前有效版本建立手动回滚备份。独立 CLI 同样提供固定命令：

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py list-backups `
  --model E:\DFL-PT-WEBUI\workspace\model\my-me
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py restore-backup `
  --model E:\DFL-PT-WEBUI\workspace\model\my-me `
  --backup-id "automatic/iter-00001000-<从列表复制的时间戳>"
```

运行恢复命令前先安全停止训练，并将示例中的占位符替换为列表返回的完整 `id`；损坏的主检查点在恢复前会留存原始副本，不把损坏数据标记为有效回滚点。

## 严格恢复与修改配置

续训使用同一模型并加 `--resume`。主 CLI 发现该实例的 `me.pt` 时也会自动续训。JSON 的部分字段与检查点合并，显式 CLI 参数再覆盖对应字段。未知字段、非法值和未授权配置差异直接报错，不会静默忽略。

| 选项 | 含义 |
| --- | --- |
| `--allow-config-change` | 允许兼容训练变更，例如 lr、batch、RG、增强和对抗强度；未指定时任何实际差异均拒绝 |
| `--reset-optimizer` | 清空优化器历史和更新计数；显示迭代及相同形状的网络/判别器权重保留 |
| `--reset-data-state` | 重建采样状态并清空困难样本回放；更换数据集时使用 |
| `--initialize-from` | 新模型仅复制原生检查点网络权重，优化器和迭代从零开始；不能与 resume 同用 |

架构、resolution、ae/e/d/mask 维度和 face_type 是结构字段，不能在同一检查点内更改。更换 AdaBelief/RMSprop 或改变已有判别器形状需 `--reset-optimizer`；形状改变的判别器重新初始化。新增对抗分支初始化新分支，已有兼容分支保留权重和优化器状态。同一 FP16 策略下的配置修改保留 scaler；回放缓存按新容量保留。

例如准备 `fine-tune.json`：

```json
{"use_rg": true, "batch_size": 2, "lr": 0.00001}
```

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py train `
  --src E:\DFL-PT-WEBUI\workspace\data_src\aligned `
  --dst E:\DFL-PT-WEBUI\workspace\data_dst\aligned `
  --model E:\DFL-PT-WEBUI\workspace\model\my-me --resume `
  --config E:\DFL-PT-WEBUI\fine-tune.json --allow-config-change `
  --device cuda:0 --steps 1000
```

只有确实更换数据集时才加 `--reset-data-state`；只改学习率无需重置采样。若 JSON 同时修改 `adabelief`，再加 `--reset-optimizer`。

## 预训练转微调

预训练要求专用 aligned 人脸目录。`pretrain=true` 时 SRC/DST 都使用 `--pretraining-data-dir`；用户配置仍保存，但有效训练配置禁用 GAN、TrueFace、脸部和背景风格项。

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py train `
  --src E:\DFL-PT-WEBUI\workspace\pretraining\aligned `
  --dst E:\DFL-PT-WEBUI\workspace\pretraining\aligned `
  --pretraining-data-dir E:\DFL-PT-WEBUI\workspace\pretraining\aligned `
  --model E:\DFL-PT-WEBUI\workspace\model\pretrained-df `
  --archi df-ud --pretrain --rg --fp16 --device cuda:0 --steps 1000
```

在同一模型上切换到 SRC/DST 微调：

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py train `
  --src E:\DFL-PT-WEBUI\workspace\data_src\aligned `
  --dst E:\DFL-PT-WEBUI\workspace\data_dst\aligned `
  --model E:\DFL-PT-WEBUI\workspace\model\pretrained-df --resume `
  --no-pretrain --allow-config-change --reset-data-state `
  --device cuda:0 --steps 1000
```

也可建立新实例，在新建命令中加 `--initialize-from E:\DFL-PT-WEBUI\workspace\model\pretrained-df`。它自动继承检查点完整配置并令 pretrain=false；未显式覆盖时无需重填结构。显式覆盖的结构字段仍须与检查点一致。新实例只复制网络，优化器、判别器和迭代重新开始。

## 配置能力

WebUI 的“训练”页可新建 ME、继续检查点、导入旧 TF ME 网络权重。ME 向导展示全部 45 项配置，按网络结构、优化、精度、损失、对抗、数据和回放分组，并提供“默认训练”“RG + FP16”“预训练”“后期微调”四种快速配置。预设之后仍可逐项调整，确认页显示完整参数与本次配置差异。

继续检查点时，全部训练配置自动带出；默认只读。要调整学习率、精度、增强或对抗参数，先开启“允许续训配置变更”。结构、分辨率、维度和人脸范围始终固定。切换 AdaBelief/RMSprop 或修改已训练 GAN 的尺寸需“重置优化器”；更换数据集、进入或退出预训练时选择“重置采样状态”。从预训练进入微调：选择已有模型、允许配置变更、应用“后期微调”、重置采样状态，最后检查 SRC/DST 路径和确认页差异。

新模型可选择“从现有模型初始化”，自动继承来源结构与配置，令 pretrain=false，并从新的迭代和优化器状态开始。旧 TF 导入向导要求原始文件目录绝对路径、完整旧文件前缀和准确的网络结构；导入目标名称必须未被使用。

人脸集可使用当前项目 workspace 内的目录、PAK 或 ZIP。预训练需另填专用人脸集。目标迭代默认用于进度估算；勾选“达到目标后自动保存并停止”才是实际总迭代上限。完成后使用“继续训练”不会默认再次套用已经达到的自动停止上限。勾选“仅检查采样”会生成两侧增强/目标/遮罩网格，并显示 Web 预览，保留检查点；检查后仍可使用同名新模型启动正式训练。

命令行接受 `--config` 文件或 `--config-json` 内联 JSON，两者互斥。WebUI 使用内联 JSON 传递配置。GPU 可填写 `0` 或 `0,1`，索引不能重复且 batch 不得少于卡数；FP16 需要 CUDA。当前真实硬件验证为单卡。姿态评测已支持默认或自定义 aligned 目录、PAK、ZIP 和预训练共用人脸集：SRC/DST 按内容指纹与逻辑成员名固定样本，跨快照比较仍要求同一 manifest。打包、自定义或预训练样本不会误跳到默认 SRC/DST 姿态图谱。

全部默认字段见 [me-default.json](configs/me-default.json)，另有 [RG/FP16 配置](configs/me-rg-fp16.json)、[预训练配置](configs/me-pretrain.json) 和 [微调部分配置](configs/me-finetune.json)。微调文件用于合并已有检查点，不独立覆盖网络结构；从预训练切换时仍需 `--no-pretrain --allow-config-change --reset-data-state`。精确类型和范围由 `me_backend/config.py` 校验。

这些示例与 Web 预设默认把优化器留在 GPU；本机固定批次短测发现 CPU offload 明显拖慢单步，只有显存确实不足时再手动开启。RG、FP16 的速度与显存收益随配置变化，应以目标模型重跑[性能基准](../../docs/ME_PERFORMANCE_PROFILE.md)。

| 配置 | 支持范围 |
| --- | --- |
| `archi` | df/liae 加可选、不可重复的 u/d/t/c；全部 32 种组合已有 CPU 测试 |
| resolution、维度、face_type | resolution 64..640 且为 32 的倍数；ae 32..1024，e/d/mask 16..256；h/mf/f/wf/head |
| `use_rg`、`use_fp16` | RG 重算保持拓扑和权重键；FP16 使用 autocast/GradScaler，主权重与优化器状态保持 FP32 |
| GAN | gan_power 0..10；gan_patch_size 3..resolution；gan_dims 4..128；gan_smoothing/gan_noise 各 0..0.5；源脸重建的双头 U-Net 判别器 |
| `true_face_power` | 0..1，DF 潜在编码判别；LIAE 非零值会拒绝 |
| 损失、掩码、风格 | SSIM/MS-SSIM/MS-SSIM+L1；masked_training、eyes_prio、mouth_prio、background_power、blur_out_mask、face_style_power、bg_style_power |
| 优化器 | lr、adabelief（AdaBelief/RMSprop）、clipgrad；FP32 更新 |
| dropout / CPU 状态 | lr_dropout 为 n/y/cpu；cpu 在 CPU 采样 dropout；optimizer_on_cpu 在 CPU 保存和更新优化器状态 |
| warp、flip、HSV | random_warp、random_src_flip、random_dst_flip、random_hsv_power；LIAE 关闭 warp 时冻结 inter_AB |
| 额外增强 | random_downsample、random_noise、random_blur、random_jpeg、random_color |
| `ct_mode` | none/rct/lct/mkl/idt/sot/mix/fs-aug/cc-aug；参考式 SRC 颜色迁移通过 PairDataLoader 取 DST 参考 |
| 数据与采样 | aligned JPG/JPEG、PAK/ZIP、人物子目录、68 点 landmarks、XSeg/凸包及眼嘴优先掩码；uniform_yaw、data_workers 0..32（每侧独立进程数，SRC+DST 共两倍） |
| 困难样本回放 | retraining_samples、retraining_capacity 1..2048 且至少容纳一个 batch、retraining_every 1..10000；缓存随检查点保存 |

`--debug-samples` 输出 warped/target/full/priority-mask 网格并退出，不更新优化器。输入降质增强与目标分开处理；初始化和随机序列不保证与旧 TF 训练逐步相同。

## 旧 TensorFlow 网络权重导入

显式提供旧结构配置，例如 `tf-import.json` 中 archi、resolution、ae/e/d/mask、face_type 必须与旧网络一致，GAN/TrueFace 强度必须为 0。`--name` 是完整旧文件前缀，例如 `legacy_ME` 对应 `legacy_ME_encoder.npy` 等组件；目标目录必须尚不存在。

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py import-tf `
  --source E:\migration-source\tf-weights --name legacy_ME `
  --config E:\DFL-PT-WEBUI\tf-import.json `
  --model E:\DFL-PT-WEBUI\workspace\model\imported-me
```

导入器按旧 ME DF/LIAE 的命名组件契约验证全部张量；旧 RG 与非 RG 网络同拓扑。卷积从 TF HWIO 转为 OIHW，Dense 保留原 DFL 的 [input, output] 排列，保存来源 SHA-256，不需要 TensorFlow。输出可被 Web 发现，但只迁移网络权重，迭代归零；旧优化器、判别器、RNG 和数据状态不恢复。导入后可显式开启新的 RG、AMP 或对抗训练。全部架构的名称/形状映射已测试，未宣称旧训练器逐步等价。

## 预测、合成与 DFM

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\me.py infer `
  --model E:\DFL-PT-WEBUI\workspace\model\my-me `
  --input E:\DFL-PT-WEBUI\workspace\data_dst\aligned\000001.jpg `
  --output E:\DFL-PT-WEBUI\workspace\my-me-inference

.\.venv\Scripts\python.exe _internal\DeepFaceLab\main.py exportdfm `
  --model-dir E:\DFL-PT-WEBUI\workspace\model --model ME --force-model-name my-me
```

预测输出目录必须为空，生成 swap、两张掩码和 aligned 合成图。视频帧合成使用 Web“模型应用”或 `main.py merge --help`。DFM 为 ONNX，保持 NHWC BGR 接口；判别器不进入预测/DFM，RG 在推理和导出时不执行重算。

此前高级向导基线与本轮工程证据见 [ME 训练验证](../../docs/ME_TRAINING_VALIDATION.md)。既有真实单 GPU 联合训练、Web 控制与续训、原生 CLI、旧 2002 次迭代模型和 DFM 数值对照继续有效。本轮增加了分离训练进程重连、校验备份、统一数据集评测与自动化测试；固定批次 RG/FP16/CPU 优化器的八组阶段计时及配置建议见 [ME 性能对比](../../docs/ME_PERFORMANCE_PROFILE.md)。[ME 长训与成片验收](../../docs/ME_LONGRUN_ACCEPTANCE.md)核对历史连续性、固定样本预测、DFM/ONNX 数值和可选视频完整解码；本机只完成了人工素材短程技术自测。[双卡验收](../../docs/ME_DUAL_GPU_ACCEPTANCE.md)记录单卡基线及双卡硬件门槛。

真实双卡 scatter/reduce、长期真人换脸质量、最终成片视觉质量和 DeepFaceLive 现场兼容性尚未验收。先前 TF 数值对照见 [历史验证](ME_VALIDATION.md)，首轮工作台集成见 [验证记录](../../docs/VALIDATION.md)。
