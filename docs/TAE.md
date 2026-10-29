# TAEQI2.1 训练器

2026-10-10：按用户决定，用 TAEQI2.1 替换 H3 训练器；ME 与 XSeg 保留。入口是 `train.tae` / `merge.tae`，内部模型类型 `TAE`，独立检查点 `tae.pt`。旧 H3 源码和本机旧 H3 模型已移入带 SHA 清单的“待手动删除”目录；不迁移 H3 权重，不依赖 H3CE 代码或资产。

## 使用

2026-10-14：新增原生 512 预设。CLI `--pretraining` 要求 SRC/DST 指向同一训练池，两路使用独立 RNG 和曝光计数；按可用来源标识/人物标签/连续帧文件名前缀确定性留出约 5% 组。没有人物标签时不能声称身份独立验证。逐文件校验与完全相同文件去重结果写入 `pretraining-data.json`、`pretraining-manifest.jsonl`；原包保持只读。大包索引显式使用有界的 512 MiB 数据解释器，普通训练仍沿用 128 MiB 限制。预训练检查点必须带 `--pretraining` 续训，不能隐式切换到身份训练。

1. 在 WebUI“新建任务”选择“训练 TAEQI2.1”，创建模型。普通训练选择 SRC/DST；通用底模勾选“通用底模预训练”，只填写一个“预训练人脸集”（当前项目内的 aligned、PAK 或 ZIP）。续训会恢复预训练模式和素材池，模式不可隐式切换。
2. 人脸类型须与素材元数据一致；原生目标像素至少达到所选分辨率。预设为 256、448、512、640、832，默认 256 / whole face / batch 1。已有 full face 数据应选择“全脸”。
3. 选择 Decoder 训练范围：默认冻结，也可仅训练末段或训练全部。Encoder 始终冻结。默认 Inter 瓶颈 256、特征宽度 96、学习率 1e-4、预热 50 步。
4. 基础权重路径留空使用本机 `_internal/DeepFaceLab/pretrained/taeqi2_1.safetensors`。显式路径或 `TAE_VAE_WEIGHTS` 可重定位同一资产；运行前严格校验 SHA，不自动下载、不执行远端代码。
5. 训练提供真实预览、分项损失、保存和安全停止。时限默认 60 分钟，从训练循环开始计时，完整更新后保存停止；加载和收尾另计。可另设总迭代上限。
6. 继续训练会恢复权重、AdamW、随机状态、采样进度和曝光计数。Decoder 范围、倍率、训练精度、Batch size、RCT 和遮罩训练范围可修改：修改任一项会保留所有模型权重、总迭代和数据状态，**重置全部 AdamW 与学习率预热**，记录新的优化阶段。不会定时自动解冻。结构、损失权重及其他增强参数锁定；更换素材仍需明确重置数据状态。
7. 在“模型应用”选择“合成 TAEQI2.1 人脸”，复用 DFL 的遮罩、回贴和视频导出流程。

模型目录含 `tae.pt`、`metadata.json`、`loss-history.jsonl` 和 `preview.png`。TAE 已支持正常及极致量化 [DFM 导出](DFM_EXPORT.md)，导出包括冻结 Encoder、实际保存的 Decoder 和独立遮罩头。备份恢复点和姿态评测快照暂未接入。安全停止必须收到匹配的保存确认且进程正常退出。

训练精度可选 `fp32` 或 `bf16`（默认）；主权重和 AdamW 状态均为 FP32，CPU 始终使用 FP32。续训切换精度与改变 Decoder 范围一样，会保留权重、数据状态和累计迭代，重置 AdamW 与预热并记录新阶段。旧检查点自动沿用原 BF16 配置。导出精度单独选择，不会量化训练权重。

## 网络与训练范围

```text
RGB + alpha=1 -> 冻结 Encoder -> 64 通道 latent（R/16）
    -> 独立 InterAB / InterB：两次步幅卷积 -> R/64 网格
    -> 全局 Dense 压缩/展开 -> AB+AB 或 B+AB
    -> 1×1 融合 -> 两级上采样 -> 64 通道 latent
    -> TAEQI2.1 Decoder -> RGB

融合特征 -> 独立监督的脸部遮罩头
```

SRC 重建使用 AB+AB，DST 重建使用 B+AB，换脸使用 AB(DST)+AB(DST)。没有原始像素或原始 latent 直通输出的旁路，不用 DST 本人的 RGB 监督换脸分支。Decoder 输出的 alpha 不作为脸部遮罩。

Encoder 在 `no_grad` 下运行；像素损失通过 Decoder 的实际算子反传到 Inter，即使 Decoder 冻结也保留反向图。保留上游 latent 的原生 tanh clamp，训练 RGB 不做显示截断。CUDA 使用 FP32 主参数，可选 BF16 autocast，关闭 TF32，不使用 GradScaler；无 BF16 的 CUDA 设备可选 GPU FP32。训练不是 INT8 模型。

| Decoder 范围 | 更新参数 | Decoder 参数数 | 默认学习率 |
|---|---|---:|---:|
| 冻结 | Inter、融合、上采样、latent 和 mask 头 | 0 | 不更新 |
| 仅末段 | 以上＋原生 Decoder 第 12 层起 | 562,960 | Inter 的 0.1 倍 |
| 全部 | 以上＋完整 Decoder | 7,644,560 | Inter 的 0.1 倍 |

Encoder 参数 7,644,416，任何模式都不更新。所有模式的检查点都保存完整 Decoder，确保切回冻结后仍使用已训练权重。仅 Encoder 从固定基础资产恢复。检查点另外绑定基础 SHA、源代码 SHA、数值契约、架构、阶段历史及逐参数 AdamW 状态。

| 分辨率 | Inter 网格 | Decoder 冻结时可训练参数 |
|---|---:|---:|
| 256 | 4×4 | 3,131,761 |
| 448 | 7×7 | 6,382,129 |
| 512 | 8×8 | 7,859,569 |
| 640 | 10×10 | 11,405,425 |
| 832 | 13×13 | 18,201,649 |

这些预设不使用 H3 分块，也不代表已比较出最佳换脸尺寸。Inter 的 Dense 大小绑定分辨率，不支持续训时改变。

损失保留 RGB、局部 DSSIM、边缘、眼嘴区域和独立遮罩监督。干净目标只做共同翻转；随机形变和可选半尺寸退化只作用于输入。顺序处理 SRC/DST 反向后执行一次 AdamW 更新。有限梯度检查、梯度裁剪及冻结张量版本检查会拒绝异常更新。

### 色彩迁移

基础学习率 `learning_rate` 可在停止后续训时调整。与其他可变训练设置一样，保留权重、总迭代和采样状态，记录含基础学习率的新优化阶段，并重置 AdamW 与预热。旧检查点的采样指纹仍包含原学习率，恢复时验证对应旧策略再迁移，不要求重置素材状态。当前策略为线性预热后固定学习率，没有自动下降调度器。

随机形变 `random_warp` 可在停止后续训时切换；保留权重、总迭代和采样进度，并记录新优化阶段、重置 AdamW 与预热。旧采样指纹包含此字段，续训时验证原策略后迁移到新指纹，不需要重置素材状态。

续训需要沿用原训练总时限时，设置 `--stop-at-utc`（WebUI 的固定截止时间），使用带时区的 ISO 时间。绝对截止时间包括本次模型加载和素材检查；训练器取固定截止时间与 `--max-minutes` 的较早者，转换为单调时钟期限，到期保存停止。固定截止时间已过时不进行新的优化更新。

`ct_mode` 支持关闭（`none`）、RCT（`rct`）和 LCT（`lct`）。两种迁移都复用项目的原生色彩变换，以随机 DST 样本为参考，同时处理 SRC 输入和重建目标，保持遮罩与 DST 不变；LCT 使用线性色彩迁移。通用预训练的参考来自同一公共素材池，SRC/DST 保持独立采样。停止后切换迁移方式保留权重、总迭代和采样状态，重置 AdamW 与预热并记录新优化阶段。

### 遮罩训练范围

`training_mask_mode` 有三档，旧检查点缺省为 `face`，不会改变原有训练行为：

- `face`：原始人脸遮罩，保持原来的 RGB、DSSIM 和边缘监督。
- `expanded_feather`：从原始遮罩的 0.5 轮廓向外按欧氏距离增加分辨率的 10%（四舍五入为像素）。轮廓内权重为 1，整个新增范围按 `max(0, 1 - 距离 / 外扩半径)` 从边界的 1 线性降到外缘的 0，不保留全权重外扩平台。512 分辨率的半径和羽化宽度均为 51 像素；图像边界自然裁切。原始软遮罩仅用于定位轮廓，不让其微弱背景残值越过外缘。
- `full`：整张对齐裁剪图参与图像重建。

三档均保留原始人脸遮罩作为遮罩头的监督目标，眼嘴优先区域不扩大；不改变合成遮罩，也不模糊背景图像。ME 的 `blur_out_mask` 是独立的背景图像处理功能，不与此项合并。外扩权重在 CPU 数据准备时生成，与预览使用同一实现，不通过生成式图像工具模拟。

2026-10-24 数值复现修正：`expanded_feather` 的精确距离变换在调用期间绕过 OpenCV 的 IPP 路径，并在正常或异常退出时恢复当前线程原有设置。[OpenCV 4.12 将此开关保存在当前线程状态中](https://github.com/opencv/opencv/blob/4.12.0/modules/core/src/system.cpp#L2581-L2604)；其他线程及后续图像处理继续使用原有设置。此修正保留欧氏距离、0.5 轮廓、外扩半径和 FP32 羽化公式，不改变检查点格式或采样指纹，也不重写已有模型。

本机 OpenCV 4.12 / IPP AVX2 2022.1 对同一二值 mask 的重复调用出现约 `9.54e-7` 的距离差，形成约 `1.19e-7` 的图像损失权重差；开启 PyTorch 确定性算法无法约束这段 CPU 预处理。上游记录见 [OpenCV #24082](https://github.com/opencv/opencv/issues/24082) 与 [4.12 距离变换实现](https://github.com/opencv/opencv/blob/4.12.0/modules/imgproc/src/distransform.cpp#L728-L775)。因此不能把这次失配直接归因于 BF16 或 GPU 反向计算。

隔离合成数据验证覆盖五个分辨率的逐位解析几何校验、内存布局与 IPP 开关变化、并发线程和异常恢复。在同一 RTX PRO 6000 Blackwell / PyTorch 2.9.1+cu128 环境、固定种子与确定性设置下，256 FP32、256 BF16 和 512 BF16 的三组小模型均完成 8 步 A/A 与同步策略 A/B：每步输入、损失、权重和 AdamW 状态逐位相同；第 4 步保存后恢复到第 8 步也与连续运行相同。该证据不承诺跨硬件、库版本或不同训练配置的逐位复现，也不构成已有模型画质结论。

停止后续训修改此模式会记录新优化阶段，保留权重、总迭代和采样状态，重置 AdamW 与预热。该选项提供可验证的监督范围，不代表已验证光照、脸型或最终换脸质量的提升。

## 来源与许可

固定基础资产 SHA256：

```text
6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e
```

源代码为 [madebyollin/taesd](https://github.com/madebyollin/taesd/tree/027d3ef404a04a141e60d5ec0fee149c2c8bdd74)，模型为 [TAEQI2.1 固定版本](https://huggingface.co/madebyollin/taeqi2_1/tree/379cc3f29b1fdbf3bc02781792d88f81036780da)。上游 MIT 许可和精确文件来源见 `_internal/DeepFaceLab/tae/_vendor/PROVENANCE.json`、`LICENSE.taesd`。基础文件 30,591,264 字节，在本机已校验；不将用户媒体或 QA 检查点纳入源码。

## 命令行

从项目根目录执行，数据几何应与 `face_type` 一致：

```powershell
.\.venv\Scripts\python.exe _internal\DeepFaceLab\tae.py web-train --src workspace\data_src\aligned --dst workspace\data_dst\aligned --model workspace\model\tae_demo --name tae_demo --config-json '{"resolution":256,"face_type":"f","decoder_mode":"frozen"}' --max-minutes 60
```

同一命令添加 `--resume` 可继续；改变 `decoder_mode` 为 `tail` 或 `full` 会开始新阶段。`--steps` 限制本次更新，`--target-iterations` 限制总迭代。检查数据可使用 `tae.py validate-data`；查看配置可使用 `tae.py inspect <tae.pt>`。`main.py train --model TAE` 和 `main.py merge --model TAE` 同样受支持。

## 验证范围

当前验证记录见 [接入验收](TAEQI21_INTEGRATION_20261010.md)。本次是有界功能接入，不是正式效果训练；不据短跑损失、合成成功或形状检查声称最终身份保持、跨视频质量、时序稳定、FPS 或全分辨率训练效果。


- 2026-10-20: TAE stopped resume may adjust eyes_mouth_weight as an explicit stage; retain weights/iterations/sampling, reset AdamW/warmup, migrate exact predecessor sampler fingerprints. Other loss weights remain immutable.
