# ME 后端原型的历史数值验证

下列记录在本次迁移前的独立 ME 原型执行，保留以说明网络和损失的数值对照。其 me-workspace/me-evidence 原始材料仍在原型目录，并未复制到新仓库。当前训练能力与证据见 [ME 训练验证](../../docs/ME_TRAINING_VALIDATION.md)，首轮 Web/工具链集成见 [验证记录](../../docs/VALIDATION.md)。本文的 non-RG、FP32 和 10 项测试描述仅对应历史原型。

# ME PyTorch 首轮验证记录

日期：2026-10-01。本记录对应本目录 `me.py` 的初始实现，具体使用范围见 `ME_README.md`。

## 实机训练、保存、续训与预测

- GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition。
- 运行时：Python 3.12.14，PyTorch 2.9.1+cu128，完整 FP32；TF32 关闭。
- 配置：non-RG liae-ud，分辨率 128，ae/e/d/mask = 256/64/64/22，batch 4。
- 参数量：82,376,555。
- 输入：源/目标各 8 张人工生成的 aligned DFLJPG；每组均覆盖 XSeg 掩码与 landmarks 凸包掩码。
- 完成从零训练 20 步，退出进程，加载检查点后再训练 5 步，最终迭代 25。
- 独立预测进程成功导出换脸预测、源掩码、目标掩码和 aligned 合成图。
- 第 1 步 src/dst loss：5.250675 / 4.942726；第 20 步：3.591920 / 3.134672。
- 20 步运行的 CUDA 峰值 allocated memory：1739.02 MiB；不包含 CUDA 上下文或其他进程。
- 排除首次编译/预热后，单次 `train_step` 中位数约 45.6 ms；不含图片读取、保存检查点，不代表真实大规模数据训练吞吐。

这是流程验证，不是人脸质量或长期训练稳定性验收。合成数据上的短期损失下降不能证明实际换脸效果。

本机文件：

- `me-evidence/final128-train.log`
- `me-evidence/final128-resume.log`
- `me-evidence/final128-infer.log`
- `me-evidence/final128-inference/`
- `me-workspace/final128/me.pt`
- `me-workspace/final128/preview.png`

## 独立 TensorFlow 数值对照

使用现有 TensorFlow 2.10.1 CPU 解释器运行原版网络与本地 ME 损失函数源码。64 分辨率小网络、56 个权重张量按完整名称和形状逐一映射，无缺失权重或按形状猜测。输入、权重来自固定随机种子。

| 张量或函数 | PyTorch CPU 最大绝对误差 | PyTorch CUDA 最大绝对误差 |
| --- | ---: | ---: |
| Encoder | 4.89e-6 | 7.15e-6 |
| Inter AB | 3.95e-7 | 1.60e-6 |
| Inter B | 3.28e-7 | 1.64e-6 |
| Swap / dst 图片及掩码 | 1.20e-7 | 1.20e-7 |
| 输入梯度 | 4.73e-13 | 4.20e-13 |
| Gaussian blur | 2.39e-7 | 1.79e-7 |
| DSSIM，偶数核 6 | 2.39e-7 | 2.09e-7 |
| Style loss | 1.51e-9 | 1.52e-9 |
| MS-SSIM | 7.16e-7 | 1.20e-7 |
| MS-SSIM+L1 | 7.01e-7 | 1.50e-8 |

容差：`rtol=2e-4, atol=2e-6`。网络较大的中间值适用相对容差。数组及机器可读报告位于 `me-evidence/tf-reference/`。

该对照覆盖网络前向、一个预测目标的输入梯度以及上述损失原语；没有宣称整个旧 ME 训练器逐步等价、所有训练选项等价或旧优化器检查点兼容。

复现命令（TensorFlow 解释器路径按本机布局）：

```powershell
..\DFL-WEBUI\_internal\python_common\python.exe -B tests/generate_tf_reference.py `
  --tf-repo ..\DFL-PyTorch-Audit-20260930\DeepFaceLab-TF `
  --me-root ..\DFL-WEBUI\_internal\DeepFaceLab `
  --output me-evidence/tf-reference
.\.venv\Scripts\python.exe -m tests.verify_tf_reference --reference me-evidence/tf-reference --device cpu
.\.venv\Scripts\python.exe -m tests.verify_tf_reference --reference me-evidence/tf-reference --device cuda
```

## 自动化与环境检查

`python -m pytest tests/test_me.py -q`：**10 passed**。包含三种损失的训练更新、准确恢复优化器/RNG 后逐位相同的 CPU 下一步、XSeg 解码、眼嘴 mask、学习率 dropout 重采样、冻结 inter_AB、数据集变更和不完整检查点拒绝。

`pip check`：无损坏的依赖。`git diff --check`：通过。原 DFL-WEBUI 工作树仍为空差异。

## 本轮修复的复用代码问题

1. Encoder 层列表改为 ModuleList，确保标准 `parameters()` / `state_dict()` 包含全部参数。
2. ModelBase 避免重复构建，模型和层恢复标准 PyTorch Module 调用路径。
3. Depth-to-space 使用 DFL/TF 的通道排列，替代不等价的 pixel shuffle。
4. Pixel normalization epsilon 从 1e-8 修正为原 ME 的 1e-6。
5. Flatten 接受非连续输入，保证从 aligned 图片转置后能预测。
6. XSeg 压缩数据在解码前兼容数组、列表和字节；修复上游 JPG 保存后读取失败。

ME 使用独立实现的损失、优化器和完整检查点，不通过旧 SAEHD 训练循环或旧 Saveable 宽松加载器。未验证的其他上游模型不因此被视为已修复。
