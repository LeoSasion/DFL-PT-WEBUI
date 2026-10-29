# DFM 导出

ME 与 TAEQI2.1 均可在「导出 → 模型导出（DFM）」或「新建任务」中导出。先选已保存模型，再选择「正常导出」或「极致量化（FP8 / FP4）」。训练使用 FP32 或 BF16，与导出选项相互独立；原检查点及优化器状态不会被改写。

| 选项 | 模型文件 | 计算与运行要求 |
| --- | --- | --- |
| 正常导出 | `model.dfm` | FP32 ONNX，标准 NHWC BGR DFM 接口；兼容的 DFM 加载器或 ONNX Runtime |
| 极致量化 | `model_extreme.dfm` | FP8 卷积、NVFP4 Inter 全连接、FP16 敏感层；Blackwell 和 TensorRT 10.16.1 |

两种模式都附带 `.dfm.export.json`，记录来源检查点 SHA256、模型族、迭代、导出任务 ID 以及所有配套文件的大小与 SHA256。网页只有在进程成功退出且本次回执逐文件校验通过后才显示成功。

发布保留固定文件名，并使用持久事务日志和旧文件备份保护整包一致性。替换任一文件失败时会回滚图、引擎、算子检查文件和回执；若文件仍被其他程序占用而无法回滚，或导出进程被中断，`<模型文件>.publish.json` 会保留。存在该日志时，整包处于待恢复状态，不能视为成功导出，也不要复制或加载。关闭占用相关文件的程序后，重新导出同一输出路径会先恢复上一笔事务，再开始新的模型导出；已提交但清理中断的事务会保留新版本并完成清理。

多个固定文件无法同时原子替换，事务期间外部读者必须检查这个待恢复标记；WebUI 会拒绝把带标记的产物显示为本次成功结果。请保留日志及同目录 `.dfm-publish-<标识>` 备份目录，直到恢复完成。若发现产物或备份被外部修改，恢复会停止并保留证据，不会覆盖未知内容。`.publish.lock` 是可重复使用的操作系统锁文件，进程结束会自动释放锁；锁文件仍存在不代表恢复未完成。旧版唯一名称的 `.data` 权重不会因新版本发布而被删除。

TAE 的训练精度在基本参数中选择；ME 在「显存与精度」中启用 BF16，关闭 FP16。ME 两项混合精度均关闭时使用 FP32，旧 FP16 设置保留兼容。

## 极致量化的精度分配

- 满足通道对齐的中间卷积：FP8 E4M3 权重和激活。
- 满足 16 对齐的 Inter 全连接：NVFP4 E2M1，16 元素一组，FP8 局部尺度加 FP32 全局尺度；激活使用 TensorRT 动态分组量化。
- 编码器和 TAE Decoder 的首尾卷积、RGB 输出、遮罩路径与不满足对齐的层：FP16。ME 归一化的平方、均值和除法保持 FP32。
- 输入、三项输出继续使用 FP32，顺序为 DST 遮罩、换脸 BGR、SRC 遮罩。

用真实 DST aligned 人脸校准激活范围，默认最多 64 张，可选 8–512 张，至少需要 8 张。支持现有目录、PAK 和 ZIP 数据读取。校准是推理，不执行反向或更新权重。量化会带来数值偏差，尤其可能改变细节和遮罩边缘；回执包含与导出前 FP16 网络的有限实图误差，不代表身份质量或完整视频认证。

导出自动构建、重新加载并运行本机 TensorRT 引擎，附带 `.dfm.engine` 和 `.dfm.layers.json`。后者可检查实际 FP8／FP4 算子。它是可执行低精度导出，不是仅保存假量化节点。引擎与 GPU／TensorRT 版本有关，换设备需要重建；旧版 DeepFaceLive 不能直接加载这种极致量化图。本项目提供独立引擎重建／推理入口，未改造外部 DeepFaceLive。

校准只扫描一次素材清单，保留首张已校准输入的独立 CPU 快照供引擎验证使用。发布前依次检查 batch 1、2、最大批量（默认 4）及再次 batch 1 的输出形状、有限值和重复输入的一致性；批次一致性绝对容差为 0.02，结果写入回执的 `engine_smoke.batch_consistency`。这项容差用于发现批量执行异常，独立于量化误差和身份质量评估；检查失败时不会发布新模型包。

2026-10-29 的隔离小模型测试在 SM120、TensorRT 10.16.1 上复现了 NVFP4 Gemm 输出与动态量化融合后仅首行正常的问题。SM120、TensorRT 10.16 系列且包含 FP4 动态量化的网络采用构建优化级别 0，保留实际 NVFP4 CUTLASS 计算并分开执行动态量化；其他网络沿用级别 3。回执记录 `builder_optimization_level` 和 `nvfp4_output_fusion_workaround`，该策略在新建引擎时生效。真实小模型及最小算子图均验证了多批次结果；速度仍需按具体模型测量。

正常导出沿用项目基础依赖。极致量化的可选依赖记录在 `requirements-export.txt`，本机已安装验证。其他环境可使用项目 Python 安装：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-export.txt
```

该文件固定了已验证的 GraphSurgeon 0.6.1、ONNXConverter Common 1.16.0、LIEF 1.0.0 和 OnnxSlim 0.1.98，支持 ModelOpt ONNX 模块导入及 CPU INT8 图量化。GraphSurgeon 0.5.8 会调用 ONNX 1.20 已删除的 helper；0.6.1 使用官方修复，并在图导入、常量折叠和导出时保留 IR 版本。合成图验证使用 IR 10、opset 19，兼容当前 ONNX 1.20.0 和 CPU ONNX Runtime 1.23.2。

安装这里列出的依赖即可；`nvidia-modelopt[onnx]` 的完整附加依赖会在 Windows 上选择另一套 GPU ONNX Runtime，不能与已有 CPU 包混装。本项目的极致导出继续使用 ModelOpt Torch 的 NVFP4 打包路径；上述 CPU 图量化验证不代表完整 ONNX 附加包中所有 GPU 或 INT4 工具均已安装。

## 大于 2GB 的模型

不再以模型总量拒绝导出。参数达到 1.5GiB 时自动使用 ONNX External Data：小型 `.dfm` 图加带唯一名称的 `.data` 权重文件。这是包装阈值，不是模型上限。导出先追踪无参数图，再分块写入参数，避免在追踪和保存期间序列化超限 protobuf；校验使用文件路径。

复制时将回执列出的文件与回执一起放到同一目录，保持文件名。加载器必须支持 ONNX 外置权重，并按路径读取 `.dfm`；把 `.dfm` 单独读成 bytes 或只复制图文件不够。实际能力仍取决于可用内存、磁盘及目标加载器。已用合成压力测试网络的单个 2,147,483,712 字节外置参数完成导出、整包移目录及双样本 ONNX Runtime 推理测试。

## CLI

从仓库根目录运行。示例中的模型名与人脸集按当前项目替换：

```powershell
.venv/Scripts/python.exe _internal/DeepFaceLab/main.py exportdfm --model TAE --model-dir workspaces/PROJECT/model --force-model-name MODEL --export-mode normal
.venv/Scripts/python.exe _internal/DeepFaceLab/main.py exportdfm --model ME --model-dir workspaces/PROJECT/model --force-model-name MODEL --export-mode extreme --calibration-dir workspaces/PROJECT/data_dst/aligned --calibration-samples 64 --force-gpu-idxs 0
.venv/Scripts/python.exe tools/dfm-runtime.py build PATH/model_extreme.dfm --device cuda:0 --max-batch 4
.venv/Scripts/python.exe tools/dfm-runtime.py infer PATH/model_extreme.dfm.engine input.npy output.npz --device cuda:0
```

`input.npy` 为 `[N,H,W,3]`、BGR、0–1 的 FP32 数组；引擎默认 batch 范围 1–4。输出 NPZ 含 `dst_mask`、`swapped_bgr`、`src_mask`。重建会更新引擎及算子检查文件；跨设备重建后的引擎不再具有原导出回执的 SHA，需要保留重建来源并重新记录本地产物。

量化契约参考：[NVIDIA Model Optimizer ONNX 导出示例](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/torch_onnx/README.md)、[ONNX External Data](https://onnx.ai/onnx/repo-docs/ExternalData.html)。未认证完整视频、外部 DeepFaceLive 现场兼容性或跨设备加速幅度。
