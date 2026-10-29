# GPU 兼容与运行时选择

项目保留一份生产代码和一个 Python 3.12 基础运行时。每台工作站只安装与其显卡匹配的一套 `.venv`；源码安装与便携包构建共用固定版本、官方 wheel 哈希和依赖清单。

| 显卡 | PyTorch / TorchVision | CUDA 构建 | 训练与导出 |
| --- | --- | --- | --- |
| GTX 10 系列，Pascal SM 6.1 | 2.9.1+cu126 / 0.24.1+cu126 | 12.6 | FP32 训练、预测、普通 DFM 导出 |
| Blackwell SM 10.0 / 12.0 | 2.9.1+cu128 / 0.24.1+cu128 | 12.8 | FP32／原生 BF16 训练；满足额外要求时可极致导出 |
| 其他显卡 | 根据所选 wheel 实际内核与设备计算能力判断 | 新安装默认 cu128；需要 Pascal 兼容时 cu126 | BF16 和极致导出分别检查硬件能力 |

官方 Windows cu126 构建包含 SM 5.0、6.0、6.1、7.0、7.5、8.0、8.6、9.0；cu128 构建包含 SM 7.0、7.5、8.0、8.6、9.0、10.0、12.0。单个官方构建无法同时覆盖 Pascal 和 Blackwell。训练在分配 GPU 模型前检查选中的设备；同一 major 下更低 minor 的 cubin 可以复用，`compute_xx` PTX 按前向兼容规则判断。

来源：[官方 Windows cu126 构建配置](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/.ci/pytorch/windows/cuda126.bat)、[cu128 构建配置](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/.ci/pytorch/windows/cuda128.bat)、[PyTorch 2.8 起的架构支持变更](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)。

## 安装与已有环境

新源码安装默认自动检测驱动报告的计算能力，检测到 Pascal 时选择 cu126，Blackwell 选择 cu128；无 GPU 或检测失败时默认 cu128，也可显式选择。已有环境按已安装构建复用，不会因当前显卡检测结果自动切换版本。安装命令仍拒绝静默替换已有构建；显卡环境变化后，使用启动器提供的明确下载与切换操作。

在新的源码目录中可明确选择：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File launcher/install-source.ps1 -CudaRuntime cu126
```

自动检测到同机同时存在 Pascal 和 Blackwell 时，新安装要求明确选择一个构建；训练只选择该环境支持的设备。已有环境仍可使用其中受支持的卡。不能把两种 CUDA wheel 混装进同一个 `.venv`。

## 显卡环境变化后的提示与切换

启动器在打开、重新获得焦点及可见且空闲期间检查显卡环境。当前运行包不能适配检测到的显卡时，提示匹配的 CUDA 版本和下载量；读取不到显卡信息时保留现有环境，不猜测需要换包。下载由用户点击“下载并切换”开始，不会在后台自动进行。

切换先下载并核对官方 Torch/TorchVision 安装包的大小与 SHA-256，在独立目录安装并执行版本、二进制内核、cuDNN、安装记录和小计算验证。验证通过且项目任务已结束后，才切换两个包所拥有的目录和入口。Python 基础运行时、其他公共依赖、素材、模型及训练参数保留；工作站仍只有一套活动 `.venv`。

新版本启用并再次验证成功后，启动器显示“保留旧版 / 删除旧版”。默认焦点在保留操作。删除需要单独确认，只清理本项目保留的旧运行包；不卸载显卡驱动、系统 CUDA 或其他依赖。保留版本以后可以用于再次适配，避免重复下载；已删除的版本需要时重新下载。

选择保留后，仍可从“清理保留的旧版”或“管理保留的运行包”查看历史版本，按版本、空间和事务记录分别决定是否清理。已经开始但未完成删除的版本只能继续清理，不能再标记为完整保留或用于复用。检测结果按项目缓存并合并并发查询；每次切换、保留、删除和恢复都绑定发起操作的项目，切换项目后的旧页面请求会被拒绝。

重试会先检查完整安装包缓存，实际使用前重新验证 SHA-256；界面区分缓存大小与本次还需下载的大小。半截下载不作为完整安装包复用。失败安装和回滚留下的临时目录，只按对应事务的包所有权记录清理；没有来源记录的文件和目录链接不会被递归删除。清理失败会显示可重试的状态；仅缓存或临时文件清理失败时，已经启用并验证通过的运行环境继续保留，重试清理不回退环境。

下载和切换使用项目内的独占操作锁与持久事务记录。训练、WebUI 和终端运行时不能切换或清理；正在切换时也不能启动新的项目任务。失败时保留旧环境并尝试回退；若意外退出留下未完成事务，启动器提供“恢复原运行环境”，恢复完成前不使用不完整的包目录。路径异常或出现未知文件时保留证据并停止，不能把恢复或删除目标改成素材、模型或任意用户目录。

保留包与事务记录位于忽略的 `.runtime/torch-variants`。它们不进入源码包或便携包；便携包只收集已选中的活动运行环境。选择保留会占用额外磁盘空间，启动器显示相应大小。切换不会自动改变模型精度，GTX 10 系列训练仍需明确选择 FP32。

`release/python-sbom.json` 和现有功能配置锁保留 cu128 基线。`release/torch-runtimes.json` 记录 cu126 两个完整官方包记录；统一解析器只替换 Torch 和 TorchVision，保留其余已验证依赖，并生成当前安装过程使用的完整带哈希锁。离线 wheelhouse 必须包含所选构建的对应文件，cu128 的文件及哈希不能代替 cu126。

只读验证当前环境：

```powershell
.venv/Scripts/python.exe -I -B tools/torch-runtime.py validate
```

便携包构建通过 `tools/build-release.py ... --cuda-runtime cu126` 选择旧卡变体，并要求用于打包的实际环境与该变体一致。现有 cu128 便携包仍不能在 GTX 10 上执行 CUDA 训练；应分发对应的 cu126 包或按源码安装。没有在本轮发布新的便携归档。

## CUDA、cuDNN 与精度

CUDA 和 cuDNN 使用所选官方 PyTorch wheel 自带的版本，不单独覆盖 DLL。当前本机 cu128 为 CUDA 12.8、cuDNN 9.10.2。cuDNN 9.10.2 的官方矩阵仍列出 Pascal；更新 PyTorch 或 cuDNN 前必须重新核对，CUDA 13 不能作为保留 Pascal 的升级方向。[cuDNN 9.10.2 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)、[NVIDIA 架构支持说明](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)。

GTX 10 系列还需保留支持 Pascal 的显卡驱动。NVIDIA 将 R580 列为支持这些架构的最后一个驱动分支，支持期延续至 2028 年；本项目不会安装或替换系统驱动。Blackwell 工作站继续使用与本机硬件匹配的驱动。

GTX 10 系列选择 FP32。BF16 门槛明确排除软件模拟；已有 TAE 检查点需要改变精度时，仍按新训练阶段规则处理，不能静默改写精度。极致 FP8/NVFP4 导出保持 Blackwell 门槛，不影响旧卡的普通 FP32 导出。

CUDA 编译缓存尊重已有 `CUDA_CACHE_PATH` 和 `CUDA_CACHE_MAXSIZE`，只有未设置时才使用默认值。不会为优化而全局开启 cuDNN benchmark、TF32 或更换优化器。

## 验证范围

本轮使用隔离小模型、合成数据、安装流程替身和计算能力模拟验证代码路径。当前 Blackwell 环境的真实单卡训练、保存、恢复继续验证；官方 Windows 构建配置与候选 wheel 取证用于检查 Pascal 内核。没有 GTX 10 实机，不能将这些证据表述为 GTX 10 实机训练、吞吐或画质验收。真实双 GPU、长训与正式画质不在本轮范围内。
