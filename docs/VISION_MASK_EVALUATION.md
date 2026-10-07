# 遮罩候选功能验证（2026-10-06）

现有 XSeg WF、BiSeNet CelebAMask-HQ 与 SAM2.1 Hiera large 已在同一组 6 张 512×512 对齐图完成原生 PyTorch 冻结推理，共输出 18 张二值遮罩及 6 张 BiSeNet 类别图。没有训练、随机替代权重、H3CE AGPL 实现复制或生产 DFL 遮罩写入。

**这次只证明候选能够运行；没有人工遮罩真值，IoU、Dice、winner 均为 null。** 三模型的语义不同，前景面积不是质量得分，不能直接选出替代 XSeg 的模型。

| 候选 | 输入及输出契约 | 官方资产与许可证 |
| --- | --- | --- |
| XSeg WF 基线 | RGB 转 BGR、Lanczos4 到 256，浮点 [0,1]；既有通用 WF 权重，阈值 0.5，输出回到输入大小 | 仓库现有 verified 通用权重；保留 SOURCE.json、原 GPL-3.0 许可证及转换来源 |
| BiSeNet CelebAMask-HQ | RGB 512、ImageNet mean/std；19 类 argmax。用于功能展示的二值并集是 1..8、10..13；排除背景、耳饰、颈部、项链、衣物、头发、帽子 | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)，MIT；README 官方 GDrive 权重 `154JgKpzCPW82qINcVieuPH3fZ2e0P812` |
| SAM2.1 Hiera large | RGB；固定相对框 [0.16W,0.12H,0.84W,0.90H] 与正点 [0.50W,0.48H]，单 mask。对象区域二值分割，无面部类别 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2)，Apache-2.0；[官方 checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

SAM2 提示在推理前固定，不根据预测调整；其 predictedIoUConfidence 是模型估计值，并非测得 IoU。使用 `apply_postprocessing=False`，hole/sprinkle area 均为 0，不编译 CUDA 扩展、不进行视频传播。本机使用既有 Torch 2.9.1+cu128，加 hydra-core 1.3.2、iopath 0.1.10；直接加载 ignored 官方源码，未安装训练工具或变更 Torch。

共享素材沿用 restoration/six-source-v1 的哈希锁定 `reference.png` 与 `valid-mask.png`。反射填充边界排除在面积统计之外；保存的遮罩是原模型完整输出。素材是 FFHQ 五点对齐而非 DFL WF，所以 XSeg 的这次结果属于跨对齐功能探测，不构成 WF 质量验收。对齐点也不是人工遮罩标注。

资产仅在 ignored `workspace/.vision-models/masks`。`assets.json` 记录原仓库 revision、官方 URL、每个 Python/YAML 源文件、许可证及权重 SHA-256；adapter 同时固定 reviewed 源清单摘要和权重，验证后才导入加载。BiSeNet 构造器原本会下载 ImageNet 初始化，adapter 仅禁用这一步，随后 strict 加载完整解析 checkpoint；推理不下载。

| 候选 | 官方源码 revision | 权重 SHA-256 |
| --- | --- | --- |
| XSeg WF | e4b7543ffa1d73b26fce1e31852727f658ba490c（原始来源） | 26e45677ef3136e0327f0fd51e452cbea81a703ee7fea9121b01ba58da65c385 |
| BiSeNet | d2e684cf1588b46145635e8fe7bcc29544e5537e | 468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567 |
| SAM2 | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318 |

运行工具 `tools/vision-mask-benchmark.py` 需要显式共享输入、全新输出目录与本地已验证资产。输出报告带每图 input/mask 哈希、有效像素域、语义/提示、面积统计和 frozen/eval 参数证据；不会修改原对齐图、模型、XSeg 多边形或 xseg_mask。缺人工真值时不创建准确率排序或自动晋升。

真实证据保存在 ignored `workspace/.vision-evaluation/masks/six-source-v3/report.json`。所有 3×6 推理成功。独立契约测试覆盖缺真值 null、有效域统计、输入验证、固定提示、语义类别并集、篡改哈希与目录逃逸拒绝。后续若需要比较质量，应先统一目标遮罩语义、用符合 WF 的对齐数据并建立人工真值，再开展独立验收。

## 原生 WF 辅助遮罩流程

`bisenet-face-preserve-v1` 是新增的辅助 profile，不淘汰通用 XSeg。`webui/python/mask_assist.py` 读取真实 DFL whole_face JPEG 的 68 点 aligned 像素坐标，拒绝普通 FFHQ 图片、PNG、98 点数组及非 WF face_type。模型在同一 aligned 图上运行，类别图回到原像素网格；face-only 并集显式包括眼睛、眼镜、嘴和上下唇，不进行腐蚀或用预测关键点强行填补遮挡孔洞。遮挡与深色区域误分类仍需复查。

先生成原始字节备份、新 JPEG 副本、二值 mask、类别图与 overlay。只在副本中写入兼容 DFL 的 embedded `xseg_mask` 和 `mask_assist` 算法来源字段；回读证明像素完全相同，landmarks、source 信息及已有人工 `seg_ie_polys` 未变，embedded mask 的完整浮点数组与二值 PNG 精确一致。不会产生人工 polygon、IoU/Dice 或伪造人工复查结论。通用 XSeg 的 verified 权重、原始 aligned 中已有的 XSeg 遮罩均不覆盖。

`mask-assist-manager.mjs` 支持带 signal/onProgress 的草稿生成与完整新 faceset 发布，操作由后台管理器调度。每个生成 pass 1..500 张；发布需用户复查确认，保留 `data_SIDE/aligned`，将选中副本与其他所有原 aligned 图片独立复制到 `data_SIDE/aligned_assisted/<id>`。不用 hardlink，避免之后编辑副本影响原文件。发布前后验证哈希与成员列表；变化/取消时不发布半成品。完整新数据集路径可预填训练 faceset，原数据仍可直接选择。

Windows 推理取消、超时及 stdout 超限均终止整个进程树，等待 venv 启动器的 close 和进程树退出核验后才返回；无法核验时返回 `MASK_STOP_UNCONFIRMED`。隔离的标准库 sleep 测试实际观察了三组启动器/解释器 PID，三个出口返回时各组两个 PID 均已退出，没有用模型或训练替代这项生命周期验证。

发布 receipt 存在新 faceset 的 `mask-assist-provenance.json`，只读查询草稿也返回 publication；同 ID 发布重试幂等返回既有结果 `reused:true`，不再次复制。丢失响应或操作历史中断后可先复查已发布数据集，再决定使用，不能把运行日志视为完成证明。

真实 WF 验证使用此前有限 ME 功能验收的既有原始 aligned，SRC/DST 各 6 张，推理前固定低锐度、预测姿态/嘴部几何极值和清晰 control，**没有再次训练**。12/12 原生推理、副本像素/元数据保留及 mask 回读通过。来源 SHA、selection、report、原始/副本/overlay 和四栏视觉复核图在 ignored `workspace/.vision-evaluation/mask-assist/wf-twelve-v1`。这组与先前 FFHQ 功能探针独立；难例代表性与质量由视觉复核单独限定，尚无人工遮罩真值。

GPT-6 Astra（low）的独立视觉记录是 `workspace/.vision-evaluation/mask-assist/wf-twelve-v1/astra-production-review.json`，处置为 `accept_with_review`：12 张显示图中的眼鼻嘴核心被保留，发际/下颌有局部边界偏差，没有列出阻断项。这只支持进入人工复核的辅助流程；样本外观相近且含近重复，不能算作 12 个独立场景的泛化证明。本批真实遮挡不足，`occlusionHandling:null`；闭眼、闭嘴不能当作应挖除的遮挡。Astra 视觉审查不是专家逐像素真值，IoU/Dice 仍为 null，也不支持未经复查自动用于全部训练/合成。
