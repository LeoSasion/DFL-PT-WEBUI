# 验证记录（2026-10-01）

本页保留首次工作台集成验收的环境、测试数字和证据。2026-10-02 新增 ME 架构、RG、AMP、对抗训练、数据和迁移能力的当前验收见 [ME 训练验证](ME_TRAINING_VALIDATION.md)；下文 29 项 Python 测试和旧 CLI 数字属于首轮记录。

这是新 `E:\DFL-PT-WEBUI` 的实测记录。所有训练和视频数据均为人工生成的 DFL aligned 图片及测试帧，不含用户素材。原型的 TF 数值对照在后端 ME_VALIDATION.md 中单独标为历史记录，不能替代本次集成测试。

## 环境和独立性

- 本地 Python 3.12.14，PyTorch 2.9.1+cu128；GPU 为 NVIDIA RTX PRO 6000 Blackwell。
- Python base、venv、Node 24.19.0 和 FFmpeg 都在新目录内；pip check 通过。
- 将 Python base、5.68GB 完整 venv、Node 和 FFmpeg 实体复制到另一个目录后，离线 bootstrap、全部六项依赖导入及 CUDA 张量计算通过。证据在 launcher/relocation-runtime-evidence.json。
- 旧 DFL-WEBUI 的 Git 状态仍为干净，HEAD 为 85fac83490e06478a266661e1acf7714e82e9aff。
- 新仓库没有远端，Git 忽略运行环境、权重、个人工作区、模型、日志和测试副本。

## Web ME 训练和控制

Chrome headless，1440×1000，实际页面 http://127.0.0.1:4173。

在人工 `me-verification` 项目中从任务向导建立 ME：128 分辨率、ae/e/d/mask 256/64/64/22、batch 4、GPU 0。预览与真实 loss/iteration 出现在总览。最终检查点迭代 2002。

保存、备份、刷新预览、评估和 close 均由 Web 控制触发，并核验后端完成记录。安全停止保存第 1993 步后，重载网页选择同一模型、启动另一个训练进程，继续到第 2002 步，再次保存停止。两次退出码均为 0；close ACK 的 requestedAt 与本次停止请求相同。

评估生成真实只读快照；备份目录有检查点与元数据。恢复训练保留检查点配置，没有重新随机初始化。测试过程中无页面异常或相关 console error。

## 合成、DFM 和 MP4

使用正确 source_landmarks 的人工 128×128 帧执行主 CLI 的 guided Merger，得到一张合成帧及遮罩；遮罩有 16273 个非零像素，输出图像与输入不同。learned 乘积模式不需要 XSeg，也没有打开原生交互窗口。

完整 128 模型导出 DFM，大小 329664699 字节。ONNX Runtime CPU 与 PyTorch 同一输入对照，三个输出的最大绝对误差为 7.75e-7、9.54e-7、1.55e-6。输入为 in_face:0，输出为 out_face_mask:0 / out_celeb_face:0 / out_celeb_face_mask:0。

独立 `pipeline-verification` 人工项目中，使用网页任务向导依次执行 merge.me、export.dfm_me、encode.mp4。三项全部 succeeded、exit 0。导出页 video readyState=4、无媒体错误，可直接播放；FFprobe 另验证 CLI 视频是 H.264、128×128、一帧。

该短视频和训练 loss 只证明流程通路，不能用于评价真人换脸质量、长期收敛或 DeepFaceLive 的现场兼容性。

## 自动检查

| 检查 | 结果与范围 |
| --- | --- |
| Python ME、合成/DFM、辅助工具 | 29 passed：ME 10 + 应用 2 + 辅助 17 |
| 隔离 Web 单测 | 172 passed、0 failed、1 skipped（Windows 文件符号链接 EPERM） |
| 默认浏览器 E2E | 5 passed、0 failed、1 skipped（未启用该脚本的真实训练选项；训练由上文独立实测覆盖） |
| Launcher Pester | 43/43 passed |
| Launcher Node | 7/7 passed |
| Ed25519 工具 | 2/2 passed |
| Runtime / vision / 版本 / i18n | 通过；新增未翻译键 0，195 个既有未翻译键由基线记录 |
| 传统 CLI | 8 类、52 个固定路由通过 |
| Python/Web CLI 契约 | Web 的 44 个旗标在主 CLI 和 ME bridge 中存在 |

辅助检查覆盖真实 S3FD、2D/3D FAN、FaceEnhancer 和 SFace；严格权重缺失拒绝；XSeg 训练、预览、保存、续训和 WF→FULL 遮罩应用；PAK/ZIP 元数据字节往返及人物同名文件；错脸分类器概率对照；排序、yaw、标注和状态工具；元数据恢复、遮罩导出、重设尺寸；FFmpeg 降噪、封装和提帧。XSeg 生成器的后台线程会正常关闭，初始化失败不会永久等待。

默认浏览器检查覆盖页名/非空内容/无框架错误覆盖层、向导、工作区、GPU 遥测、断线恢复、姿态对比和 XSeg 草稿保护。真实训练与应用测试的截图、临时脚本在系统临时 dfl-pt-qa 目录，不提交 Git。

## 复现

从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest _internal\DeepFaceLab\tests\test_me.py _internal\DeepFaceLab\tests\test_me_application.py _internal\DeepFaceLab\tests\test_auxiliary_tools.py -q
.\.venv\Scripts\python.exe tools\smoke_test.py
powershell -NoProfile -ExecutionPolicy Bypass -File tools\prepare-launcher-tests.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\verify-release.ps1
```

Web 的 pnpm test 使用隔离目录，运行资料不会进入测试副本。浏览器默认 E2E 见 webui/tests/browser-e2e.mjs；启用真实训练选项前须使用人工测试项目。

本机日志在忽略目录 `.runtime/verification`：backend-tests.log、application-tests.log、browser-e2e.log、merge-preflight-tests.log、dfm-full-cli.log、dfm-full.json、merge-cli.log、merge-video.json、video-cli.log、release-gates.log、release-gates-final.log。Launcher 细节见 launcher/MIGRATION_VALIDATION.md。

未制作或发布 RAR 包，未配置 Git 远端，未做原生启动器窗口交互验收。XnView/VisiPics/EbSynth 外部程序及通用 XSeg 权重的边界见 MIGRATION.md。约6GB的运行环境搬移验证副本因自动审批拒绝清理而保留在忽略的 launcher/vendor 中；打包已显式排除该目录。

## 最终发布门槛

首次 verify-release 在已经通过 Python/Web/Node 后，发现缺少项目本地 Pester。补齐实体 Pester 4.10.1（90个文件 SHA256 一致）后，prepare 与 Pester runner 均成功；随后续跑 launcher/build/HTTP smoke，release-gates-final.log 返回 exit 0 并确认全部所需门槛通过。没有将失败的首次执行当作通过。

首轮最终额外检查：生产构建、4项 Sites 静态路由测试、dist freshness 和 loopback HTTP smoke 均通过；900×1000 紧凑视口可选择已有 ME 模型及默认 learned 模式4，无页面异常。当轮源码和构建一致；本轮后端更新的发布门槛见 ME 训练验证，不将历史 Pester/build 计为本轮重跑。
