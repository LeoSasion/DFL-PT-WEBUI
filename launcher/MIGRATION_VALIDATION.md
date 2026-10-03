# DFL-PT-WEBUI 启动与运行时迁移验证

2026-10-01，在 E:/DFL-PT-WEBUI 验证。

- 本地 bootstrap -NoNetwork 通过：Node 24.19.0、Python 3.12.14 / torch 2.9.1+cu128、FFmpeg。Python 检查要求 sys.base_prefix 指向本项目 _internal/python_base，torch 必须来自本项目 .venv。
- Windows PowerShell Pester 启动器测试：43 passed，0 failed。新的 manifest 必须包含 node/python/ffmpeg，拒绝旧 schema；项目定位要求 PyTorch ME 入口；所有远端均拒绝。
- Node 启动器终端和管理器身份测试：7 passed。管理器拒绝旧产品、不同仓库、缺少身份的 health，以及 PID 不匹配、过期或属于其他目录的状态记录。
- 根 WebUI BAT 的 status 在现有新项目服务上成功执行；根传统 BAT 的 --check 通过：8 类、52 个固定工具路由。测试期间没有停止已运行的 WebUI。
- build-host.ps1 -NoDownload 完整构建成功，产物 launcher/bin/DFL-PT-WEBUI.Launcher.exe。WebView2 SDK 校验 SHA-256 后使用本地副本；C# 无编译警告。仅确认构建成功，没有声称已对原生窗口作交互验收。

旧 CPython 3.7 / TensorFlow wheelhouse、CUDA 11 / cuDNN 8 下载链以及它们的测试已退休。新启动器使用独立命名空间、配置目录和产物名称，没有发布源，在线项目安装/更新和启动器自更新均关闭。

Windows venv 的绝对 base 路径由 setup-runtime.ps1 在本地检查时重写。WebUI 管理器启动前执行此检查，保证复制或移动运行包后使用包内的 Python base。发布包必须同时携带 _internal/python_base 和 .venv，不能因为它们被 Git 忽略而省略。

重定位另做了真实独立副本验证：复制约 56 MB Python base、5.68 GB venv、Node 和 FFmpeg 到独立测试目录，没有 junction/共享 site-packages；以副本自身的 bootstrap -NoNetwork 执行本地检查后，sys.base_prefix、sys.prefix 以及 torch/numpy/cv2/scipy/PySide6/onnxruntime 的实际模块路径全部指向副本。随后在 NVIDIA RTX PRO 6000 Blackwell 上执行 CUDA 张量计算，arange(16).sum() 返回 120。路径和版本证据保留在 relocation-runtime-evidence.json，测试副本暂保留在被 Git 忽略的 launcher/vendor/relocation-standalone-20261001。此项验证的是独立运行时重定位，没有声称已经生成并验收完整发布压缩包。

XnViewMP、EbSynth、VisiPics 保留为可配置的外部工具；没有找到前两项本体，也没有将 VisiPics 安装包当成已经安装的查重程序。VisiPics 原始安装包保留在 _internal/installers/VisiPics-setup.exe。缺失工具会明确退出，路径配置保存在项目 .launcher-install/external-tools.json。aligned 浏览可使用 WebUI。
