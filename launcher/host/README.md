# DFL-PT-WEBUI 原生启动器

Windows WPF/WebView2 启动器使用独立 `DflPtWebUi.Launcher` 命名空间，用户配置保存在 `%LocalAppData%/DFL-PT-WEBUI/Launcher`。项目身份要求 Git 目录、webui 和 `_internal/DeepFaceLab/me.py` 同时存在，旧项目不会被视为本产品。

`powershell.exe -File launcher/build-host.ps1` 构建 `launcher/bin/DFL-PT-WEBUI.Launcher.exe`；构建下载的唯一原生 SDK 是 Microsoft 官方 WebView2 NuGet 包。先在 launcher/ui 使用仓库 Node 执行 npm ci。运行环境检测以 launcher/runtime-manifest.json 为准。

启动器内嵌本地 bootstrap、setup-runtime、运行时清单、UI 和固定终端桥。不会启动在线仓库安装或更新，也不会读取旧产品设置。支持启动/停止 WebUI、打开传统工具路由、检查和修复本地依赖。WebUI 在收到明确启动操作后才运行。

配套工具使用新 PyTorch 后端。运行时及媒体/模型目录不提交 Git；项目创建时提供了独立的 Python、Node、FFmpeg。本地 setup 可由仓库 Python 重建 venv，完整依赖在根 requirements.txt。
