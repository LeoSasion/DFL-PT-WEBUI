# DFL-PT-WEBUI 原生启动器

Windows WPF/WebView2 启动器使用独立 `DflPtWebUi.Launcher` 命名空间，用户配置保存在 `%LocalAppData%/DFL-PT-WEBUI/Launcher`。项目身份由 PT 发行标识、WebUI 与 ME/PyTorch 入口共同确认；源码 ZIP 和便携包不需要 Git 元数据。服务状态同时核对 PT 服务标识和实际项目路径。

`powershell.exe -File launcher/build-host.ps1` 完整构建 UI 和 `launcher/bin/DFL-PT-WEBUI.Launcher.exe`，并在项目根目录保存 `DFL-PT-WEBUI.exe`；构建下载的唯一原生 SDK 是 Microsoft 官方 WebView2 NuGet 包。先在 launcher/ui 使用仓库 Node 执行 npm ci。运行环境检测以 launcher/runtime-manifest.json 为准。独立 EXE 的发布验证见 [一键启动器说明](../../docs/LAUNCHER.md)。

启动器内嵌官方源码获取、完整运行时安装、本地 bootstrap、setup-runtime、运行时清单、UI 和固定终端桥。首次安装获取 PT 官方源码 ZIP；Git 更新和 EXE 自动更新继续关闭。支持启动/停止 WebUI、打开传统工具路由、检查和修复本地依赖。WebUI 在收到明确启动操作后才运行。

配套工具使用新 PyTorch 后端。运行时及媒体/模型目录不提交 Git；项目创建时提供了独立的 Python、Node、FFmpeg。本地 setup 可由仓库 Python 重建 venv，完整依赖在根 requirements.txt。
