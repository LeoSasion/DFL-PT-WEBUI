# DFL-PT-WEBUI 传统工具菜单

根 `传统命令菜单.bat` 打开固定路由菜单，8 个分类保留视频/SRC/DST/XSeg/ME训练/合成导出/封装/辅助工具。注册表是 commands.json，添加任意 BAT 不会成为执行入口。

只有 PyTorch ME 主模型训练；XSeg 保留为辅助遮罩模型。ME 训练、merge、DFM 使用新后端 main.py 的兼容命令入口；Ctrl+C 保存并停止。所有工具使用本项目 setenv.bat 与 .venv，不调用旧项目、旧 TensorFlow 解释器或 RG 切换。旧模型训练、格式转换及 Explorer 隐藏菜单已移除。

直接使用某些配套工具时，可通过工具 BAT 提供相应输入参数；数据默认位于 workspace。WebUI 工具面板提供更完整的参数与可恢复操作。

XnViewMP、EbSynth 和 VisiPics 是可配置的外部程序。aligned 浏览已有 WebUI 替代；这三项的缺失程序不会被报告为已安装。使用 external-tools.ps1 的 Tool、ExecutablePath、ConfigureOnly 参数保存实际程序位置，配置保存在项目 .launcher-install/external-tools.json。原工作区的 VisiPics 安装包保留在 _internal/installers/VisiPics-setup.exe，没有自动安装。
