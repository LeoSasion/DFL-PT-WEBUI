# DFL-PT-WEBUI 传统工具菜单

根 `传统命令菜单.bat` 打开固定路由菜单，8 个分类保留视频/SRC/DST/XSeg/ME训练/合成导出/封装/辅助工具。注册表是 commands.json，添加任意 BAT 不会成为执行入口。

只有 PyTorch ME 主模型训练；XSeg 保留为辅助遮罩模型。ME 训练、merge、DFM 使用新后端 main.py 的兼容命令入口；Ctrl+C 保存并停止。所有工具使用本项目 setenv.bat 与 .venv，不调用旧项目、旧 TensorFlow 解释器或 RG 切换。旧模型训练、格式转换及 Explorer 隐藏菜单已移除。

直接使用某些配套工具时，可通过工具 BAT 提供相应输入参数。每次调用 setenv 都读取 WebUI 同一 activeId：default 位于 workspace，其余位于 workspaces/<id>；菜单与命令显示绝对工作区。失效选择会停止，不回退到另一个项目。WebUI 工具面板提供更完整的参数与可恢复操作。

aligned 预览与清洗默认使用 WebUI 的 SRC/DST 人脸浏览器，无需安装 XnViewMP。日常相似图审查默认使用“工具实验室 → 数据审计 → 相似组清洗”，无需另装 VisiPics；每次仅分析当前 aligned 的前 500 张，需人工复核并保留代表图，暂不支持切换批次或跨批查重。

EbSynth 的关键帧结果传播尚未实现，列为后续 WebUI 功能候选；现有场景检测和分段提帧不包含这项能力。传统菜单保留这三项已有外部程序的可选适配入口，缺失程序不会被报告为已安装，也不属于 WebUI 安装要求。需要使用已有程序时，可通过 external-tools.ps1 的 Tool、ExecutablePath、ConfigureOnly 参数保存实际程序位置，配置保存在项目 .launcher-install/external-tools.json。
