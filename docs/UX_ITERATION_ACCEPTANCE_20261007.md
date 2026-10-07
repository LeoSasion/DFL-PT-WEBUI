# UX 迭代实现与验收记录

日期：2026-10-07。状态：**实施与本轮隔离功能验收完成，真实浏览器 200% 缩放未验证**。独立发行验收随 v7 外部报告记录（该文档不自证包 SHA）。

范围来自 [已批准的 UX 开发计划](UX_DEVELOPMENT_PLAN_20261007.md)。原完成标准和验收场景保留不变；本文将实际实现、已有证据和剩余检查分别记录。操作步骤见 [用户指南](USER_GUIDE.md)。

## 1. 本轮边界

- 顶部九阶段完整保留：素材、提帧、切脸、清洗、遮罩、训练、诊断、合成、封装。左侧原页面和工具入口继续可达；“下一步”是建议，不能取代导航或自由跳转。
- 原始 aligned、已保存选集、已有模型继续是合法工作路径。Best 筛选可选，用户仍可直接采用原 aligned。
- 项目草稿按项目及侧别保存；来源或参照失效时撤回旧身份确认，恢复草稿不启动分析、发布或训练。
- SRC/DST 项目选择成对持久化；只在明确点击填入动作后改变本次参数。项目偏好不自动覆盖检查点配置，原配置变更与采样状态确认继续生效。
- 人工复核另建版本，保留旧算法建议、旧计划和旧发布回执。keep/exclude/defer、撤销、批量决定、版本冲突检查和明确发布均须有可追溯证据；普通 keep 不能绕过训练输入硬约束。
- 本轮 UX 变更未改 ME 桥接，未启动新训练。本轮不运行长训或双 GPU 验收，不把历史训练或双 GPU 未验证范围改写为通过。后续若触及桥接，继续遵守 [短程验收范围](ME_SHORT_RUN_ACCEPTANCE.md)。

## 2. 十二项实现映射

“本轮功能验收”表示下文所列真实操作、行为回归和目标界面检查已有证据，不代表最终训练画质、全片视频交付或整界面可访问性认证。原批准标准不删改，未验证项单列。

| 项目 | 当前实现 | 主要实现位置 | 最终状态 |
|---|---|---|---|
| UX-01 状态一致性 | 产物发现、当前任务、历史检查点与本次保存分别表达；已有成片/合成优先推荐；九阶段和手动入口保留；审计明确基本校验与需关注可重叠 | `workflow-readiness.js`、`App.jsx`、`Chrome.jsx`、`TrainingView.jsx`、`ToolLabView.jsx` | 本轮功能验收 |
| UX-02 两侧输入 | 项目级 SRC/DST 选择、版本与有效性读取；分别修改保留另一侧；明确填入本次配置，检查点配置不自动改 | `project-ui-state-manager.mjs`、`TrainingInputPicker.jsx`、`App.jsx`、`METrainingFields.jsx` | 本轮功能验收 |
| UX-03 草稿恢复 | 按项目/侧保存参照、确认、计划、分类、页码和数量偏好；冲突暂停；来源变化撤回旧确认 | `project-ui-state-manager.mjs`、`BestFacesetPanel.jsx` | 本轮功能验收 |
| UX-04 人工版本 | 创建子版本，keep/exclude/defer、批量、撤销、剩余分类数量；版本 CAS、重试、发布与撤回保护 | `core/best_training_faceset.py`、`best_training_faceset.py`、`best-faceset-manager.mjs`、`BestFacesetPanel.jsx`、`BestFacesetPreview.jsx` | 本轮功能验收 |
| UX-05 后台到结果 | 终端上方独立结果栏显示最新结果；最近三项主动展开并支持 Esc 还焦；查看/继续定位正确项目、侧别与选集，沿用切换保护 | `BackgroundOperations.jsx`、`App.jsx`、`operation-manager.mjs` | 本轮功能验收 |
| UX-06 筛选首屏 | 参照、数量上限和分析生成优先；500 分批、60 分页分别说明；历史可辨识，高级评分折叠 | `BestFacesetPanel.jsx`、`best-faceset-workflow.js`、`QualityPipelinePanel.jsx` | 本轮功能验收 |
| UX-07 项目入口 | 标题处打开项目菜单，可查看、创建和切换；目录预览与运行保护保留 | `Chrome.jsx`、`ProjectPickerDialog.jsx`、`ProjectManagerPanel.jsx`、`App.jsx` | 本轮功能验收 |
| UX-08 向导与续训 | 具体动作直接进入参数，全局新任务保留选择；续训显示继承摘要、输入和运行计划，全参数可展开 | `Overlays.jsx`、`METrainingFields.jsx`、`task-configuration.js` | 本轮功能验收 |
| UX-09 遮罩与小样 | 遮罩默认简洁栏保留手工修正与配置动作，展开用途后显示三类说明；中文合成说明；1–20 帧独立小样、回执、三联图和参数继承 | `OperationsView.jsx`、`MergePreviewPanel.jsx`、`merge-preview-manager.mjs`、`command-registry.mjs`、`main.py`、`mainscripts/Merger.py` | 本轮功能验收 |
| UX-10 诊断与交付 | 诊断可选；没有评测进程时提供有效返回路径；导出按合成帧/遮罩状态引导，已有成片优先查看 | `QualityDiagnosticsView.jsx`、`ExportView.jsx`、`workflow-readiness.js` | 页面/行为验收；未做新成片导出 |
| UX-11 图像与键盘 | 遮罩更多菜单、280–680 px 面板宽度及记忆，720 短屏编辑区最低 240 px 并可滚动；连续复核、方向键和决定快捷键；控件含文件名 | `OperationsView.jsx`、`BestFacesetPreview.jsx`、`best-faceset.css`、`styles.css` | 本轮功能验收；真实 200% 缩放未验证 |
| UX-12 文档与发行 | README、两份操作指南、计划和本记录同步；FINAL2 构建及来源校验完成，发行归档由独立外部报告证明 | `README.md`、`docs/USER_GUIDE.md`、`docs/BEST_TRAINING_FACESET_WEBUI.md`、本计划、本记录 | 文档/回归/构建同步；发行外部记录 |

源码相对位置：Python 后端在 `_internal/DeepFaceLab` 或 `webui/python`，Node 服务在 `webui/server`，组件在 `webui/src/components`，领域模块在 `webui/src/domain`。映射仅列主要文件，不能作为全部变更清单。

## 3. 已取得的后端与行为证据

| 检查 | 已观察结果 | 证据及范围 |
|---|---|---|
| Best 选择、发布和人工复核 Python | 45 passed，0 failed；11.89 秒 | `.validation/UX_REVIEW_BACKEND_REGRESSION_20261007.json`；相关三个 pytest 文件 |
| 项目状态与 Best Node 定向回归 | 状态测试 14 passed / 1 skipped；HTTP 6 passed / 0 skipped | 同上；最小副本无 Python 的单项跳过由独立真实 helper 检查覆盖，不宣称完整套件通过 |
| 小样与合成质量守卫 Python | 28 passed | `tools/tests/test_merge_preview.py`、`tools/tests/test_merge_quality_upgrade.py`；本轮定向检查，非训练验收 |
| 工作流状态与小样 Node 定向回归 | 10 passed / 1 skipped | `webui/tests/workflow-readiness.test.mjs`、`webui/tests/merge-preview.test.mjs`；只读路径下隔离媒体夹具按设计跳过 |
| Windows 8.3 别名隔离小样 | 2 passed / 0 skipped | `.validation/ux-workflow-views-20261007/actual-preview/alias-path-regression.json`；测试副本位于短路径 Temp |
| 前端域、客户端、翻译定向检查 | 最后修正后 16 passed / 0 failed / 0 skipped；约 268.495 ms | `C:/Users/Administrator/AppData/Local/Temp/dfl-ux-best-finish-tests.out`；包含数量上限数字等价编辑的草稿保存检查，原初次 15 项日志 `dfl-ux-best-frontend-tests.out` 保留 |
| 独立组件渲染检查 | 8 组通过 | `C:/Users/Administrator/.codex/visualizations/2026/10/07/dfl-ux-best-components/qa.json`；使用模拟接口，仅证明组件行为，不能替代真实后端 GUI 闭环 |

上述结果互为不同层级的证据，不以简单相加替代完整回归。最后完整套件记录为 V5。

完整回归历史检查点保留：Node V3 为 **411 total / 409 passed / 0 failed / 2 skipped，108.212 秒，exit 0**；V4 为同样数量、**111.235 秒、exit 0**。最后 Node V5 为 **411 total / 409 passed / 0 failed / 2 skipped，115.470237 秒、exit 0**，完成于 `2026-10-07T07:41:36Z`。证据分别为 `.validation/UX_ALL_UNIT_V3_20261007.*`、`V4` 和 `V5` 日志及 exit JSON。

V5 两项跳过是 `linked result files cannot be read or overwritten through recovery`（Windows 文件 symlink 不可用）和 `quarantine resolver rejects symbolic-link files when the platform permits them`（创建 symlink 被 Windows EPERM 拒绝）。这两项不是缺少 Python；小样回执夹具实际通过。

Python 最后必要回归已实际记录 **73 passed，14.31 秒，exit 0**，覆盖三个 Best 文件以及 `test_merge_preview.py`、`test_merge_quality_upgrade.py`；见 `.validation/UX_PYTHON_FINAL_20261007.out.log`、`.err.log` 和 `.exit.json`。

FINAL2 构建于 `2026-10-07T07:41:08Z` 完成，exit 0，见 `.validation/UX_BUILD_FINAL2_20261007.*`；dist-provenance check passed，摘要为 `23ec8c84380611528556dd9458c8a80e1ffdb8fbbd55e85c36f597b48f2b17ba`。V5 已包含英文结果标签修补；其后加载按钮的纯文案修改由 FINAL2 构建和实际浏览器验证，不把这项文案验证称为又一次完整套件。

### 3.1 真实人工复核、发布与撤回

证据：`.validation/UX_REVIEW_BACKEND_ACCEPTANCE_20261007.json`，原始步骤在 `.validation/ux-review-real-ffd9274dd3/`。

对 14 张隔离素材执行真实 YOLO26/TUFA/SFace 路径，约 18.797 秒。21 个 CLI 步骤覆盖初始化、分析、全局选择、父版本发布、新建人工版本及重试、批量排除、暂缓、撤销、保留、旧版本冲突拒绝、硬约束 keep 拒绝、来源变化拒绝、发布预演、正式副本发布、已发布版本不可修改、继续建立新版本、撤回预演和撤回。

原建议 selected 12 / review 0 / rejected 2；人工决定后 selected 11 / review 0 / rejected 3。原隔离输入、原版本计划及回执字节未变，发布和撤回副本字节精确，未启动训练。该小批关联样本仅为功能证据；开发指定参照不构成人工身份真值，未提供可靠原 PTS，也不声明身份准确率、语义表情准确率或时间去重收益。

真实 CLI 检查早于最后的并发读取与重复创建守卫修正；修正已有行为回归和 V5 完整套件，后续真实 GUI 闭环见第 3.3 节。

### 3.2 真实独立合成小样与正式参数预检

证据目录：`.validation/ux-workflow-views-20261007/actual-preview/`。

- `report-v3.json`：真实 CPU 推理使用既有 64 px 开发检查点和真实 DST/人脸隔离副本；输出 `00040.png`、`00041.png`，整个正向及两个拒绝步骤共约 10.609 秒。实际环境和参数与注册命令相同。
- 同一报告验证正式 merged 输出位置拒绝、小样超过 20 帧拒绝；越界范围在创建输出目录前被拒绝。原素材、隔离输入和正式输出哨兵哈希不变。
- `node-lifecycle-report.json`：11 项检查全部通过，包括原生回执和全部输入/结果哈希、任务完成检查、六个三联图资产返回精确字节、按相同参数进入正式合成预检、参数变更拒绝、明确解除关联后的普通合成预检、非成员与路径穿越拒绝，以及模型/帧/人脸副本或帧清单变化拒绝。
- 本轮小样未触发新训练，也没有执行正式全片合成；11 项过程中的改动只发生在隔离副本并恢复。局部结果不是全片视觉质量或身份质量证明。

小样范围为显式 1–20 帧，输出至当前项目 `.webui/merge-previews/<job-id>/`，不替换正式序列。回执绑定模型、帧清单、源帧、人脸输入、合成参数、结果与遮罩。继承动作只打开正式配置；启动前重新验证关联回执、输入和参数，仍须通过正式 preflight。

旧 v2 检查脚本读取 UTF-8 回执时曾遇到默认 GBK 解码失败，原记录保留；v3 修正脚本读取并得到以上证据。不能把旧脚本失败记录改写为通过。

### 3.3 真实 GUI、输入完整性与来源失效

主代理使用隔离 QA 项目实际操作，原 fixture 不写入。六项最终界面检查及流程汇总见 `.validation/UX_ITERATION_GUI_ACCEPTANCE_20261007.json`；输入保护证据为 `.validation/UX_GUI_INPUT_INTEGRITY_20261007.json`、`UX_GUI_SOURCE_CHANGED_20261007.json`、`UX_GUI_SOURCE_RESTORED_20261007.json`。逐步截图位于 `C:/Users/Administrator/.codex/visualizations/2026/10/07/dfl-ux-iteration/`，最终 `12-final-paired-inputs-1280.jpg` 已实际查看，SRC6/DST1 与 64 px 继承摘要可见。

- 完整九阶段逐项等待对应内容可见后检查，通过；原 aligned 路径仍可用。
- SRC 14 张得到母计划 selected/review/rejected **6/6/2**；人工子版本经 keep/exclude/defer、批量和 undo 成为 **6/5/3**，完成新版本 dry-run 与 publish。DST 3 张得到 **1/2/0** 并发布。这是另一次 GUI 执行，不改写第 3.1 节 CLI 的 12→11 结果。
- 两侧引用 revision 2；切页、切 SRC/DST、reload、服务重启、切到新项目后返回均恢复。填入明确选择后仍需确认，原 64 px 检查点参数没有自动改变。
- 只对 QA 副本 `00004_0.jpg` 临时附加一字节：草稿 API 的 source/identity false，SRC 引用 valid false，DST 仍 true。实际界面禁用“将两侧选择填入”，模型原 SRC 参数仍为 `data_src/aligned`。立即恢复精确 SHA 后两侧 valid true，原 fixture 从未写入。
- 完整性检查在上述临时变化前确认 **24/24** 原 fixture 与对应 QA 副本的 SHA、字节长度一致，另有 **2/2** 顶层正式输出哨兵不变。新生成的四个 Best 结果文件另列，不误报为原输入变化；脚本首次 LF/全目录假设修正保存在报告说明中。
- 小样任务 **20261007072345-bce54c5f** 使用 CPU、选择第 **2–3** 帧，成功约 **7 秒、exit 0**；三联图 **2/2** 和下一张通过。“按相同参数配置全片合成”的正式预检通过后取消，未执行全片合成。
- 终端上方最新结果可定位正确 DST 选集；最近三项按需展开、Esc 还焦。具体动作直达参数、全局任务选择与续训摘要保持可用，向导跨刷新未被下一步推荐改写。
- 1280×720、1440×1000、1100×800 下 document 无横向溢出。640×360 是 CSS 视口的 **200% 内容容量模拟**，不是实际浏览器缩放测试。

本轮没有启动新训练、执行真实全片合成或生成新成片视频。诊断与导出完成页面和状态/前置条件检查，不将这些结果升级为成片交付质量验收。

### 3.4 三处真实文本对比度

证据：`.validation/UX_GUI_TEXT_CONTRAST_20261007.json`。实测 DOM 的有效文本/背景为不透明 solid 色，无参与计算的渐变或 opacity 合成。

| 样本 | 字号 | 前景 / 有效背景 | 对比度 |
|---|---|---|---|
| 检查点配置摘要标题 | 14 px | rgb(234,244,238) / rgb(11,23,18) | 16.30:1 |
| 两侧输入说明 | 13 px | rgb(160,183,169) / rgb(8,19,15) | 8.86:1 |
| 模型分辨率标签 | 13 px | rgb(192,204,197) / rgb(11,23,18) | 11.07:1 |

这里只证明三个所测文本样本，不宣称整界面 WCAG 合规或完成全面可访问性审计。

### 3.5 独立审查及最后布局复检

`.validation/UX_ITERATION_INDEPENDENT_REVIEW_20261007.md` 和 `.json` 记录 R1–R10 均修复、当前 open source/gui blockers 0。独立局部 Node **50/50**、人工复核 Python **19/19**；这些是另行执行的定向检查，不冒充主代理完整套件。

真实独立 GUI 验证项目菜单焦点圈定/Esc 还焦、小样模型下拉、320/600 px 查看宽度、五列遮罩工具栏、更多菜单与结果栏交互、短屏预览。XSeg 三点未保存编辑的原生 Cancel 保持 XSeg，Accept 返回总览；无写请求，原 JPG 前后 SHA 相同。最后 V10 终端空态可滚动到 CTA，实际打开与关闭任务向导通过。

最后整段独立重跑在主代理临时源变化期间读到 SRC count 0，缩略图等待超时；原记录保留，没有改写为通过。来源随后恢复。此前目标检查和最后终端单项复检已有证据；实际 200% 浏览器缩放始终未执行。

## 4. 本轮验收结果与独立发行范围

| 检查 | 结果 | 证据及限制 |
|---|---|---|
| 完整 Node 隔离套件 | 通过，含两项平台跳过 | V5：411 total / 409 passed / 0 failed / 2 skipped，115.470237 秒，exit 0；Windows symlink 不可用/EPERM，名称见第 3 节 |
| Python 必要回归 | 通过 | 73 passed，14.31 秒，exit 0；`.validation/UX_PYTHON_FINAL_20261007.*`；未启动训练 |
| 最后前端构建与来源验证 | 通过 | FINAL2 exit 0；dist-provenance check passed，摘要见第 3 节；最后加载纯文案另经实际浏览器确认 |
| 项目菜单、草稿与成对输入恢复 | 本轮功能验收通过 | 真实切页/侧、刷新、服务重启和新项目返回；检查点配置不自动改 |
| 来源失效拒绝与字节恢复 | 通过 | 临时 QA 变化时 source/identity/SRC invalid，DST valid；实际填入禁用；精确恢复后有效，原 fixture 只读 |
| 人工复核、独立发布与恢复保护 | 通过 | GUI SRC 6/6/2→6/5/3、DST 1/2/0；真实 CLI 覆盖撤回、硬约束及旧版本字节不变 |
| 后台结果、向导与续训 | 本轮功能验收通过 | 正确侧/版本定位，最新与最近结果、Esc 还焦，具体/全局入口、原配置保护 |
| 遮罩更多、宽度和未保存保护 | 通过 | 独立实际短屏菜单、320/600 px、Cancel/Accept；JPG SHA 不变 |
| 两帧小样与正式参数预检 | 通过 | 2–3 帧、CPU、约 7 秒、2/2 三联图；正式预检通过后取消，未执行全片 |
| 诊断与导出 | 页面与行为分支通过 | 诊断可选、空态/前置条件/产物状态检查；未执行新成片真任务 |
| 完整九阶段与原 aligned | 通过 | 每阶段内容等待可见；自由进入与原输入保留，不强制 Best |
| 三个桌面视口 | 通过 | 1280×720、1440×1000、1100×800 无 document 横向溢出 |
| 实际浏览器 200% 缩放 | 未验证 | 640×360 CSS 容量模拟不能替代实际缩放 |
| 键盘、焦点和文本对比度 | 已测范围通过 | 连续复核、菜单/结果栏/CTA 焦点；仅三处文本比值，不声明整界面合规 |
| 独立审查 | 无未修复阻塞项 | R1–R10 修复；目标 GUI 与最终终端单项复检，独立整段重跑超时保留说明 |
| README 与四份说明 | 已同步 | 当前入口、状态、恢复、输入、小样和验收范围一致 |
| v7 源码/portable 归档及新恢复 | 独立外部报告记录 | 独立发行验收随 v7 外部报告记录（该文档不自证包 SHA），不在包内预先声明新包通过 |

首轮完整 Node 运行保留在 `.validation/UX_ALL_UNIT_20261007.out.log`、`.err.log` 和 `.exit.json`：400 项中 395 passed、3 failed、2 skipped。失败涉及新增小样的模型类别、运行时命令允许列表及 Windows 短路径比较；相应修复经后续定向和 V3/V4/V5 完整检查验证，旧失败日志不改写。

v7 只读发行准备位于 `.validation/UX_V7_DISTRIBUTION_PREPARATION_20261007.json`，准备不是新包验收。独立发行报告记录最终冻结源、归档/分卷 SHA、完整恢复、默认 setup 和 smoke；本文不预先给出尚未确认的新包结果。已有 v6 归档、SHA 和验收记录继续代表原快照，不宣称包含本轮 UX。

## 5. 尚未证明的范围

本轮不证明最终换脸视觉质量、身份保真、正式长训练收益或真实双 GPU。组件模拟检查不能替代隔离项目的真实后端与界面恢复闭环，文件发现与数量相等也不能替代任务预检中的内容校验。

本记录中的本地 `.validation` 和 Temp/visualizations 文件是忽略的开发证据。新发行快照的独立证据由外部报告记录；本轮通过、平台跳过、真实缩放限制及历史失败分别保留，不以包内文档自行证明发行归档。
