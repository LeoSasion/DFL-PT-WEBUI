# 历史人脸集的数据读取与恢复升级

日期：2026-10-07。对应 `LEGACY_ITERATION_AUDIT_20261007.md` 的排序、pickle 和元数据恢复项。用户素材目录未执行修改；功能验收全部在临时人脸集或只读生产样本副本中完成。

## 数据读取

`core.safe_pickle` 是 pickle 数据 opcode 解释器，不使用 `pickle.Unpickler`。输入里的 GLOBAL 只会解析为固定数据符号；不 import、不调用输入的函数，也不构造输入指定的 Python 实例。

- 兼容 NumPy 0–5 协议的 numeric/bool 数组、标量、dtype、大小端和 Fortran 顺序。先验证维数、整数维度、元素数、dtype、原始字节长度和累计预算，再用受信代码构造数组；拒绝 object/structured/subarray dtype。
- APP15 限制 65,533 字节；faceset/session 限制 128 MiB；历史权重限制 1 GiB。所有路径另有限制树深度、节点、memo、opcode、单数组和累计数组规模。单个 ndarray 不能通过小 pickle 声明巨大 shape；整数长度也在转换为 bigint 之前检查。
- DFL 字典、坐标、仿射、分割多边形、压缩遮罩画布、人脸集配置和 `meta.dat` 备份有 schema 验证。PNG/JPEG 遮罩先检查头部尺寸，再交给图像解码器。
- DFLJPG、PAK/ZIP config、Util、Saveable、XSeg、ModelBase/options、XSegUtil 和 XSegEditor 共用 reader。旧 ME 导入的 whole-dict 与 count + single-tensor records 也共用该 reader，每个文件共享预算；原有保存格式保持兼容。
- session profile 额外只解析三个历史合成类型为 `DataRecord(type_name, state)`，以及 pathlib 的词法路径字符串。它不执行这些类的构造器；合成 adapter 明确验证并转为纯数据 schema 2。metadata、faceset、weights profile 拒绝这些对象；未知类仍拒绝。内部 IPC pickle 不作为磁盘素材加载入口。
- 可选默认配置或编辑器配置被拒绝时，先独立保留 `.invalid-<sha>` 原件，再允许新配置保存。重要模型或素材加载失败明确报错。

## 排序和原始文件名恢复

`Sorter.main(..., dry_run=False, target_count=None)` 保留旧调用兼容；`final_process(..., dry_run=True)` 可只读预览。排序保留项改名、未选中项归档、hash 绑定的 landmarks sidecar 一起进入事务。

每次排序独立保存至 `aligned_trash/faceset-<id>/`：`receipt.json`、`originals/`、`staging/` 和 `discarded/`。已有 `aligned_trash` 及所有以前批次都保留。其他恢复操作保存到 `aligned_history/faceset-<id>/`。

原件和全部输出先完整复制并验证 SHA，随后才提交。目标碰撞、重复来源、软链接/路径越界、源图改变和写入失败会拒绝提交；已开始的失败或取消会恢复整批原件。提交采用不覆盖已有目标的原子发布，原件备份是独立字节副本。

`core.faceset_transaction.recover_transaction(receipt_path, dry_run=False)` 按回执恢复；已恢复调用幂等。它拒绝后来修改的输出、损坏备份和异常路径。中断批次有 PID + 创建时间身份，仍在运行的其他事务不能恢复；已退出、只完成部分 staging 的准备阶段也可以安全清理锁。无法完成回滚时保留锁、所有备份和 `rollback_failed` 回执。

目标数量现在是正整数且精确受可读样本总数限制，1、2、32、64、65、129、130 等小数额不会变成零配额。选择按占用的 yaw/pitch 桶分配，记录实际 CPBD 或源框面积、姿态和选择结果；这些指标没有被包装成模型准确率。“最佳人脸”的显示名改为“姿态覆盖筛选”，旧 final/final-fast 参数仍兼容。缺失源框记为未知/0，不当作读取损坏。该修复只证明数量、覆盖和数据保护，不宣称新的视觉质量赢家。

## 元数据恢复

保存 `meta.dat` 时附来源 SHA 记录；第二次保存先归档旧备份。恢复先验证整批，包括丢失文件、来源记录、目标 canvas、68/98 点、多边形和遮罩；全部输出准备好后才提交。编辑后需要恢复尺寸的图片采用原尺寸 JPEG100 staging，当前编辑稿完整备份。`meta.dat` 不删除；旧 landmarks sidecar 如像素 SHA 已变会独立归档，嵌入的已保存 68/98/手绘/遮罩保持。

## 已验证证据

- 新安全/事务用例 29 项通过，包括禁止 global/opcode、巨大 shape/bigint、深度/循环/引用放大、正常 NumPy 各协议、磁盘失败、输出失败、提交后取消、旧 trash 保留、新编辑冲突、live owner 拒绝、以及真正的子进程 `os._exit(77)` 后完整恢复。
- 旧 ME 导入测试 56 项、合成质量及旧 session adapter 测试通过。生产 JPEG 的三图副本还实际运行了旧 FinalLoader 子进程→姿态覆盖选择→dry-run 回执，选中 2 张、分项可见、全部源图 SHA 保持。完整项目验收由总实施记录另行汇总。
- 生产 TUFA JPEG 的临时副本实际排序→恢复→元数据保存→恢复成功，68/98 元数据保留，源文件 SHA 未变。
- `.validation/legacy-safety-20261007/read-only-evidence.json`：真实 ME checkpoint 15,814 iteration 由 `torch.load(weights_only=True)` 只读回读；56 tensor 的 NumPy snapshot 在数据 reader 中精确往返。这是读取兼容证据，不是新增训练、预测或质量验收。
- 发行 generic XSeg 的固定 SHA `26e45677ef3136e0327f0fd51e452cbea81a703ee7fea9121b01ba58da65c385` 未变。其 222 数组由新 reader 加载，实际 CPU 256×256 推理输出有限；当前安全测试同时验证发行 2DFAN 的 945 数组。此前 FaceEnhancer 的安全回读只保留为历史事实，该旧模型已按后续用户要求移出生产。

## Worker 与 RPC 生命周期

`Subprocessor` 保留旧构造，新增 `initialize_timeout_sec=120`、`finalize_timeout_sec=30`、`shutdown_timeout_sec=2`。初始化失败、静默退出、运行失败和超时有阶段/PID/原因记录；发出的数据至多退回一次。最后一个 worker 丢失或 finalize 失败不会返回假成功。主回调异常也会清理所有 worker；所有 stop/join 有界，无法确认退出明确失败。单 worker 队列的父进程重复端点关闭，避免子进程半条响应后退出而接收侧无法看到 EOF。

`MPFunc`、`MPClassFuncOnDemand` 新增 `rpc_timeout_sec=300` 和 `close()`。所有调用方共享一个 owner 的 stdlib Manager 服务，每个请求有独立 Pipe、id、deadline；owner 在已注册 UI 回调线程顺序执行。调用方没有跨进程等待锁，崩溃不会永久占用后续调用；超时、迟到结果、owner 退出/PID 复用、callback 异常都有明确结果。Manager 启动/关闭有界且自带 owner 创建时间 watchdog，owner `os._exit` 后服务自动退出。owner-only `close` 注销回调并引用计数关闭服务；调用方 close 只关闭本地使用。直接发生在 owner 的函数调用仍是同步 Python 调用，不声称能抢占正在执行的 GPU/Python 函数。

辅助 `SubprocessGenerator` 也增加并行启动失败/deadline、第一次输出/后续输出 deadline、有限 stop/kill/join 和闭合队列；closed 后不重启。该模块不是当前 ME bridge 的数据加载器。

22 项真实 spawn 生命周期测试已通过，覆盖 crash/hang/exit、串行启动、finalize、host callback异常、5 个并行 RPC caller、懒加载、迟到回复、已退出 caller 后新调用、真正 owner `os._exit(41)` 后 caller 和 Manager 退出，以及有限 generator/生成异常/初始化失败。实际提取和合成输出对照由总体实施验收另行记录。

仍保留的边界：多文件提交依靠 durable 回执恢复，而不是声称整个目录一次性原子替换；用户直接手工修改中的文件不会被强行覆盖。事务备份占用额外磁盘空间，自动清理旧备份不在本次范围。未新增 ME 验收训练，未发布发行包。
