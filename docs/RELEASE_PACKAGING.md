# 发布打包与文件校验

`tools/build-release.py` 只使用 Python 标准库。它负责生成可复查的文件清单、源码 ZIP、Windows 便携 ZIP 和 SHA-256；不会安装依赖、运行应用、训练、执行功能测试或双卡测试。旧的 `tools/package_release.ps1` 已改为该工具的入口，默认只预览；旧 RAR、外部 node_modules 和包含 Git 历史的参数会被拒绝。

## 先预览

在仓库根目录运行：

```powershell
.\_internal\python_base\python.exe tools\build-release.py preview --kind both --write-plan
```

输出每种包的文件数、原始总字节数、各材料类别统计、JavaScript 依赖和需要处理的说明，并将路径与大小清单写入 ignored 的 `release-output/source.plan.json` 和 `portable.plan.json`。同名预览清单默认不覆盖；重新盘点时使用新的输出子目录，或明确加上 `--force-plan`（PowerShell 入口为 `-WritePlan -ForcePlan`）。预览不会创建 ZIP，也不会执行应用。正式构建从指定提交的 Git blob 导出源码，预览不会将未提交的新源码视作已发布文件；请提交最终改动后重新预览。

源码包包含指定提交中的可分发 tracked 文件，包括测试、验收工具和复现文档；不带 `.git` 或任何历史记录。发现具体私人资产时，在发布前删除/取消跟踪，或在 `release/package-config.json` 的 `sourceExclude` 中逐项列出。私有 `workspace` / `workspaces`、QA 运行结果、日志、安装缓存和本机状态不会因存在于磁盘而进入包。

便携包在源码快照基础上按配置排除测试目录、设计草图和私有 QA 证据，保留小体积说明文档和现有说明引用的检查脚本，再纳入已有的 WebUI 构建产物、launcher 可执行文件、项目独立 Python base、venv 的生产包、Node、FFmpeg 发行材料、辅助权重和许可。它不纳入整个工作目录、launcher/vendor、安装器缓存、外部工具、旧项目运行环境、Playwright、pnpm store、pip 下载缓存、Python 字节码或 pytest 临时依赖。第三方包中供运行时导入的 `testing`/`cache` 等模块仍保留，避免误删生产模块。

JavaScript 依赖从 `webui/package.json` 的 `dependencies` 和已安装的 optional/peer 依赖解析，文件链接被转换成普通文件，所有真实来源必须位于当前仓库的 `webui/node_modules`。不会使用全局包目录。当前启动器和本地管理器需要 Vite/esbuild，且源码时间发生变化时可重建 UI，因此便携包保留这些生产依赖；不从 `devDependencies` 收集 Playwright。`node-pty` 的 Windows x64 native binding 和辅助 DLL/EXE 必须随包保留。

`.venv/pyvenv.cfg` 在包中采用无本机绝对路径的版本，启动时现有 `launcher/setup-runtime.ps1` 会写入当前位置。venv 只纳入 `Scripts/python.exe`、`pythonw.exe` 及生产 site-packages；不会打包带本机路径的激活脚本或无关开发命令入口。Python 包的 `.dist-info`、licences、CUDA DLL、Qt 运行库及 pip 模块保留，便于识别许可和维护。Python base 的 `ensurepip` 随附 wheel 属于创建/修复 venv 的必要材料，保留在包中；site-packages 中残留的 wheel 文件及下载缓存不纳入。

## 生成发布材料

正式构建要求 Git 工作区干净，包含未跟踪文件的检查；本机构建产物须来自当前提交。WebUI 的源文件与构建来源会做只读 freshness 检查。该检查只核对文件，不证明程序功能通过验收。

```powershell
.\_internal\python_base\python.exe tools\build-release.py build --kind both --output-dir release-output --max-part-mib 1900
```

版本从该提交的 `release/version.json` 读取。对于 `0.1.0-preview`，输出名称如下：

```text
DFL-PT-WEBUI-0.1.0-preview-source.zip
DFL-PT-WEBUI-0.1.0-preview-source.manifest.json
DFL-PT-WEBUI-0.1.0-preview-source.archive.json
DFL-PT-WEBUI-0.1.0-preview-portable.zip.part001
DFL-PT-WEBUI-0.1.0-preview-portable.zip.part002
...
DFL-PT-WEBUI-0.1.0-preview-portable.manifest.json
DFL-PT-WEBUI-0.1.0-preview-portable.archive.json
restore-release.ps1
SHA256SUMS
```

未超过分卷大小的 ZIP 保持单个文件；超过时自动拆成普通二进制分卷，不依赖 RAR/7-Zip。默认每卷最多 1900 MiB，留在 GitHub 2 GiB 单文件上限以内。归档内部固定顶层目录为 `DFL-PT-WEBUI`。分卷需要构建磁盘暂时容纳完整 ZIP 和分卷，工具保守要求至少两倍原始总大小的空闲空间。现有同名材料不会被覆盖；重做时使用新的 `release-output/` 子目录。

每个 `.manifest.json` 记录公开源码提交、代码树 SHA、配置 SHA、包类型以及每个 payload 文件的路径、字节数、SHA-256 和材料类别。包内含相同 `RELEASE-MANIFEST.json` 和逐文件 `PAYLOAD-SHA256SUMS`；manifest 不对自身递归计算摘要。`.archive.json` 描述完整 ZIP 及分卷的尺寸和 SHA-256。外部 `SHA256SUMS` 覆盖发布的 ZIP/分卷、清单、索引和恢复脚本。哈希用于检测损坏和内容不一致，不是发行者签名。

若公开仓库采用没有本机开发历史的干净首提交，可以显式指定公开快照仓库。工具要求公开提交与当前 `--ref` 的 Git tree 完全相同，manifest 只记录公开 `sourceCommit` / `sourceTree`：

```powershell
.\_internal\python_base\python.exe tools\build-release.py build --kind both --source-commit <public-root-commit> --source-repository <public-snapshot-repository> --output-dir release-output
```

辅助权重的路径与固定 SHA 在配置中逐项列出。四个 `.npy` 的来源审查确认它们存在于固定 iperov/DeepFaceLab 上游树、该仓库根许可为 GPL-3.0；没有发现单独的权重许可文本。`licenseStatus: verified` 表示上游来源和仓库许可依据已核对，不表示存在单独的权重授权声明。SFace 使用专门的 Apache-2.0 文本。配置记录审查依据，禁止在来源审查尚未确认时静默捆绑。`--weights download` 可生成省略四个 `.npy` 的运行包，用户首次使用提取/增强前需执行现有下载准备脚本；脚本按 SHA 检查下载文件。通用 XSeg checkpoint 不在本次包中，需用户自行提供合法来源的模型。

## 用户恢复与展开

下载同一种包的全部 `.partNNN`（或单个 `.zip`）、对应 `.archive.json` / `.manifest.json`，以及 `restore-release.ps1` 和 `SHA256SUMS`，放在同一个目录。该脚本只要求本次包的相关文件；源码和便携包可以分别下载。运行一条命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\restore-release.ps1 -ArchiveIndex .\DFL-PT-WEBUI-0.1.0-preview-portable.archive.json -Destination .\unpacked
```

脚本校验索引、manifest、每个分卷和重组 ZIP，再检查 ZIP 文件路径和尺寸，展开至 `unpacked/DFL-PT-WEBUI`。目标目录须不存在或为空；已有文件不覆盖。恢复脚本不启动程序。临时重组 ZIP 会在结束后删除，下载的分卷保留。源码包同样使用此命令，修改索引文件名即可。

## 从清单只读核验

```powershell
.\_internal\python_base\python.exe tools\build-release.py verify --archive-index release-output\DFL-PT-WEBUI-0.1.0-preview-portable.archive.json --sha256sums release-output\SHA256SUMS
```

核验器直接读取单 ZIP 或分卷，不重组、不解压、不改文件。它检查每卷和完整 ZIP 的哈希、manifest 一致性、唯一文件路径、精确文件集合、尺寸、逐 payload SHA-256 与清单总数。`--sha256sums` 可选；指定时会检查该文件列出的所有已发布材料，因此该只读命令需要其列出的源码与便携材料都已下载。只下载一种包时省略此参数，其归档索引仍覆盖该包的所有分卷及 manifest。
