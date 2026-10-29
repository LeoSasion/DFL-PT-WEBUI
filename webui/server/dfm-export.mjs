import { sha256File } from "./hash-file.mjs";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";
import { runHelperProcess } from "./helper-process.mjs";
import { buildDflEnvironment } from "./environment.mjs";
import { PATHS } from "./paths.mjs";

const PROBE = `import importlib.util,json,sys
names=('modelopt','huggingface_hub','tensorrt')
missing=[name for name in names if importlib.util.find_spec(name) is None]
import torch
i=int(sys.argv[1]); available=torch.cuda.is_available() and i<torch.cuda.device_count()
print(json.dumps({'missing':missing,'capability':list(torch.cuda.get_device_capability(i)) if available else None}))`;

export async function verifyExtremeExport({ gpuIndex = "0", run = runHelperProcess } = {}) {
  if (!/^\d+$/.test(String(gpuIndex))) throw new Error("极致量化需要选择一张 GPU");
  const result = await run(PATHS.python, ["-I", "-c", PROBE, String(gpuIndex)], {
    cwd: PATHS.repositoryRoot, env: buildDflEnvironment("pytorch"),
    timeoutMs: 20_000, maxBytes: 16 * 1024, label: "DFM 极致量化环境检查",
  });
  if (result.code !== 0) throw new Error("无法检查 DFM 量化环境，请检查本项目 Python");
  const report = JSON.parse(result.stdout.trim());
  if (!Array.isArray(report.missing)) throw new Error("DFM 量化环境返回了无效结果");
  if (report.missing.length) throw new Error(`极致量化缺少 ${report.missing.join("、")}；请使用本项目 Python 安装 requirements-export.txt，或选择正常导出。`);
  if (!Array.isArray(report.capability) || ![10, 12].includes(report.capability[0])) throw new Error("极致量化需要 Blackwell GPU；当前设备可使用正常导出。");
  return report;
}

export function dfmFilename(mode) { return mode === "extreme" ? "model_extreme.dfm" : "model.dfm"; }

export async function isDfmPublicationPending(directory, name) {
  try {
    await lstat(path.join(directory, name + ".publish.json"));
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function assertDfmPublicationComplete(directory, name) {
  if (await isDfmPublicationPending(directory, name)) {
    throw new Error("DFM 发布尚未完成；请重新导出以恢复完整文件组");
  }
}

export async function verifyDfmReceipt(directory, { family, mode, jobId } = {}) {
  const name = dfmFilename(mode);
  await assertDfmPublicationComplete(directory, name);
  const reportPath = path.join(directory, name + ".export.json");
  const info = await lstat(reportPath);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 4 * 1024 * 1024) throw new Error("DFM 导出回执无效");
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  if (report.version !== 1 || report.family !== family || report.mode !== mode
      || (jobId && report.export_id !== jobId) || !Array.isArray(report.files)
      || !report.files.some(file => file.name === name)) throw new Error("DFM 回执与本次导出不匹配");
  for (const file of report.files) {
    if (typeof file.name !== "string" || path.basename(file.name) !== file.name || /[\\/:]/.test(file.name)
        || !Number.isSafeInteger(file.bytes) || file.bytes <= 0 || !/^[a-f0-9]{64}$/.test(file.sha256)) throw new Error("DFM 配套文件记录无效");
    const target = path.join(directory, file.name);
    const actual = await lstat(target);
    if (!actual.isFile() || actual.isSymbolicLink() || actual.size !== file.bytes) throw new Error(`DFM 配套文件不完整：${file.name}`);
    if (await sha256File(target) !== file.sha256) throw new Error(`DFM 配套文件校验失败：${file.name}`);
  }
  // A publication may have started while the receipt and large weights were read.
  await assertDfmPublicationComplete(directory, name);
  return report;
}
