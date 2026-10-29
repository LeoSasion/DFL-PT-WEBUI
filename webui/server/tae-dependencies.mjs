import { runHelperProcess } from "./helper-process.mjs";
import { buildDflEnvironment } from "./environment.mjs";
import { PATHS } from "./paths.mjs";

const MODULES = ["safetensors"];
const PROBE = "import importlib.util,json;print(json.dumps({'missing':[name for name in ('safetensors',) if importlib.util.find_spec(name) is None]}))";

// This checks package availability only. The TAE loader still validates the
// exact VAE asset and runtime contract before constructing the model.
export async function verifyTaeDependencies({ run = runHelperProcess } = {}) {
  let result;
  try {
    result = await run(PATHS.python, ["-I", "-c", PROBE], {
      cwd: PATHS.repositoryRoot, env: buildDflEnvironment("pytorch"),
      timeoutMs: 10_000, maxBytes: 16 * 1024, label: "TAE 依赖检查",
    });
    if (result.code !== 0) throw new Error("Python probe failed");
    const report = JSON.parse(result.stdout.trim());
    if (!Array.isArray(report.missing) || report.missing.some(name => !MODULES.includes(name))) {
      throw new Error("Invalid dependency probe response");
    }
    if (report.missing.length) {
      const error = new Error(`TAE 缺少依赖：${report.missing.join("、")}。请使用本项目 Python 安装 requirements-tae.txt 中的固定版本后重试。`);
      error.code = "TAE_DEPENDENCY_MISSING";
      throw error;
    }
    return report;
  } catch (error) {
    if (error.code === "TAE_DEPENDENCY_MISSING") throw error;
    const failure = new Error("无法检查 TAE 依赖；请检查本项目 Python 运行环境后重试。");
    failure.code = "TAE_RUNTIME_INVALID";
    throw failure;
  }
}
