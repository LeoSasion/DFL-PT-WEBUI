import { spawn } from "node:child_process";
import path from "node:path";
import { PATHS } from "./paths.mjs";

const mode = process.argv[2];
if (!["standard", "lossless", "avi", "mov-lossless", "master", "quality"].includes(mode)) {
  process.stderr.write("[WEB] 编码模式无效；只允许 standard、lossless、avi、mov-lossless、master 或 quality。\r\n");
  process.exit(2);
}
const bitrate = process.argv[3] ?? "";
if (bitrate && (!/^\d{1,3}$/.test(bitrate) || Number(bitrate) < 1 || Number(bitrate) > 200)) {
  process.stderr.write("[WEB] 视频码率必须是 1–200 Mbps 的整数。\r\n");
  process.exit(2);
}

const common = [
  PATHS.currentMain,
  "videoed",
  "video-from-sequence",
];
const extension = mode === "master" ? "nut" : mode === "avi" ? "avi" : mode === "mov-lossless" ? "mov" : "mp4";
const resultIsLossless = ["lossless", "mov-lossless", "master"].includes(mode);

function outputSteps(selectedMode) {
 const extension = selectedMode === "master" ? "nut" : selectedMode === "avi" ? "avi" : selectedMode === "mov-lossless" ? "mov" : "mp4";
 const resultIsLossless = ["lossless", "mov-lossless", "master"].includes(selectedMode);
 return [
  {
    label: `正在生成 result.${extension}`,
    args: [
      ...common,
      "--input-dir",
      path.join(PATHS.workspaceRoot, "data_dst", "merged"),
      "--output-file",
      path.join(PATHS.workspaceRoot, `result.${extension}`),
      "--reference-file",
      path.join(PATHS.workspaceRoot, "data_dst.*"),
      "--include-audio",
      ...(bitrate ? ["--bitrate", bitrate] : selectedMode === "quality" ? ["--bitrate", "25"] : []),
      ...(resultIsLossless ? ["--lossless"] : []),
    ],
  },
  {
    label: `正在生成 result_mask.${extension}`,
    args: [
      ...common,
      "--input-dir",
      path.join(PATHS.workspaceRoot, "data_dst", "merged_mask"),
      "--output-file",
      path.join(PATHS.workspaceRoot, `result_mask.${extension}`),
      "--reference-file",
      path.join(PATHS.workspaceRoot, "data_dst.*"),
      "--lossless",
    ],
  },
];
}
const steps = mode === "quality" ? [...outputSteps("master"), ...outputSteps("quality")] : outputSteps(mode);

function runStep(step, index) {
  return new Promise((resolve, reject) => {
    process.stdout.write(
      `\r\n\u001b[38;2;44;227;159m[WEB ${index + 1}/${steps.length}]\u001b[0m ${step.label}\r\n`,
    );
    const child = spawn(PATHS.python, step.args, {
      cwd: PATHS.repositoryRoot,
      env: process.env,
      stdio: "inherit",
      windowsHide: true,
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(
        signal
          ? `${step.label} 被信号 ${signal} 终止`
          : `${step.label} 失败（退出码 ${code ?? "unknown"}）`,
      ));
    });
  });
}

try {
  process.stdout.write(mode === "master" || mode === "quality"
    ? "[WEB] FFV1 RGB 母版：封装后逐帧回读校验 RGB；原整数 PTS 与输出误差记录在 .media.json。\r\n"
    : resultIsLossless ? "[WEB] H.264 CRF 0 只保证编码后的 YUV 域；YUV420p 转换并非 RGB 全链路无损。\r\n" : "");
  for (const [index, step] of steps.entries()) {
    await runStep(step, index);
  }
  process.stdout.write(
    `\r\n\u001b[38;2;44;227;159m[WEB]\u001b[0m ${mode === "quality" ? "RGB 母版与 MP4 播放版" : `两个 ${extension.toUpperCase()} 文件`}均已生成。\r\n`,
  );
} catch (error) {
  process.stderr.write(
    `\r\n\u001b[38;2;255;122;122m[WEB]\u001b[0m ${error instanceof Error ? error.message : String(error)}\r\n`,
  );
  process.exitCode = 1;
}
