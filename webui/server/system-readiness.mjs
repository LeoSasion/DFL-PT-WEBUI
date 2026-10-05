import { access } from "node:fs/promises";
import path from "node:path";

export async function inspectToolResources(root) {
  const required = {
    python: ".venv/Scripts/python.exe", ffmpeg: "_internal/ffmpeg/ffmpeg.exe", ffprobe: "_internal/ffmpeg/ffprobe.exe",
    extractor: "_internal/DeepFaceLab/main.py", me: "_internal/DeepFaceLab/me.py",
    detector: "_internal/DeepFaceLab/facelib/S3FD.npy", landmarks: "_internal/DeepFaceLab/facelib/2DFAN.npy",
  };
  const checks = Object.fromEntries(await Promise.all(Object.entries(required).map(async ([name, relative]) => {
    try { await access(path.join(root, relative)); return [name, true]; }
    catch { return [name, false]; }
  })));
  return { toolsReady: Object.values(checks).every(Boolean), toolsStatus: "resource-presence-checked",
    trainingReady: null, trainingStatus: "requires-command-preflight", checks };
}
