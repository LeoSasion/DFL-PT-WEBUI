const count = (workspace, key) => Number(workspace?.datasets?.[key]?.count ?? 0);
const hasFaces = (workspace, side) => count(workspace, `${side}Faces`) > 0;
const hasFrames = (workspace, side) => count(workspace, `${side}Frames`) > 0;

export function getUsageReadiness({ serviceState, health, telemetry, release } = {}) {
  const runtime = health?.runtime;
  const checks = release?.readiness;
  const toolsReady = checks?.toolsReady ?? (runtime ? Boolean(runtime.pythonAvailable && runtime.currentAvailable && runtime.pytorchAvailable) : null);
  const trainingReady = checks?.trainingReady ?? null;
  return {
    webReady: serviceState === "online",
    toolsReady,
    toolsVerified: checks?.toolsReady != null && checks?.toolsStatus !== "resource-presence-checked",
    trainingReady,
    gpuDetected: Boolean(telemetry?.available && telemetry.gpus?.length),
  };
}

export function getNextWorkflowStep(workspace, snapshotCount = 0, canEvaluate = false, environment = {}) {
  if (!workspace) return { stage: "material", task: "extract", label: "读取项目与素材状态", target: "workspace", pending: true };
  const readiness = workspace.readiness ?? {};
  const missing = ["src", "dst"].filter(side => !workspace.materials?.[side] && !hasFrames(workspace, side) && !hasFaces(workspace, side));
  if (missing.length && !readiness.materials && !readiness.faces) {
    return { stage: "material", task: "extract", label: `导入 ${missing.map(side => side.toUpperCase()).join(" / ")} 素材`, target: "workspace" };
  }
  const setup = environment.toolsReady === false
    ? { stage: "material", task: "extract", label: "检查工具资源与运行环境", target: "settings" }
    : null;
  const frameSide = ["src", "dst"].find(side => !hasFaces(workspace, side) && !hasFrames(workspace, side));
  if (!readiness.faces && !readiness.frames && frameSide) {
    return setup ?? { stage: "frames", task: "extract", label: `提取 ${frameSide.toUpperCase()} 视频帧`, commandId: `${frameSide}.extract_frames` };
  }
  if (!readiness.faces && !(hasFaces(workspace, "src") && hasFaces(workspace, "dst"))) {
    const side = hasFaces(workspace, "src") ? "dst" : "src";
    return setup ?? { stage: "faces", task: side, label: `提取 ${side.toUpperCase()} aligned 人脸`, commandId: `${side}.extract_faces` };
  }
  const training = () => setup ?? (environment.trainingReady === false
    ? { stage: "train", task: "me", label: "检查 GPU 训练条件", target: "settings" }
    : { stage: "train", task: "me", label: "配置 ME 并检查训练条件", commandId: "train.me" });
  if (!readiness.me) {
    if (!workspace.roles?.src?.current || !workspace.roles?.dst?.current) return { stage: "clean", task: "sort", label: "选择人物", target: "roles" };
    return training();
  }
  if (snapshotCount < 2) return canEvaluate
    ? { stage: "diagnose", task: "diagnose", label: "生成质量评估快照", target: "diagnostics" }
    : training();
  if (!readiness.merged) return setup ?? { stage: "merge", task: "merge", label: "合成 ME 人脸", commandId: "merge.me" };
  if (!readiness.encoded) return setup ?? { stage: "encode", task: "export", label: "导出 MP4", commandId: "encode.mp4" };
  return { stage: "encode", task: "export", label: "查看项目成果", target: "export" };
}
