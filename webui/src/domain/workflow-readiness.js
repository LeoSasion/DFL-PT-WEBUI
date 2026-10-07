const count = (workspace, key) => Number(workspace?.datasets?.[key]?.count ?? 0);
const hasFaces = (workspace, side) => count(workspace, `${side}Faces`) > 0;
const hasFrames = (workspace, side) => count(workspace, `${side}Frames`) > 0;
const activeTrainingStates = new Set(["queued", "starting", "running", "waiting_input", "stopping"]);

// These are discovery facts, not media/checkpoint integrity verification. The
// task preflight remains responsible for validating individual input members.
export function getWorkflowArtifactState(workspace) {
  const readiness = workspace?.readiness ?? {};
  const mergedCount = count(workspace, "merged");
  const maskCount = count(workspace, "mergedMask");
  const countsKnown = workspace?.datasets?.merged != null && workspace?.datasets?.mergedMask != null;
  const mergedAvailable = countsKnown ? mergedCount > 0 && maskCount > 0 : Boolean(readiness.merged);
  return {
    meAvailable: Boolean(readiness.me || workspace?.models?.some(model => model.type?.toUpperCase() === "ME" && model.ready !== false)),
    mergedAvailable,
    mergedCount,
    maskCount,
    countsKnown,
    partialMerge: countsKnown && ((mergedCount > 0 || maskCount > 0) && mergedCount !== maskCount),
    exportConfiguredReady: mergedAvailable && (!countsKnown || mergedCount === maskCount),
    encodedAvailable: Boolean(readiness.encoded || workspace?.outputs?.some(output => !output.name?.includes("_mask"))),
    evidence: "file-discovery",
  };
}

export function getTrainingSaveStatus(job, savedModel) {
  const historyAvailable = Boolean(savedModel && savedModel.ready !== false);
  let saveState = "unknown";
  let detail = job ? "最近任务的结束状态不代表本次检查点已经保存；请检查模型文件与任务日志。" : "还没有本次训练任务的保存记录。";
  if (job?.state === "stopping") {
    saveState = "pending";
    detail = "正在等待 Trainer 保存并退出；完成前不要将当前预览视为已保存成果。";
  } else if (job?.commandId === "train.me" && job?.state === "cancelled" && job.stopReason === "safe-stop") {
    saveState = "confirmed";
    detail = "最近 ME 任务已收到保存并退出确认；模型完整性仍由使用前检查确认。";
  } else if (["safe-stop-timeout", "safe-stop-unconfirmed", "force-kill"].includes(job?.stopReason)) {
    saveState = "unconfirmed";
    detail = historyAvailable
      ? "已有历史检查点仍可查看；最近任务的本次保存未确认，请先检查恢复点与日志。"
      : "最近任务的本次保存未确认，请检查模型目录与日志后再继续。";
  } else if (job?.stopReason === "safe-stop-before-start") {
    saveState = "not-started";
    detail = "任务在训练开始前停止，没有产生本次训练保存。";
  } else if (activeTrainingStates.has(job?.state)) {
    detail = "正在运行的任务与磁盘上的历史检查点分别记录；当前预览不代表已保存。";
  }
  return { historyAvailable, saveState, detail, active: activeTrainingStates.has(job?.state),
    historyLabel: historyAvailable ? "检测到历史检查点" : "尚未检测到历史检查点" };
}

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
  const artifacts = getWorkflowArtifactState(workspace);
  // An existing deliverable remains useful without original training inputs or
  // two diagnostics snapshots. Full navigation is available independently.
  if (artifacts.encodedAvailable) return { stage: "encode", task: "export", label: "查看项目成果", target: "export" };
  if (artifacts.exportConfiguredReady) return environment.toolsReady === false
    ? { stage: "encode", task: "export", label: "检查导出工具资源与运行环境", target: "settings" }
    : { stage: "encode", task: "export", label: "导出母版与播放版", commandId: "encode.quality" };
  if (artifacts.partialMerge) return { stage: "merge", task: "merge", label: "检查并补齐合成序列", commandId: "merge.me" };
  if (artifacts.meAvailable) {
    if (snapshotCount < 2 && canEvaluate) return { stage: "diagnose", task: "diagnose", label: "生成质量评估快照", target: "diagnostics" };
    return environment.toolsReady === false
      ? { stage: "merge", task: "merge", label: "检查合成工具资源与运行环境", target: "settings" }
      : { stage: "merge", task: "merge", label: "检查模型并配置合成", commandId: "merge.me" };
  }
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
  if (!workspace.roles?.src?.current || !workspace.roles?.dst?.current) return { stage: "clean", task: "sort", label: "选择人物", target: "roles" };
  return training();
}
