import { cudaPreview } from "./cuda-preview.js";
import { sameProject } from "./cuda-runtime.js";

const listeners = new Map();
const pending = new Map();
let sequence = 0;

const query = new URLSearchParams(window.location.search);
const forcedMode = query.get("demo");
const nativeWebView = window.chrome && window.chrome.webview;
const mockRequests = [];
let prepareAttempts = 0;
let deleteAttempts = 0;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function now() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

const installPreview = {
  release: { applicationVersion: "0.1.2-preview", launcherVersion: "0.1.3-preview", installationSource: "local-source / unverified", sourceCommit: "unverified" },
  mode: "install",
  environmentStatus: "installing",
  installPath: "C:\\DFL-PT-WEBUI",
  projectDir: "C:\\DFL-PT-WEBUI",
  mirror: "official",
  mirrorLabel: "官方固定来源",
  gitProxyMode: "auto",
  gitProxy: "",
  gitMirror: "",
  gitNetworkLabel: "自动检测（未发现代理）",
  lastCheck: "—",
  webuiRunning: false,
  terminalUrl: "",
  steps: [
    { id: "environment", status: "active" },
    { id: "project", status: "upcoming" },
    { id: "dependencies", status: "upcoming" },
    { id: "finish", status: "upcoming" },
  ],
  runtimeItems: [
    { id: "project", label: "本地项目", status: "installed", detail: "仓库内容已获取", source: "GitHub", link: null },
    { id: "node", label: "Node.js", status: "installed", detail: "v24.19.0 · 可用", source: "官方固定来源" },
    { id: "python", label: "Python / PyTorch", status: "installed", detail: "3.12 / 2.9.1 cu128", source: "项目本地运行环境" },
    { id: "ffmpeg", label: "FFmpeg", status: "installed", detail: "本地视频工具", source: "项目本地运行环境" },
  ],
  logs: [
    { time: "12:15:03", text: "启动环境检测…" },
    { time: "12:15:04", text: "获取官方 PT 源码，不需要预装 Git…" },
    { time: "12:15:05", level: "success", text: "Node.js: v24.19.0 · 已安装" },
    { time: "12:15:06", text: "Python / PyTorch: 正在检测" },
    { time: "12:15:06", text: "运行环境：项目本地 Python 3.12" },
    { time: "12:15:12", level: "success", text: "Python / PyTorch / FFmpeg: 校验完成" },
  ],
};

const readyPreview = {
  release: { applicationVersion: "0.1.2-preview", launcherVersion: "0.1.3-preview", installationSource: "local-source / unverified", sourceCommit: "unverified" },
  mode: "ready",
  environmentStatus: "ready",
  installPath: "C:\\DFL-PT-WEBUI",
  projectDir: "C:\\DFL-PT-WEBUI\\workspace",
  mirror: "official",
  mirrorLabel: "官方固定来源",
  gitProxyMode: "auto",
  gitProxy: "",
  gitMirror: "",
  gitNetworkLabel: "自动检测（未发现代理）",
  lastCheck: "—",
  webuiRunning: false,
  terminalUrl: "",
  steps: [],
  runtimeItems: [],
  logs: [
    { time: "12:00:00", tag: "INFO", text: "DFL-PT-WEBUI 启动器已准备就绪" },
    { time: "12:00:00", tag: "INFO", text: "项目目录: C:\\DFL-PT-WEBUI\\workspace" },
    { time: "12:00:00", tag: "INFO", text: "检测运行环境…" },
    { time: "12:00:01", tag: "INFO", text: "Python 环境: OK" },
    { time: "12:00:01", tag: "INFO", text: "依赖环境: OK" },
    { time: "12:00:01", tag: "INFO", text: "项目文件完整性: OK" },
    { time: "12:00:01", tag: "INFO", text: "网络连接: OK" },
    { time: "12:00:01", tag: "OK", level: "success", text: "环境检查完成" },
    { text: "" },
    { level: "success", text: "================================================" },
    { level: "success", text: "  DFL-PT-WEBUI 启动器 - 交互菜单" },
    { level: "success", text: "================================================" },
    { text: "" },
    { text: "请选择操作:" },
    { text: "[1] 启动 WebUI（推荐）" },
    { text: "[2] 传统 BAT 模式" },
    { text: "[3] 首次配置 / 重新检测" },
    { text: "[4] 升级请使用发行页的新启动器" },
    { text: "[5] 修复依赖" },
    { text: "[6] PyTorch 官方匹配运行包" },
    { text: "[0] 退出启动器" },
  ],
};

let mockState = {
  ...(forcedMode === "install" ? installPreview : readyPreview),
  cudaRuntime: nativeWebView ? null : cudaPreview(query.get("cuda") || "compatible"),
  ...(query.get("cudaBusy") === "1" && !nativeWebView ? { webuiRunning: true } : {}),
};

export function previewState() {
  return clone(mockState);
}

export function previewRequests() {
  return clone(mockRequests);
}

function emit(event, data) {
  const handlers = listeners.get(event) || [];
  for (const handler of handlers) handler(data);
}

function mockRequest(method, params = {}) {
  mockRequests.push({ method, params: clone(params) });
  if (["prepareCudaRuntime", "keepCudaRuntime", "deleteCudaRuntime", "recoverCudaRuntime"].includes(method)) {
    return mockCudaRequest(method, params);
  }
  return new Promise((resolve) => {
    const delay = method === "getState" ? 20 : 450;
    window.setTimeout(() => {
      if (method === "getState") {
        resolve(clone(mockState));
        return;
      }
      if (method === "checkCudaRuntime") {
        resolve({ cudaRuntime: clone(mockState.cudaRuntime) });
        return;
      }
      if (method === "prepareFeedback") {
        resolve({ preview: "产品：DFL-PT-WEBUI\n应用版本：0.1.2-preview\n启动器版本：0.1.3-preview\n安装来源：本地预览 / 未验证\n源码快照：未验证\n失败步骤：" + (params.step || "unknown") + "\n\n重现步骤：\n1. \n\n预期行为：\n\n实际行为：", issueUrl: "https://github.com/LeoSasion/DFL-PT-WEBUI/issues/new?template=bug_report.yml" }); return;
      }
      if (method === "toggleMirror") {
        const china = mockState.mirror !== "china";
        mockState = { ...mockState, mirror: china ? "china" : "official", mirrorLabel: china ? "国内镜像" : "官方源" };
      } else if (method === "setGitNetwork") {
        const mode = params.mode || "auto";
        const label = mode === "direct" ? "直连" : mode === "manual" ? "手动代理" : "自动检测（未发现代理）";
        mockState = { ...mockState, gitProxyMode: mode, gitProxy: params.proxy || "", gitMirror: params.mirror || "", gitNetworkLabel: label + (params.mirror ? " · 已配置后备镜像" : "") };
      } else if (method === "checkUpdates") {
        mockState = { ...mockState, lastCheck: "刚刚" };
        emit("log", { time: now(), tag: "OK", level: "success", text: "已获取 origin/main，当前项目无需更新。" });
      } else if (method === "startWebUi") {
        mockState = { ...mockState, webuiRunning: true, webuiPid: 18420 };
        emit("log", { time: now(), tag: "INFO", text: "WebUI 正在启动，等待本地服务响应…" });
      } else if (method === "stopWebUi") {
        mockState = { ...mockState, webuiRunning: false, webuiPid: null };
        emit("log", { time: now(), tag: "OK", level: "success", text: "WebUI 已停止。" });
      } else if (method === "runFirstSetup") {
        mockState = { ...installPreview, logs: installPreview.logs };
      } else if (method === "runBootstrap" || method === "retryBootstrap") {
        mockState = { ...readyPreview, lastCheck: "刚刚" };
      } else if (method === "repairDependencies") {
        emit("log", { time: now(), tag: "OK", level: "success", text: "依赖校验完成，未发现缺失组件。" });
      } else if (method === "chooseInstallPath") {
        emit("log", { time: now(), tag: "INFO", text: "原生宿主中将打开目录选择器。" });
      }
      emit("state", clone(mockState));
      resolve({ state: clone(mockState) });
    }, delay);
  });
}

function wait(ms) { return new Promise((resolve) => window.setTimeout(resolve, ms)); }

async function mockCudaRequest(method, params) {
  if (!sameProject(params.expectedProjectDir, mockState.projectDir || mockState.installPath)) throw new Error("项目目录已变化，请重新检测后操作。");
  if (method !== "keepCudaRuntime" && (mockState.webuiRunning || mockState.terminalRunning)) throw new Error("请先结束任务、终端和 WebUI。");
  if (method === "prepareCudaRuntime") {
    if (!["cu126", "cu128"].includes(params.targetRuntime)) throw new Error("请选择匹配的运行环境。");
    prepareAttempts += 1;
    const previous = clone(mockState.cudaRuntime);
    const downloadBytes = previous.downloadBytes || cudaPreview("change-required").downloadBytes;
    const downloading = { ...previous, targetRuntime: params.targetRuntime, targetLabel: params.targetRuntime === "cu126" ? "CUDA 12.6" : "CUDA 12.8", status: "downloading", canSwitch: false, error: "", downloadedBytes: 0, downloadBytes };
    mockState = { ...mockState, cudaRuntime: downloading };
    emit("cudaRuntime", clone(downloading));
    await wait(350);
    if (query.get("cudaFailure") === "once" && prepareAttempts === 1) {
      mockState = { ...mockState, cudaRuntime: { ...previous, status: "error", error: "下载中断，原运行环境保持不变。请检查网络后重试。" } };
      emit("cudaRuntime", clone(mockState.cudaRuntime));
      throw new Error(mockState.cudaRuntime.error);
    }
    emit("cudaRuntime", { status: "verifying", downloadedBytes: downloadBytes, message: "下载完成，正在校验并启用新运行环境。" });
    await wait(350);
    const ready = cudaPreview("cleanup-choice");
    ready.currentRuntime = params.targetRuntime;
    ready.currentLabel = downloading.targetLabel;
    ready.targetRuntime = params.targetRuntime;
    ready.targetLabel = downloading.targetLabel;
    ready.retainedRuntime = previous.currentRuntime;
    ready.retainedRuntimes = ready.retainedRuntimes.map((row) => ({ ...row, runtime: previous.currentRuntime }));
    ready.message = `${ready.currentLabel} 已验证并启用，原运行包暂时保留。`;
    mockState = { ...mockState, cudaRuntime: ready };
  } else if (method === "recoverCudaRuntime") {
    if (!params.transactionId) throw new Error("缺少待恢复操作。");
    emit("cudaRuntime", { status: "recovering", message: "正在恢复原运行环境。" });
    await wait(450);
    mockState = { ...mockState, cudaRuntime: mockState.cudaRuntime.canRetryCleanup
      ? { ...mockState.cudaRuntime, canRetryCleanup: false, cleanupWarning: "", message: "已清理本项目下载与临时残留。" }
      : cudaPreview("compatible") };
  } else {
    const historical = mockState.cudaRuntime.retainedRuntimes.find((row) => row.transactionId === params.transactionId);
    if (!params.transactionId || (!historical && params.transactionId !== mockState.cudaRuntime.transactionId)) throw new Error("这份旧运行包的状态已变化，请重新检测。");
    if (method === "deleteCudaRuntime") {
      if (params.confirmed !== true) throw new Error("删除旧版需要单独确认。");
      deleteAttempts += 1;
      await wait(400);
      if (query.get("cudaDeleteFailure") === "once" && deleteAttempts === 1) throw new Error("旧运行包正在被占用，暂时无法删除。请结束相关进程后重试。");
      const remaining = mockState.cudaRuntime.retainedRuntimes.filter((row) => row.transactionId !== params.transactionId);
      mockState = { ...mockState, cudaRuntime: { ...mockState.cudaRuntime, ...(params.transactionId === mockState.cudaRuntime.transactionId ? { status: "compatible", canDelete: false, retainedBytes: 0, retainedRuntime: null, transactionId: null } : {}), retainedRuntimes: remaining, message: "旧版已删除，当前运行环境继续可用。" } };
    } else {
      if (historical?.deleting) throw new Error("旧版清理已经开始，不能将不完整的旧版标记为保留。");
      await wait(250);
      mockState = { ...mockState, cudaRuntime: { ...mockState.cudaRuntime, ...(params.transactionId === mockState.cudaRuntime.transactionId ? { status: "retained" } : {}), retainedRuntimes: mockState.cudaRuntime.retainedRuntimes.map((row) => row.transactionId === params.transactionId ? { ...row, kept: true } : row), message: "旧版 CUDA 运行包已保留，可以稍后清理。" } };
    }
  }
  emit("state", clone(mockState));
  return { state: clone(mockState) };
}

function handleNativeMessage(raw) {
  let message = raw;
  if (typeof raw === "string") {
    try {
      message = JSON.parse(raw);
    } catch {
      return;
    }
  }
  if (!message || typeof message !== "object") return;
  if (message.id && pending.has(message.id)) {
    const entry = pending.get(message.id);
    pending.delete(message.id);
    window.clearTimeout(entry.timeout);
    if (message.error) entry.reject(new Error(message.error.message || String(message.error)));
    else entry.resolve(message.result);
    return;
  }
  if (message.event) emit(message.event, message.data);
}

if (nativeWebView) {
  nativeWebView.addEventListener("message", (event) => handleNativeMessage(event.data));
}

export const launcherBridge = {
  get isNative() {
    return Boolean(nativeWebView);
  },
  on(event, handler) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(handler);
    return () => listeners.get(event) && listeners.get(event).delete(handler);
  },
  request(method, params = {}) {
    if (!nativeWebView) return mockRequest(method, params);
    const id = "ui-" + Date.now() + "-" + (++sequence);
    return new Promise((resolve, reject) => {
      const longRunning = ["runBootstrap", "retryBootstrap", "runFirstSetup", "repairDependencies", "applyUpdate", "prepareCudaRuntime", "deleteCudaRuntime", "recoverCudaRuntime"].includes(method);
      const timeoutMs = longRunning ? 2 * 60 * 60 * 1000 : 120000;
      const timeout = window.setTimeout(() => {
        pending.delete(id);
        reject(new Error("启动器宿主响应超时"));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timeout });
      nativeWebView.postMessage({ id, method, params });
    });
  },
};
