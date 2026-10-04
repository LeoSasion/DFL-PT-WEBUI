const listeners = new Map();
const pending = new Map();
let sequence = 0;

const query = new URLSearchParams(window.location.search);
const forcedMode = query.get("demo");
const nativeWebView = window.chrome && window.chrome.webview;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function now() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

const installPreview = {
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
    { text: "[6] PyTorch 官方 cu128 运行包" },
    { text: "[0] 退出启动器" },
  ],
};

let mockState = forcedMode === "install" ? installPreview : readyPreview;

export function previewState() {
  return clone(mockState);
}

function emit(event, data) {
  const handlers = listeners.get(event) || [];
  for (const handler of handlers) handler(data);
}

function mockRequest(method, params = {}) {
  return new Promise((resolve) => {
    const delay = method === "getState" ? 20 : 450;
    window.setTimeout(() => {
      if (method === "getState") {
        resolve(clone(mockState));
        return;
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
      const longRunning = ["runBootstrap", "retryBootstrap", "runFirstSetup", "repairDependencies", "applyUpdate"].includes(method);
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
