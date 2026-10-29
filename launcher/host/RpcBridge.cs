using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using Microsoft.Web.WebView2.Core;

namespace DflPtWebUi.Launcher
{
    internal sealed class RpcBridge : IDisposable
    {
        private readonly MainWindow window;
        private readonly CoreWebView2 webView;
        private readonly HostController controller;
        private readonly JavaScriptSerializer serializer = new JavaScriptSerializer();
        private volatile bool disposed;

        public RpcBridge(MainWindow window, CoreWebView2 webView, HostController controller)
        {
            this.window = window;
            this.webView = webView;
            this.controller = controller;
            serializer.MaxJsonLength = Int32.MaxValue;

            webView.WebMessageReceived += OnWebMessageReceived;
            controller.Logs.EntryAdded += OnLogEntryAdded;
            controller.ProgressChanged += OnProgressChanged;
            controller.CudaRuntimeChanged += OnCudaRuntimeChanged;
        }

        public void Dispose()
        {
            disposed = true;
            webView.WebMessageReceived -= OnWebMessageReceived;
            controller.Logs.EntryAdded -= OnLogEntryAdded;
            controller.ProgressChanged -= OnProgressChanged;
            controller.CudaRuntimeChanged -= OnCudaRuntimeChanged;
        }

        internal static bool IsTrustedMessageSource(string source)
        {
            Uri uri;
            if (!Uri.TryCreate(source, UriKind.Absolute, out uri))
            {
                return false;
            }
            return String.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase)
                && String.Equals(
                    uri.DnsSafeHost,
                    LauncherConstants.VirtualHost,
                    StringComparison.OrdinalIgnoreCase)
                && uri.IsDefaultPort
                && String.IsNullOrEmpty(uri.UserInfo);
        }

        public void SendEvent(string eventName, object data)
        {
            Dictionary<string, object> payload = new Dictionary<string, object>();
            payload["event"] = eventName;
            payload["data"] = data;
            Post(payload);
        }

        public async Task PushStateAsync()
        {
            try
            {
                object state = await controller.GetStateAsync();
                SendEvent("state", state);
            }
            catch (Exception error)
            {
                controller.Logs.Add("launcher", "状态刷新失败：" + error.Message, "error");
            }
        }

        private async void OnWebMessageReceived(object sender, CoreWebView2WebMessageReceivedEventArgs args)
        {
            if (disposed) return;
            if (!IsTrustedMessageSource(args.Source))
            {
                controller.Logs.Add("security", "已拒绝来自非启动器页面的 WebMessage。", "warning");
                return;
            }

            object requestId = null;
            try
            {
                string json;
                try
                {
                    json = args.TryGetWebMessageAsString();
                }
                catch
                {
                    json = args.WebMessageAsJson;
                }

                object parsed;
                lock (serializer) { parsed = serializer.DeserializeObject(json); }
                Dictionary<string, object> request = parsed as Dictionary<string, object>;
                if (request == null)
                {
                    throw new RpcException("invalid_request", "WebMessage 必须是 JSON 对象。");
                }

                if (request.ContainsKey("id"))
                {
                    requestId = request["id"];
                }
                string method = ReadString(request, "method");
                if (String.IsNullOrWhiteSpace(method))
                {
                    method = ReadString(request, "action");
                }
                if (String.IsNullOrWhiteSpace(method))
                {
                    throw new RpcException("method_required", "缺少 RPC method。");
                }

                Dictionary<string, object> parameters = ReadDictionary(request, "params");
                if (parameters == null)
                {
                    parameters = ReadDictionary(request, "payload") ?? new Dictionary<string, object>();
                }

                object result = await DispatchAsync(method, parameters);
                Dictionary<string, object> response = new Dictionary<string, object>();
                response["id"] = requestId;
                response["result"] = result;
                Post(response);

                if (ChangesState(method))
                {
                    await PushStateAsync();
                }
            }
            catch (Exception error)
            {
                RpcException rpcError = error as RpcException;
                Dictionary<string, object> detail = new Dictionary<string, object>();
                detail["code"] = rpcError == null ? "host_error" : rpcError.Code;
                detail["message"] = error.Message;
                if (error.InnerException != null)
                {
                    detail["detail"] = error.InnerException.Message;
                }
                Dictionary<string, object> response = new Dictionary<string, object>();
                response["id"] = requestId;
                response["error"] = detail;
                Post(response);
                controller.Logs.Add("launcher", error.Message, "error");
            }
        }

        private async Task<object> DispatchAsync(string method, Dictionary<string, object> parameters)
        {
            switch (method)
            {
                case "getState":
                case "status.get":
                case "getStatus":
                    return await controller.GetStateAsync();

                case "runBootstrap":
                case "retryBootstrap":
                case "runFirstSetup":
                    return await controller.RunBootstrapAsync(false);

                case "repairDependencies":
                    return await controller.RunBootstrapAsync(true);

                case "checkCudaRuntime":
                    return await controller.CheckCudaRuntimeAsync(ReadBoolean(parameters, "force", false));

                case "prepareCudaRuntime":
                    return await controller.PrepareCudaRuntimeAsync(ReadRequiredString(parameters, "targetRuntime"), ReadRequiredString(parameters, "expectedProjectDir"));

                case "keepCudaRuntime":
                    return await controller.KeepCudaRuntimeAsync(ReadRequiredString(parameters, "transactionId"), ReadRequiredString(parameters, "expectedProjectDir"));

                case "deleteCudaRuntime":
                    return await controller.DeleteCudaRuntimeAsync(ReadRequiredString(parameters, "transactionId"), ReadBoolean(parameters, "confirmed", false), ReadRequiredString(parameters, "expectedProjectDir"));

                case "recoverCudaRuntime":
                    return await controller.RecoverCudaRuntimeAsync(ReadRequiredString(parameters, "transactionId"), ReadRequiredString(parameters, "expectedProjectDir"));

                case "chooseInstallPath":
                case "project.select":
                    return controller.ChooseInstallPath(ReadString(parameters, "path"));

                case "setMirror":
                    return controller.SetMirror(ReadString(parameters, "mirror") ?? ReadString(parameters, "value"));

                case "toggleMirror":
                    return controller.ToggleMirror();

                case "setGitNetwork":
                    return controller.SetGitNetwork(
                        ReadString(parameters, "mode"),
                        ReadString(parameters, "proxy"),
                        ReadString(parameters, "mirror"));

                case "checkUpdates":
                case "git.check":
                    return await controller.CheckUpdatesAsync();

                case "applyUpdate":
                case "git.update":
                    return await controller.ApplyUpdateAsync();

                case "rollbackUpdate":
                    return await controller.RollbackUpdateAsync();

                case "prepareFeedback":
                    return await controller.PrepareFeedbackAsync(ReadString(parameters, "step"));

                case "startWebUi":
                case "webui.start":
                    return await controller.StartWebUiAsync();

                case "stopWebUi":
                case "webui.stop":
                    return await controller.StopWebUiAsync();

                case "openLegacy":
                    return await controller.OpenLegacyAsync();

                case "terminal.start":
                case "startTerminalBridge":
                    return await controller.StartTerminalAsync();

                case "openExternal":
                    return controller.OpenExternal(ReadString(parameters, "url"));

                case "logs.poll":
                case "getLogs":
                    return controller.PollLogs(ReadLong(parameters, "since", 0), ReadInt(parameters, "limit", 200));

                case "window.minimize":
                    window.MinimizeWindow();
                    return Success();

                case "window.maximize":
                    window.ToggleMaximizeWindow();
                    return new Dictionary<string, object> { { "maximized", window.IsWindowMaximized } };

                case "window.close":
                    window.CloseWindow();
                    return Success();

                case "window.drag":
                    window.DragWindow();
                    return Success();

                default:
                    throw new RpcException("method_not_found", "未知 RPC method：" + method);
            }
        }

        private void OnLogEntryAdded(LogEntry entry)
        {
            SendEvent("log", new Dictionary<string, object>
            {
                { "channel", entry.Channel },
                { "line", entry.Line },
                { "level", entry.Level },
                { "sequence", entry.Sequence },
                { "timestamp", entry.Timestamp }
            });
        }

        private void OnProgressChanged(object value)
        {
            SendEvent("progress", value);
        }

        private void OnCudaRuntimeChanged(object value)
        {
            SendEvent("cudaRuntime", value);
        }

        private void Post(object value)
        {
            if (disposed) return;
            // Background package operations outlive a closed or navigating view.
            // Posting their progress must never abort the operation or its logs.
            try
            {
                string json;
                lock (serializer) { json = serializer.Serialize(value); }
                window.Dispatcher.BeginInvoke(new Action(delegate
                {
                    try { if (!disposed) webView.PostWebMessageAsJson(json); } catch { }
                }));
            }
            catch { }
        }

        private static bool ChangesState(string method)
        {
            return method != "getState"
                && method != "status.get"
                && method != "getStatus"
                // These return their state explicitly. Periodic CUDA checks
                // must not launch the full Torch/WebUI health inspection.
                && method != "checkCudaRuntime"
                && method != "prepareCudaRuntime"
                && method != "keepCudaRuntime"
                && method != "deleteCudaRuntime"
                && method != "recoverCudaRuntime"
                && method != "logs.poll"
                && method != "getLogs"
                && method != "window.drag";
        }

        private static Dictionary<string, object> ReadDictionary(Dictionary<string, object> source, string key)
        {
            if (source == null || !source.ContainsKey(key))
            {
                return null;
            }
            return source[key] as Dictionary<string, object>;
        }

        private static string ReadString(Dictionary<string, object> source, string key)
        {
            if (source == null || !source.ContainsKey(key) || source[key] == null)
            {
                return null;
            }
            return Convert.ToString(source[key]);
        }

        private static string ReadRequiredString(Dictionary<string, object> source, string key)
        {
            object value;
            if (source == null || !source.TryGetValue(key, out value) || !(value is string) || String.IsNullOrWhiteSpace((string)value))
                throw new RpcException("invalid_params", "缺少有效参数：" + key);
            return (string)value;
        }

        private static bool ReadBoolean(Dictionary<string, object> source, string key, bool fallback)
        {
            object value;
            if (source == null || !source.TryGetValue(key, out value)) return fallback;
            if (!(value is bool)) throw new RpcException("invalid_params", "参数必须为布尔值：" + key);
            return (bool)value;
        }

        private static int ReadInt(Dictionary<string, object> source, string key, int fallback)
        {
            if (source == null || !source.ContainsKey(key) || source[key] == null)
            {
                return fallback;
            }
            int value;
            return Int32.TryParse(Convert.ToString(source[key]), out value) ? value : fallback;
        }

        private static long ReadLong(Dictionary<string, object> source, string key, long fallback)
        {
            if (source == null || !source.ContainsKey(key) || source[key] == null)
            {
                return fallback;
            }
            long value;
            return Int64.TryParse(Convert.ToString(source[key]), out value) ? value : fallback;
        }

        private static object Success()
        {
            return new Dictionary<string, object> { { "ok", true } };
        }
    }

    internal sealed class RpcException : Exception
    {
        public string Code { get; private set; }

        public RpcException(string code, string message)
            : base(message)
        {
            Code = code;
        }
    }
}
