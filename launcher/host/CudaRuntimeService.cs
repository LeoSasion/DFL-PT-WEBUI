using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace DflPtWebUi.Launcher
{
    internal delegate Task<CommandResult> CudaRuntimeCommand(
        string executable, string arguments, string workingDirectory,
        IDictionary<string, string> environment, string channel, Func<string, bool> outputLine);

    // Checking hardware never installs packages. All mutations are separate,
    // explicitly invoked operations whose journal is owned by the Python tool.
    internal sealed class CudaRuntimeService
    {
        private const string EventPrefix = "DFL_CUDA_EVENT|";
        private readonly object gate = new object();
        private readonly CudaRuntimeCommand execute;
        private readonly Action<object> changed;
        private readonly Func<DateTime> now;
        private readonly JavaScriptSerializer serializer = new JavaScriptSerializer { MaxJsonLength = 1024 * 1024 };
        private sealed class CachedStatus
        {
            public Dictionary<string, object> Value;
            public DateTime At;
        }
        private readonly Dictionary<string, CachedStatus> cached = new Dictionary<string, CachedStatus>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, Task<Dictionary<string, object>>> pending = new Dictionary<string, Task<Dictionary<string, object>>>(StringComparer.OrdinalIgnoreCase);
        private long generation;
        private bool paused;
        private bool changing;

        public CudaRuntimeService(ProcessRunner runner, Action<object> changed)
            : this(runner.RunAsync, changed, delegate { return DateTime.UtcNow; }) { }

        internal CudaRuntimeService(CudaRuntimeCommand execute, Action<object> changed, Func<DateTime> now)
        {
            if (execute == null || now == null) throw new ArgumentNullException();
            this.execute = execute;
            this.changed = changed;
            this.now = now;
        }

        public bool IsChanging { get { lock (gate) { return changing || paused; } } }

        public Task<Dictionary<string, object>> GetStatusAsync(string projectRoot, bool force)
        {
            string root = NormalizeRoot(projectRoot);
            lock (gate)
            {
                if (paused || changing)
                    return Task.FromResult(CachedOrUnavailable(root));
                CachedStatus value;
                if (!force && cached.TryGetValue(root, out value))
                {
                    TimeSpan age = now() - value.At;
                    if (age >= TimeSpan.Zero && age < TimeSpan.FromSeconds(20)) return Task.FromResult(Copy(value.Value));
                }
                PruneCompletedQueries();
                Task<Dictionary<string, object>> query;
                if (pending.TryGetValue(root, out query)) return query;
                query = ReadStatusAsync(root, generation);
                pending[root] = query;
                return query;
            }
        }

        // Finish our own Python status query before the host checks for project
        // processes. Otherwise that query could be mistaken for a training task.
        public async Task PauseChecksAsync()
        {
            Task<Dictionary<string, object>>[] queries;
            lock (gate) { paused = true; queries = new List<Task<Dictionary<string, object>>>(pending.Values).ToArray(); }
            // Project selection can leave queries for several roots in flight.
            // Drain all of them before an idle check or a package mutation.
            try { await Task.WhenAll(queries); } catch { }
            lock (gate) { PruneCompletedQueries(); }
        }

        public void ResumeChecks()
        {
            lock (gate) { paused = false; }
        }

        public Task<Dictionary<string, object>> PrepareAsync(string root, string target)
        {
            ValidateTarget(target);
            return MutateAsync(root, "prepare", " --target " + target, target);
        }

        public Task<Dictionary<string, object>> KeepAsync(string root, string transactionId)
        {
            ValidateTransactionId(transactionId);
            return MutateAsync(root, "keep-retained", " --transaction-id " + transactionId, null);
        }

        public Task<Dictionary<string, object>> DeleteAsync(string root, string transactionId, bool confirmed)
        {
            ValidateTransactionId(transactionId);
            if (!confirmed) throw new InvalidOperationException("请先明确确认删除旧 GPU 运行环境。");
            return MutateAsync(root, "delete-retained", " --transaction-id " + transactionId + " --confirm-delete", null);
        }

        public Task<Dictionary<string, object>> RecoverAsync(string root, string transactionId)
        {
            ValidateTransactionId(transactionId);
            return MutateAsync(root, "recover", " --transaction-id " + transactionId, null);
        }

        internal static void ValidateTarget(string target)
        {
            if (target != "cu126" && target != "cu128")
                throw new InvalidOperationException("请选择受支持的 CUDA 12.6 或 CUDA 12.8 运行环境。");
        }

        internal static void ValidateTransactionId(string transactionId)
        {
            if (String.IsNullOrWhiteSpace(transactionId) || !Regex.IsMatch(transactionId, "^[0-9a-f]{32}$"))
                throw new InvalidOperationException("GPU 环境切换记录无效，请重新检测环境。");
        }

        internal static bool Flag(IDictionary<string, object> state, string name)
        {
            object value;
            return state != null && state.TryGetValue(name, out value) && value is bool && (bool)value;
        }

        private async Task<Dictionary<string, object>> ReadStatusAsync(string root, long version)
        {
            Dictionary<string, object> state;
            try { state = await ReadStatusDirectAsync(root); }
            catch (Exception error) { state = Unavailable("GPU 环境检测未完成：" + error.Message); }
            state["projectDir"] = root;
            lock (gate)
            {
                if (!changing && generation == version)
                {
                    Store(root, state);
                    return Copy(state);
                }
                // The caller can still be awaiting an old read when a
                // mutation publishes newer progress. Do not return that old
                // assessment even though it was excluded from the cache.
                return CachedOrUnavailable(root);
            }
        }

        private async Task<Dictionary<string, object>> ReadStatusDirectAsync(string root)
        {
            string python, tool;
            if (!TryResolveTool(root, out python, out tool))
                return Unavailable("当前项目尚未提供 GPU 环境切换工具，请使用包含此功能的项目版本。");
            CommandResult result = await execute(python, Arguments(tool, "status"), root, EnvironmentFor(root), "cuda-runtime", null);
            if (!result.Success) throw Failure(result, "GPU 环境检测失败");
            return ParseFinal(result.StandardOutput);
        }

        private async Task<Dictionary<string, object>> MutateAsync(string projectRoot, string command, string suffix, string target)
        {
            string root = NormalizeRoot(projectRoot);
            string python, tool;
            if (!TryResolveTool(root, out python, out tool))
                throw new InvalidOperationException("当前项目缺少 GPU 环境切换工具，请先更新项目源码。");
            Dictionary<string, object> initial;
            lock (gate)
            {
                if (changing) throw new InvalidOperationException("GPU 环境操作正在进行，请等待完成。");
                changing = true;
                generation++;
                initial = CachedOrUnavailable(root);
                initial["status"] = command == "prepare" ? "downloading" : command == "keep-retained" ? "cleanup-choice" : "switching";
                initial["canSwitch"] = false;
                initial["canDelete"] = false;
                initial["canRecover"] = false;
                initial["canRetryCleanup"] = false;
                initial["error"] = null;
                foreach (string key in new[] { "downloadedBytes", "cachedBytes", "downloadRemainingBytes", "totalBytes", "percent", "phase" }) initial.Remove(key);
                if (target != null) { initial["targetRuntime"] = target; initial["transactionId"] = null; }
                initial["message"] = command == "prepare" ? "正在准备匹配显卡的 GPU 运行环境…"
                    : command == "keep-retained" ? "正在保留旧版运行环境…"
                    : command == "recover" ? "正在恢复未完成的环境切换…" : "正在清理已确认删除的旧版运行环境…";
                Store(root, initial);
            }
            Notify(initial);
            try
            {
                Dictionary<string, object> state = null;
                Exception failure = null;
                try
                {
                    CommandResult result = await execute(python, Arguments(tool, command) + suffix, root,
                        EnvironmentFor(root), "cuda-runtime", delegate(string line) { return HandleProgress(root, line); });
                    if (!result.Success) throw Failure(result, "GPU 环境操作未完成");
                    state = ParseFinal(result.StandardOutput);
                }
                catch (Exception error) { failure = error; }
                // C# 5 cannot await inside catch. Refresh outside it, preserving
                // the original operation error after exposing recovery state.
                if (failure != null)
                {
                    try { state = await ReadStatusDirectAsync(root); }
                    catch { state = Unavailable("环境操作未完成，请重新检测并查看日志。", failure.Message); }
                    lock (gate) { Store(root, state); }
                    Notify(state);
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
                }
                lock (gate) { Store(root, state); }
                Notify(state);
                return Copy(state);
            }
            finally
            {
                lock (gate) { changing = false; PruneCompletedQueries(); }
            }
        }

        private bool HandleProgress(string root, string line)
        {
            if (line == null || !line.StartsWith(EventPrefix, StringComparison.Ordinal)) return false;
            Dictionary<string, object> update;
            try { update = Deserialize(line.Substring(EventPrefix.Length)); }
            catch { return true; }
            if (update.ContainsKey("transactionId"))
            {
                try { ValidateTransactionId(Text(update, "transactionId")); }
                catch { return true; }
            }
            Dictionary<string, object> state;
            lock (gate)
            {
                state = CachedOrUnavailable(root);
                foreach (string key in new[] { "message", "downloadedBytes", "downloadBytes", "cachedBytes", "downloadRemainingBytes", "totalBytes", "phase", "percent", "currentRuntime", "targetRuntime", "currentLabel", "targetLabel", "transactionId", "downloadRequired" })
                    if (update.ContainsKey(key)) state[key] = update[key];
                object phase;
                if (update.TryGetValue("status", out phase) || update.TryGetValue("phase", out phase))
                {
                    string name = Convert.ToString(phase);
                    if (name == "switching" || name == "activating" || name == "recovering") state["status"] = "switching";
                }
                state["canSwitch"] = false;
                state["canDelete"] = false;
                state["canRecover"] = false;
                state["canRetryCleanup"] = false;
                Store(root, state);
            }
            Notify(state);
            return true;
        }

        private Dictionary<string, object> ParseFinal(string output)
        {
            string last = null;
            foreach (string line in (output ?? String.Empty).Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                if (!line.StartsWith(EventPrefix, StringComparison.Ordinal) && line.TrimStart().StartsWith("{", StringComparison.Ordinal)) last = line;
            if (last == null) throw new InvalidOperationException("GPU 环境工具未返回有效状态。");
            Dictionary<string, object> state = Deserialize(last);
            string status = Text(state, "status");
            if (Array.IndexOf(new[] { "compatible", "change-required", "unavailable", "ambiguous", "cleanup-choice", "retained", "error", "downloading", "switching" }, status) < 0)
                throw new InvalidOperationException("GPU 环境工具返回了未知状态。");
            string current = Text(state, "currentRuntime"), target = Text(state, "targetRuntime");
            if ((!String.IsNullOrEmpty(current) && current != "cu126" && current != "cu128")
                || (!String.IsNullOrEmpty(target) && target != "cu126" && target != "cu128"))
                throw new InvalidOperationException("GPU 环境工具返回了不受支持的运行版本。");
            bool hasTransaction = !String.IsNullOrEmpty(Text(state, "transactionId"));
            if (hasTransaction) ValidateTransactionId(Text(state, "transactionId"));
            state["canDelete"] = Flag(state, "canDelete") && hasTransaction && (status == "cleanup-choice" || status == "retained");
            state["canRecover"] = Flag(state, "canRecover") && hasTransaction;
            state["canRetryCleanup"] = Flag(state, "canRetryCleanup") && hasTransaction;
            return state;
        }

        private Dictionary<string, object> Deserialize(string value)
        {
            lock (serializer)
            {
                Dictionary<string, object> state = serializer.DeserializeObject(value) as Dictionary<string, object>;
                if (state == null) throw new InvalidOperationException("GPU 环境状态格式无效。");
                return state;
            }
        }

        private static string Text(IDictionary<string, object> value, string key)
        {
            object item;
            return value.TryGetValue(key, out item) && item is string ? (string)item : null;
        }

        private static InvalidOperationException Failure(CommandResult result, string label)
        {
            string detail = result.StandardError;
            if (String.IsNullOrWhiteSpace(detail)) detail = result.StandardOutput;
            if (detail != null && detail.Length > 3000) detail = detail.Substring(detail.Length - 3000);
            return new InvalidOperationException(label + "（退出码 " + result.ExitCode + "）" + (String.IsNullOrWhiteSpace(detail) ? "。" : "：" + detail));
        }

        private static string NormalizeRoot(string projectRoot)
        {
            if (String.IsNullOrWhiteSpace(projectRoot)) throw new InvalidOperationException("请先选择项目目录。");
            string root = Path.GetFullPath(projectRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (root.Length <= 2) throw new InvalidOperationException("项目目录不能是磁盘根目录。");
            return root;
        }

        private static bool SameRoot(string first, string second) { return String.Equals(first, second, StringComparison.OrdinalIgnoreCase); }

        private static bool TryResolveTool(string root, out string python, out string tool)
        {
            python = Path.Combine(root, "_internal", "python_base", "python.exe");
            tool = Path.Combine(root, "tools", "torch-runtime-switch.py");
            foreach (string path in new[] { root, python, tool })
            {
                string parent = path;
                while (SameRoot(parent, root) || parent.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                {
                    if ((Directory.Exists(parent) || File.Exists(parent)) && (File.GetAttributes(parent) & FileAttributes.ReparsePoint) != 0)
                        throw new InvalidOperationException("GPU 环境工具路径不能经过目录链接。");
                    if (SameRoot(parent, root)) break;
                    parent = Path.GetDirectoryName(parent);
                }
            }
            return File.Exists(python) && File.Exists(tool);
        }

        private static string Arguments(string tool, string command) { return "-I -B -X utf8 " + ProcessRunner.Quote(tool) + " " + command; }

        private static IDictionary<string, string> EnvironmentFor(string root)
        {
            return new Dictionary<string, string> { { "PYTHONDONTWRITEBYTECODE", "1" }, { "NN_PROCESS_COUNT", "0" } };
        }

        private Dictionary<string, object> CachedOrUnavailable(string root)
        {
            CachedStatus value;
            Dictionary<string, object> state = cached.TryGetValue(root, out value) ? Copy(value.Value) : Unavailable("正在检查 GPU 运行环境。");
            state["projectDir"] = root;
            return state;
        }

        private void PruneCompletedQueries()
        {
            List<string> complete = new List<string>();
            foreach (KeyValuePair<string, Task<Dictionary<string, object>>> item in pending) if (item.Value.IsCompleted) complete.Add(item.Key);
            foreach (string root in complete) pending.Remove(root);
        }
        private void Store(string root, Dictionary<string, object> state) { state["projectDir"] = root; cached[root] = new CachedStatus { Value = Copy(state), At = now() }; }
        private static Dictionary<string, object> Copy(Dictionary<string, object> state) { return new Dictionary<string, object>(state); }
        private void Notify(Dictionary<string, object> state)
        {
            // A closed UI is not an operation failure. The manager continues
            // and persists its own transaction even if no view receives progress.
            try { if (changed != null) changed(Copy(state)); } catch { }
        }

        internal static Dictionary<string, object> Unavailable(string message, string error = null)
        {
            return new Dictionary<string, object> {
                { "status", "unavailable" }, { "currentRuntime", null }, { "targetRuntime", null },
                { "currentLabel", null }, { "targetLabel", null }, { "gpuNames", new object[0] },
                { "message", message }, { "downloadBytes", 0L }, { "retainedBytes", 0L },
                { "transactionId", null }, { "retainedRuntime", null }, { "downloadRequired", false },
                { "canSwitch", false }, { "canDelete", false }, { "canRecover", false }, { "canRetryCleanup", false },
                { "error", error }, { "retainedRuntimes", new object[0] }
            };
        }
    }
}
