using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using Forms = System.Windows.Forms;

namespace DflPtWebUi.Launcher
{
    internal sealed class HostController : IDisposable
    {
        private readonly SettingsStore settings;
        private readonly LogBuffer logs;
        private readonly ProcessRunner runner;
        private readonly GitService git;
        private readonly TerminalBridgeService terminal;
        private readonly SemaphoreSlim operationGate = new SemaphoreSlim(1, 1);

        public event Action<object> ProgressChanged;

        public HostController(SettingsStore settings, LogBuffer logs)
        {
            this.settings = settings;
            this.logs = logs;
            string knownRoot = ProjectLocator.Resolve(settings.Current);
            if (ProjectLocator.IsProject(knownRoot) || ProjectLocator.IsInstallWorkspace(knownRoot))
                logs.SetDirectory(Path.Combine(ProjectLocator.PrepareInstallWorkspace(knownRoot), "logs"));
            runner = new ProcessRunner(logs);
            git = new GitService(settings, runner, logs);
            terminal = new TerminalBridgeService(logs);
        }

        public LogBuffer Logs
        {
            get { return logs; }
        }

        public async Task<object> GetStateAsync()
        {
            LauncherSettings snapshot = settings.Current;
            string projectRoot = ProjectLocator.Resolve(snapshot);
            bool projectReady = ProjectLocator.IsProject(projectRoot);
            if (projectReady)
            {
                try
                {
                    IList<string> createdWorkspaceDirectories = WorkspaceTemplate.Ensure(projectRoot);
                    if (createdWorkspaceDirectories.Count > 0)
                    {
                        logs.Add(
                            "workspace",
                            "已补建工作区模板目录：" + String.Join("、", createdWorkspaceDirectories) + "。",
                            "success");
                    }
                }
                catch (Exception error)
                {
                    logs.Add("workspace", "工作区模板补建未完成：" + error.Message, "warning");
                }
            }
            string runtimeRoot = projectReady ? projectRoot : PreparedRuntimeStore.GetRoot(projectRoot);

            RuntimeBootstrapResources bootstrapResources = RuntimeBootstrapLocator.Resolve(
                projectReady ? projectRoot : null,
                LauncherPayload.GetPath("bootstrap"));
            string manifestPath = bootstrapResources.ManifestPath;
            RuntimeManifestValidation runtimeValidation = await Task.Run(delegate
            {
                return RuntimeManifestValidator.Validate(runtimeRoot, manifestPath);
            });

            RuntimeComponentValidation nodeRuntime = runtimeValidation.Get("node");
            RuntimeComponentValidation pythonRuntime = runtimeValidation.Get("python");
            RuntimeComponentValidation ffmpegRuntime = runtimeValidation.Get("ffmpeg");

            string webuiBuild = Path.Combine(projectRoot, "webui", "dist", "client", "index.html");
            WebUiDependencyHealth webUiHealth = projectReady && nodeRuntime.Ready
                ? await Task.Run(delegate { return WebUiDependencies.Inspect(projectRoot, null); })
                : new WebUiDependencyHealth { Ready = false, Detail = "等待项目与 Node.js 就绪。" };
            bool buildReady = File.Exists(webuiBuild);
            bool webUiServicesOnline = await AreWebUiServicesOnlineAsync(projectRoot);
            bool webuiRunning = webUiServicesOnline;
            int? webUiPid = webuiRunning ? TryReadManagedWebUiPid(projectRoot) : null;
            GitStatus gitStatus = ProjectLocator.IsGitRepository(projectRoot)
                ? await git.InspectAsync(projectRoot, false)
                : new GitStatus();
            gitStatus.UpdateAvailable = LauncherConstants.OnlineUpdatesEnabled && gitStatus.UpdateAvailable;

            string mirrorLabel = "官方固定来源";
            List<object> runtimeItems = new List<object>();
            runtimeItems.Add(RuntimeItem("project", "PT 项目源码", projectReady, projectReady ? "ME PyTorch 源码已就绪" : "开始安装以获取官方 PT 源码，或选择已有 PT 项目", projectRoot, "官方 GitHub / 本地项目", null));
            runtimeItems.Add(RuntimeItem("node", "Node.js", nodeRuntime.Ready, RuntimeDetail(nodeRuntime), nodeRuntime.TargetPath, mirrorLabel, null));
            runtimeItems.Add(RuntimeItem("python", "Python", pythonRuntime.Ready, RuntimeDetail(pythonRuntime), pythonRuntime.TargetPath, "项目运行环境", null));
            runtimeItems.Add(RuntimeItem("ffmpeg", "FFmpeg", ffmpegRuntime.Ready, RuntimeDetail(ffmpegRuntime), ffmpegRuntime.TargetPath, "项目运行环境", null));
            runtimeItems.Add(RuntimeItem("webui-dependencies", "WebUI 依赖", webUiHealth.Ready, webUiHealth.Detail,
                Path.Combine(projectRoot, "webui", "node_modules"), "本地检查", null));
            runtimeItems.Add(RuntimeItem("webui", "WebUI build", buildReady, buildReady ? "已构建" : "尚未构建", webuiBuild, "本地构建", null));

            bool runtimesReady = runtimeValidation.RequiredComponentsReady;
            bool dependenciesReady = runtimesReady && webUiHealth.Ready && buildReady;
            bool environmentReady = projectReady && dependenciesReady;
            List<object> steps = new List<object>();
            steps.Add(Step("environment", "环境检测", "complete"));
            steps.Add(Step("project", "本地项目", projectReady ? "complete" : "active"));
            steps.Add(Step("dependencies", "安装依赖", dependenciesReady ? "complete" : "active"));
            steps.Add(Step("finish", "准备完成", environmentReady ? "complete" : "upcoming"));

            Dictionary<string, object> result = new Dictionary<string, object>();
            result["mode"] = environmentReady ? "ready" : "install";
            result["environmentStatus"] = environmentReady ? "ready" : "incomplete";
            result["installPath"] = projectRoot;
            result["mirror"] = snapshot.Mirror;
            result["mirrorLabel"] = mirrorLabel;
            result["gitProxyMode"] = snapshot.GitProxyMode;
            result["gitProxy"] = snapshot.GitProxy ?? String.Empty;
            result["gitMirror"] = snapshot.GitMirror ?? String.Empty;
            result["gitNetworkLabel"] = GitNetworkOptions.Describe(snapshot);
            result["steps"] = steps;
            result["runtimeItems"] = runtimeItems;
            result["runtimeManifest"] = runtimeValidation.Loaded ? "validated" : runtimeValidation.Error;
            result["projectDir"] = projectRoot;
            result["projectReady"] = projectReady;
            result["onlineInstallationEnabled"] = LauncherConstants.OnlineInstallationEnabled;
            result["lastCheck"] = snapshot.LastCheck;
            result["terminalUrl"] = terminal.CurrentUrl;
            result["terminalRunning"] = terminal.IsRunning;
            result["webuiRunning"] = webuiRunning;
            result["webuiPid"] = webUiPid.HasValue ? (object)webUiPid.Value : null;
            result["webuiUrl"] = LauncherConstants.WebUiUrl;
            result["settingsPath"] = settings.SettingsPath;
            result["logs"] = GetUiLogs();
            result["git"] = gitStatus;
            result["updateAvailable"] = gitStatus.UpdateAvailable;
            result["release"] = await GetReleaseIdentityAsync(projectRoot);
            result["upgradeTarget"] = ReleaseIdentity.UpgradeTarget();
            result["upgradePending"] = File.Exists(Path.Combine(projectRoot, ".launcher-install", "upgrade-current", "journal.json"));
            return result;
        }

        public object ChooseInstallPath(string requestedPath)
        {
            string selected = requestedPath;
            if (String.IsNullOrWhiteSpace(selected))
            {
                using (Forms.FolderBrowserDialog dialog = new Forms.FolderBrowserDialog())
                {
                    dialog.Description = "选择 DFL-PT-WEBUI 安装位置：空文件夹直接使用；磁盘根目录或非空文件夹使用 DFL-PT-WEBUI 子文件夹。";
                    dialog.ShowNewFolderButton = true;
                    string current = ProjectLocator.Resolve(settings.Current);
                    string parent = Directory.Exists(current) ? current
                        : (Directory.GetParent(current) == null ? current : Directory.GetParent(current).FullName);
                    if (Directory.Exists(parent))
                    {
                        dialog.SelectedPath = parent;
                    }
                    if (dialog.ShowDialog() != Forms.DialogResult.OK)
                    {
                        return new Dictionary<string, object>
                        {
                            { "cancelled", true },
                            { "path", current }
                        };
                    }
                    selected = dialog.SelectedPath;
                }
            }

            string fullPath = ProjectLocator.SelectInstallPath(selected);
            settings.Update(delegate(LauncherSettings value) { value.ProjectRoot = fullPath; });
            logs.Add("launcher", "最终安装目录：" + fullPath + "。", "info");
            return new Dictionary<string, object>
            {
                { "cancelled", false },
                { "path", fullPath },
                { "installPath", fullPath },
                { "projectDir", fullPath }
            };
        }

        public object SetMirror(string mirror)
        {
            string normalized = String.Equals(mirror, "official", StringComparison.OrdinalIgnoreCase)
                ? "official"
                : (String.Equals(mirror, "china", StringComparison.OrdinalIgnoreCase) ? "china" : "auto");
            settings.Update(delegate(LauncherSettings value) { value.Mirror = normalized; });
            logs.Add("launcher", "下载源已切换为：" + GetMirrorLabel(normalized) + "。", "info");
            return new Dictionary<string, object>
            {
                { "mirror", normalized },
                { "mirrorLabel", GetMirrorLabel(normalized) }
            };
        }

        public object ToggleMirror()
        {
            string current = settings.Current.Mirror;
            string next = String.Equals(current, "auto", StringComparison.OrdinalIgnoreCase)
                ? "china"
                : (String.Equals(current, "china", StringComparison.OrdinalIgnoreCase) ? "official" : "auto");
            return SetMirror(next);
        }

        public object SetGitNetwork(string mode, string proxy, string mirror)
        {
            string normalizedMode = GitNetworkOptions.NormalizeProxyMode(mode);
            string normalizedProxy = normalizedMode == "manual"
                ? GitNetworkOptions.NormalizeProxy(proxy)
                : null;
            if (normalizedMode == "manual" && String.IsNullOrWhiteSpace(normalizedProxy))
            {
                throw new InvalidOperationException("手动代理模式需要填写代理地址。");
            }
            string normalizedMirror = GitNetworkOptions.NormalizeMirror(mirror);
            settings.Update(delegate(LauncherSettings value)
            {
                value.GitProxyMode = normalizedMode;
                value.GitProxy = normalizedProxy;
                value.GitMirror = normalizedMirror;
            });
            LauncherSettings snapshot = settings.Current;
            string label = GitNetworkOptions.Describe(snapshot);
            logs.Add("git", "GitHub 网络策略已更新：" + label + "。", "info");
            return new Dictionary<string, object>
            {
                { "gitProxyMode", snapshot.GitProxyMode },
                { "gitProxy", snapshot.GitProxy ?? String.Empty },
                { "gitMirror", snapshot.GitMirror ?? String.Empty },
                { "gitNetworkLabel", label }
            };
        }

        public async Task<object> RunBootstrapAsync(bool repair)
        {
            await operationGate.WaitAsync();
            try
            {
                string projectRoot = ProjectLocator.Resolve(settings.Current);
                ProjectLocator.AssertInstallTarget(projectRoot);
                ProjectLocator.AssertWritableChildPath(projectRoot, ".launcher-install");
                await AssertMaintenanceIdleAsync(projectRoot);
                logs.SetDirectory(Path.Combine(ProjectLocator.PrepareInstallWorkspace(projectRoot), "logs"));
                if (!ProjectLocator.IsProject(projectRoot))
                {
                    if (!LauncherConstants.OnlineInstallationEnabled)
                        throw new InvalidOperationException("请先下载并解压 DFL-PT-WEBUI 源码或便携包。");
                    ReportProgress("project", "正在获取 DFL-PT-WEBUI 官方源码…", "downloading", 1, 4);
                    string sourceInstaller = Path.Combine(LauncherPayload.GetPath("bootstrap"), "install-project.ps1");
                    await RunSetupScriptAsync(sourceInstaller, projectRoot, "源码安装失败");
                    if (!ProjectLocator.IsProject(projectRoot))
                        throw new InvalidOperationException("源码获取后未通过 PT 项目身份检查。");
                    settings.Update(delegate(LauncherSettings value) { value.ProjectRoot = projectRoot; });
                    ReportProgress("project", "PT 项目源码已就绪。", "complete", 2, 4);
                }
                ReportProgress("dependencies", "正在校验本地 PyTorch 运行环境…", "active", 2, 4);
                RuntimeBootstrapResources resources = RuntimeBootstrapLocator.Resolve(projectRoot, LauncherPayload.GetPath("bootstrap"));
                await RunBootstrapScriptAsync(projectRoot, resources, repair);
                ReportProgress("finish", "正在检查 WebUI 构建…", "active", 3, 4);
                await BuildWebUiIfPossibleAsync(projectRoot, repair);
                object state = await GetStateAsync();
                if (!String.Equals(Convert.ToString(((Dictionary<string, object>)state)["environmentStatus"]), "ready", StringComparison.Ordinal))
                    throw new InvalidOperationException("本地健康检查未通过，请查看组件检测结果。");
                try
                {
                    string installed = LauncherInstallation.CopyVerified(
                        System.Reflection.Assembly.GetExecutingAssembly().Location, projectRoot);
                    logs.Add("launcher", "项目启动入口已保存：" + installed, "success");
                }
                catch (IOException error)
                {
                    logs.Add("launcher", "运行环境已就绪，但已有启动器文件已保留：" + error.Message, "warning");
                }
                ReportProgress("finish", "环境检查完成。", "complete", 4, 4);
                return state;
            }
            finally { operationGate.Release(); }
        }

        public async Task<object> CheckUpdatesAsync()
        {
            await operationGate.WaitAsync();
            try
            {
                string root = RequireGitProjectRoot();
                ReportProgress("updates", "正在检查 GitHub 更新…", "running", 0, 1);
                GitStatus result = await git.CheckUpdatesAsync(root);
                ReportProgress("updates", result.UpdateAvailable ? "发现项目更新。" : "当前已是最新版本。", "complete", 1, 1);
                return result;
            }
            finally
            {
                operationGate.Release();
            }
        }

        public async Task<object> ApplyUpdateAsync()
        {
            await operationGate.WaitAsync();
            try
            {
                string root = RequireProjectRoot();
                await AssertMaintenanceIdleAsync(root);
                ReportProgress("updates", "正在备份并升级到此启动器固定的应用版本…", "running", 0, 2);
                Exception updateFailure = null;
                try {
                    await RunUpgradeScriptAsync(root, "apply");
                    await BuildWebUiIfPossibleAsync(root, true);
                    object state = await GetStateAsync();
                    if (!String.Equals(Convert.ToString(((Dictionary<string, object>)state)["environmentStatus"]), "ready", StringComparison.Ordinal))
                        throw new InvalidOperationException("升级后的环境检查未通过。");
                    await RunUpgradeScriptAsync(root, "complete");
                } catch (Exception error) { updateFailure = error; }
                if (updateFailure != null) {
                    if (File.Exists(Path.Combine(root, ".launcher-install", "upgrade-current", "journal.json"))) {
                        await RunUpgradeScriptAsync(root, "rollback");
                        logs.Add("updates", "升级失败，已经恢复升级前的源码和构建。", "warning");
                    }
                    throw updateFailure;
                }
                ReportProgress("updates", "更新完成。", "complete", 2, 2);
                return await GetStateAsync();
            }
            finally
            {
                operationGate.Release();
            }
        }

        public async Task<object> StartWebUiAsync()
        {
            await operationGate.WaitAsync();
            try
            {
                string root = RequireProjectRoot();
                string node = Path.Combine(root, "_internal", "node", "bin", "node.exe");
                string manager = Path.Combine(root, "webui", "scripts", "local-manager.mjs");
                if (!File.Exists(node) || !File.Exists(manager))
                {
                    throw new InvalidOperationException("WebUI 运行环境不完整，请先执行首次设置或修复依赖。");
                }

                IDictionary<string, string> environment = PortableNodeEnvironment.Ensure(
                    DflEnvironment.Load(root, logs),
                    node);
                environment["DFL_UI_LANG"] = "zh";
                CommandResult result = await runner.RunAsync(node, ProcessRunner.Quote(manager) + " start", root, environment, "webui");
                EnsureSuccess(result, "WebUI 启动失败");
                OpenUrl(LauncherConstants.WebUiUrl);
                return await GetStateAsync();
            }
            finally
            {
                operationGate.Release();
            }
        }

        public async Task<object> StopWebUiAsync()
        {
            await operationGate.WaitAsync();
            try
            {
                string root = RequireProjectRoot();
                string node = Path.Combine(root, "_internal", "node", "bin", "node.exe");
                string manager = Path.Combine(root, "webui", "scripts", "local-manager.mjs");
                if (!File.Exists(node) || !File.Exists(manager))
                {
                    return await GetStateAsync();
                }
                IDictionary<string, string> environment = PortableNodeEnvironment.Ensure(
                    DflEnvironment.Load(root, logs),
                    node);
                environment["DFL_UI_LANG"] = "zh";
                CommandResult result = await runner.RunAsync(node, ProcessRunner.Quote(manager) + " stop", root, environment, "webui");
                EnsureSuccess(result, "WebUI 停止失败");
                return await GetStateAsync();
            }
            finally
            {
                operationGate.Release();
            }
        }

        public async Task<bool> HasRunningWebUiAsync()
        {
            string projectRoot = ProjectLocator.Resolve(settings.Current);
            int? managedPid = ProjectLocator.IsProject(projectRoot)
                ? TryReadManagedWebUiPid(projectRoot)
                : null;
            return managedPid.HasValue || await IsOwnedRuntimeOnlineAsync(projectRoot);
        }

        private async Task AssertMaintenanceIdleAsync(string root)
        {
            if (terminal.IsRunning || await HasRunningWebUiAsync())
                throw new InvalidOperationException("项目或 WebUI 正在运行。请先结束任务、终端和 WebUI，再修复或升级。");
            await RunUpgradeScriptAsync(root, "guard");
            // Inspect executable identity without exposing command lines, media
            // paths or secrets. This also covers detached training processes.
            foreach (Process process in Process.GetProcesses()) {
                try {
                    string name = process.ProcessName;
                    if (name != "python" && name != "pythonw" && name != "node" && name != "ffmpeg") continue;
                    string executable = process.MainModule.FileName;
                    if (executable.StartsWith(Path.GetFullPath(root).TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException("检测到项目运行中的任务。请先保存并停止任务，再修复或升级。");
                } catch (System.ComponentModel.Win32Exception) { }
                  catch (InvalidOperationException error) { if (error.Message.StartsWith("检测到")) throw; }
                finally { process.Dispose(); }
            }
        }

        private async Task<object> GetReleaseIdentityAsync(string root)
        {
            string node = Path.Combine(root, "_internal", "node", "bin", "node.exe");
            string module = Path.Combine(LauncherPayload.GetPath("bootstrap"), "installation.mjs");
            if (File.Exists(node) && File.Exists(module)) {
                try {
                    CommandResult result = await runner.RunAsync(node, ProcessRunner.Quote(module) + " --root " + ProcessRunner.Quote(root), root, null, "release");
                    if (result.Success) {
                        Dictionary<string, object> identity = new JavaScriptSerializer().DeserializeObject(result.StandardOutput) as Dictionary<string, object>;
                        Dictionary<string, object> source = identity["source"] as Dictionary<string, object>;
                        identity["applicationVersion"] = identity["appVersion"];
                        identity["launcherVersion"] = "0.1.3-preview";
                        identity["sourceCommit"] = source["revision"] ?? "unverified";
                        identity["archiveSha256"] = source["archiveSha256"];
                        identity["sourceStatus"] = source["status"];
                        return identity;
                    }
                } catch { }
            }
            return ReleaseIdentity.Get(root);
        }

        private async Task RunUpgradeScriptAsync(string root, string action)
        {
            string bootstrap = LauncherPayload.GetPath("bootstrap");
            string script = Path.Combine(bootstrap, "upgrade-project.ps1");
            string arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + ProcessRunner.Quote(script)
                + " -ProjectRoot " + ProcessRunner.Quote(root) + " -Action " + action
                + " -SourcePinPath " + ProcessRunner.Quote(Path.Combine(bootstrap, "source-pin.json"));
            CommandResult result = await runner.RunAsync("powershell.exe", arguments, bootstrap, null, "updates");
            EnsureSuccess(result, action == "rollback" ? "自动回退失败，请保留安装缓存并使用回退入口" : "固定版本升级未完成");
        }

        public async Task<object> RollbackUpdateAsync()
        {
            await operationGate.WaitAsync();
            try { string root = RequireProjectRoot(); await AssertMaintenanceIdleAsync(root); await RunUpgradeScriptAsync(root, "rollback"); return await GetStateAsync(); }
            finally { operationGate.Release(); }
        }

        public async Task<object> PrepareFeedbackAsync(string step)
        {
            string[] allowed = { "install", "start", "upgrade", "repair", "project", "import", "extract", "train", "image", "export", "unknown" };
            if (Array.IndexOf(allowed, step) < 0) step = "unknown";
            string root = ProjectLocator.Resolve(settings.Current);
            string node = Path.Combine(root, "_internal", "node", "bin", "node.exe");
            string feedback = Path.Combine(LauncherPayload.GetPath("bootstrap"), "feedback.mjs");
            if (File.Exists(node) && File.Exists(feedback)) {
                CommandResult result = await runner.RunAsync(node, ProcessRunner.Quote(feedback) + " --root " + ProcessRunner.Quote(root) + " --step " + step, root, null, "feedback");
                EnsureSuccess(result, "反馈信息准备失败");
                return new JavaScriptSerializer().DeserializeObject(result.StandardOutput);
            }
            Dictionary<string, object> template = new JavaScriptSerializer().DeserializeObject(File.ReadAllText(Path.Combine(LauncherPayload.GetPath("bootstrap"), "feedback-template.json"))) as Dictionary<string, object>;
            return new Dictionary<string, object> {
                { "preview", Convert.ToString(template["offlinePreview"]) + "\n\n失败步骤：" + step + "\n启动器版本：0.1.3-preview\n应用版本/安装来源/源码快照：尚未验证\n" },
                { "issueUrl", template["issueUrl"] }, { "release", ReleaseIdentity.Get(root) }
            };
        }

        public async Task<object> StartTerminalAsync()
        {
            string url = await terminal.StartAsync(RequireProjectRoot());

            return new Dictionary<string, object>
            {
                { "terminalUrl", url },
                { "protocol", "dflsn-terminal-v1" }
            };
        }

        public Task<object> OpenLegacyAsync()
        {
            return StartTerminalAsync();
        }

        public object OpenExternal(string url)
        {
            OpenUrl(url);
            return new Dictionary<string, object> { { "ok", true } };
        }

        public LogSnapshot PollLogs(long sequence, int limit)
        {
            return logs.ReadSince(sequence, limit);
        }

        public void Dispose()
        {
            terminal.Dispose();
            operationGate.Dispose();
        }

        private async Task<Exception> CloneProjectWithRetryAsync(string projectRoot, CancellationToken cancellationToken)
        {
            Exception lastFailure = null;
            int attempt = 0;
            while (!cancellationToken.IsCancellationRequested && !ProjectLocator.IsProject(projectRoot))
            {
                attempt++;
                try
                {
                    logs.Add("git", "GitHub 后台克隆第 " + attempt + " 次尝试开始。", "info");
                    ReportProgress("project", "后台获取项目 · 第 " + attempt + " 次尝试", "downloading", 1, 4);
                    await git.CloneAsync(projectRoot, attempt);
                    logs.Add("git", "GitHub 项目已在后台获取完成。", "success");
                    ReportProgress("project", "项目源码获取完成。", "complete", 2, 4);
                    return null;
                }
                catch (Exception error)
                {
                    lastFailure = error;
                    logs.Add("git", "第 " + attempt + " 次获取失败；其他依赖继续安装，60 秒后自动重试：" + error.Message, "warning");
                    ReportProgress("project", "网络不稳定；60 秒后自动重试", "checking", 1, 4);
                }
                if (cancellationToken.IsCancellationRequested) break;
                try
                {
                    await Task.Delay(TimeSpan.FromMinutes(1), cancellationToken);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
            }
            return lastFailure;
        }
        private async Task EnsurePortableGitAsync()
        {
            if (!String.IsNullOrWhiteSpace(git.LocateGit()))
            {
                return;
            }

            string bootstrapRoot = LauncherPayload.GetPath("bootstrap");
            string bootstrapScript = Path.Combine(bootstrapRoot, "bootstrap.ps1");
            string manifest = Path.Combine(bootstrapRoot, "runtime-manifest.json");
            if (!File.Exists(bootstrapScript) || !File.Exists(manifest))
            {
                throw new FileNotFoundException("未找到便携 Git 引导文件；请重新解压完整启动器。", bootstrapScript);
            }

            string toolsRoot = Path.Combine(LauncherConstants.SettingsDirectory, "bootstrap-tools");
            string arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                + ProcessRunner.Quote(bootstrapScript)
                + " -GitOnly -ProjectRoot " + ProcessRunner.Quote(toolsRoot)
                + " -ManifestPath " + ProcessRunner.Quote(manifest)
                + " -Mirror " + ProcessRunner.Quote(settings.Current.Mirror);
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            string failureMessage = null;
            logs.Add("bootstrap", "未检测到 Git，正在准备便携 MinGit…", "info");
            CommandResult result = await runner.RunAsync(
                "powershell.exe",
                arguments,
                bootstrapRoot,
                MirrorEnvironment(null),
                "bootstrap",
                delegate(string line)
                {
                    string candidate = TryGetBootstrapFailureMessage(line, serializer);
                    if (!String.IsNullOrWhiteSpace(candidate)) { failureMessage = candidate; }
                    return TryForwardBootstrapProgress(line, serializer);
                });
            EnsureBootstrapSuccess(result, "便携 Git 安装失败", failureMessage);
            if (String.IsNullOrWhiteSpace(git.LocateGit()))
            {
                throw new FileNotFoundException("便携 Git 安装完成，但未找到 git.exe。", Path.Combine(toolsRoot, "git", "cmd", "git.exe"));
            }
        }

        private async Task RunBootstrapScriptAsync(
            string projectRoot,
            RuntimeBootstrapResources resources,
            bool repair)
        {
            string arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                + ProcessRunner.Quote(resources.ScriptPath)
                + " -ProjectRoot " + ProcessRunner.Quote(projectRoot)
                + " -ManifestPath " + ProcessRunner.Quote(resources.ManifestPath)
                + " -Mirror official";
            if (repair)
            {
                arguments += " -Repair";
            }
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            string failureMessage = null;
            CommandResult result = await runner.RunAsync(
                "powershell.exe",
                arguments,
                projectRoot,
                MirrorEnvironment(null),
                "bootstrap",
                delegate(string line)
                {
                    string candidate = TryGetBootstrapFailureMessage(line, serializer);
                    if (!String.IsNullOrWhiteSpace(candidate)) { failureMessage = candidate; }
                    return TryForwardBootstrapProgress(line, serializer);
                });
            EnsureBootstrapSuccess(result, "依赖安装失败", failureMessage);
        }

        private async Task RunSetupScriptAsync(string script, string projectRoot, string failureLabel)
        {
            if (!File.Exists(script)) throw new FileNotFoundException("启动器内嵌安装资源不完整。", script);
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            string failureMessage = null;
            CommandResult result = await runner.RunAsync(
                "powershell.exe",
                "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                    + ProcessRunner.Quote(script) + " -ProjectRoot " + ProcessRunner.Quote(projectRoot),
                projectRoot,
                MirrorEnvironment(null),
                "bootstrap",
                delegate(string line)
                {
                    string candidate = TryGetBootstrapFailureMessage(line, serializer);
                    if (!String.IsNullOrWhiteSpace(candidate)) failureMessage = candidate;
                    return TryForwardBootstrapProgress(line, serializer);
                });
            EnsureBootstrapSuccess(result, failureLabel, failureMessage);
        }

        private async Task BuildWebUiIfPossibleAsync(string projectRoot, bool force)
        {
            ProjectLocator.AssertWritableChildPath(projectRoot, "webui/node_modules");
            ProjectLocator.AssertWritableChildPath(projectRoot, "webui/dist");
            string corepackCache = ProjectLocator.AssertWritableChildPath(projectRoot, ".launcher-install/source/corepack");
            string node = Path.Combine(projectRoot, "_internal", "node", "bin", "node.exe");
            string corepack = Path.Combine(projectRoot, "_internal", "node", "bin", "node_modules", "corepack", "dist", "corepack.js");
            string webuiRoot = Path.Combine(projectRoot, "webui");
            string vite = Path.Combine(webuiRoot, "node_modules", "vite", "bin", "vite.js");
            string index = Path.Combine(webuiRoot, "dist", "client", "index.html");
            if (!File.Exists(node))
            {
                throw new FileNotFoundException("WebUI 构建需要项目内的便携 Node.js。", node);
            }

            IDictionary<string, string> environment = PortableNodeEnvironment.Ensure(
                DflEnvironment.Load(projectRoot, logs),
                node);
            environment = MirrorEnvironment(environment);
            environment["COREPACK_HOME"] = corepackCache;
            environment["COREPACK_ENABLE_DOWNLOAD_PROMPT"] = "0";
            bool dependencyTreePresent = Directory.Exists(Path.Combine(webuiRoot, "node_modules"));
            bool dependencyFilesPresent = WebUiDependencies.EntryPointsPresent(projectRoot);
            bool dependenciesLoad = dependencyFilesPresent
                && await CanLoadWebUiDependenciesAsync(webuiRoot, environment);
            if (!force && dependenciesLoad && File.Exists(index))
            {
                logs.Add("bootstrap", "WebUI 依赖加载通过且构建已存在，无需重复安装。", "success");
                return;
            }

            if (force || !dependenciesLoad)
            {
                if (!File.Exists(corepack))
                {
                    throw new FileNotFoundException("缺少 Node.js Corepack，无法安装 WebUI 依赖。", corepack);
                }
                logs.Add("bootstrap", "正在按 pnpm 锁文件安装 WebUI 依赖…", "info");
                string installArguments = ProcessRunner.Quote(corepack)
                    + " pnpm install --frozen-lockfile --prefer-offline";
                if (dependencyTreePresent && !dependenciesLoad)
                {
                    installArguments += " --force";
                    logs.Add("bootstrap", "检测到依赖残留或正在修复，将强制重建原生 Node.js 依赖。", "warning");
                }
                CommandResult install = await runner.RunAsync(
                    node,
                    installArguments,
                    webuiRoot,
                    environment,
                    "bootstrap");
                EnsureSuccess(install, "WebUI 依赖安装失败");

                if (!await CanLoadWebUiDependenciesAsync(webuiRoot, environment))
                {
                    throw new InvalidOperationException(
                        "WebUI 依赖安装后加载验证失败；具体模块及异常见上方检测输出和本地错误日志。");
                }
            }

            // A repaired dependency tree also invalidates an existing build.
            if (force || !dependenciesLoad || !File.Exists(index))
            {
                CommandResult build = await runner.RunAsync(
                    node,
                    ProcessRunner.Quote(vite) + " build --configLoader runner",
                    webuiRoot,
                    environment,
                    "bootstrap");
                EnsureSuccess(build, "WebUI 构建失败");
            }
        }

        private async Task<bool> CanLoadWebUiDependenciesAsync(
            string webuiRoot,
            IDictionary<string, string> environment)
        {
            WebUiDependencyHealth validation = await Task.Run(delegate
            {
                return WebUiDependencies.Inspect(Directory.GetParent(webuiRoot).FullName, environment);
            });
            if (!validation.Ready)
            {
                logs.Add("bootstrap", validation.Detail, "error");
                logs.Add("bootstrap", "检测到未完成的 Node.js 依赖安装，将自动修复。", "warning");
            }
            return validation.Ready;
        }

        private IDictionary<string, string> MirrorEnvironment(IDictionary<string, string> source)
        {
            Dictionary<string, string> environment = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            if (source != null)
            {
                foreach (KeyValuePair<string, string> item in source)
                {
                    environment[item.Key] = item.Value;
                }
            }

            // Match the fixed registries used by the PT source installer.
            environment["DFL_MIRROR"] = "official";
            environment["NPM_CONFIG_REGISTRY"] = "https://registry.npmjs.org";
            environment["COREPACK_NPM_REGISTRY"] = "https://registry.npmjs.org";
            environment["PIP_INDEX_URL"] = "https://pypi.org/simple";
            environment["PIP_TRUSTED_HOST"] = String.Empty;
            return environment;
        }

        private string RequireProjectRoot()
        {
            string root = ProjectLocator.Resolve(settings.Current);
            if (!ProjectLocator.IsProject(root))
            {
                throw new InvalidOperationException("项目尚未安装，请先完成首次设置。");
            }
            return root;
        }

        private string RequireGitProjectRoot()
        {
            if (!LauncherConstants.OnlineUpdatesEnabled)
                throw new InvalidOperationException("预览版的 Git 更新未启用，请通过发行页手动升级。");
            string root = RequireProjectRoot();
            if (!ProjectLocator.IsGitRepository(root))
                throw new InvalidOperationException("当前项目来自源码或便携发行包，没有 Git 元数据，不能执行 Git 更新。");
            return root;
        }

        private static async Task<string> ReadVersionQuietlyAsync(string executable, string arguments)
        {
            if (!File.Exists(executable))
            {
                return null;
            }
            return await Task.Run(delegate
            {
                try
                {
                    ProcessStartInfo startInfo = new ProcessStartInfo(executable, arguments);
                    startInfo.UseShellExecute = false;
                    startInfo.CreateNoWindow = true;
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    using (Process process = Process.Start(startInfo))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit(5000);
                        string value = String.IsNullOrWhiteSpace(output) ? error : output;
                        return String.IsNullOrWhiteSpace(value) ? null : value.Trim();
                    }
                }
                catch
                {
                    return null;
                }
            });
        }

        private static int CountDlls(string directory)
        {
            try
            {
                return Directory.Exists(directory)
                    ? Directory.GetFiles(directory, "*.dll", SearchOption.AllDirectories).Length
                    : 0;
            }
            catch
            {
                return 0;
            }
        }

        private static string FindFile(string directory, string fileName)
        {
            try
            {
                if (!Directory.Exists(directory))
                {
                    return null;
                }
                string[] matches = Directory.GetFiles(directory, fileName, SearchOption.AllDirectories);
                return matches.Length == 0 ? null : matches[0];
            }
            catch
            {
                return null;
            }
        }

        private static async Task<bool> IsUrlOnlineAsync(string url)
        {
            return await Task.Run(delegate
            {
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
                    request.Method = "GET";
                    request.Timeout = 800;
                    request.ReadWriteTimeout = 800;
                    request.Proxy = null;
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    {
                        return (int)response.StatusCode >= 200 && (int)response.StatusCode < 500;
                    }
                }
                catch
                {
                    return false;
                }
            });
        }

        private static async Task<bool> IsOwnedRuntimeOnlineAsync(string projectRoot)
        {
            return await Task.Run(delegate
            {
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(LauncherConstants.RuntimeHealthUrl);
                    request.Timeout = 1500;
                    request.ReadWriteTimeout = 1500;
                    request.AllowAutoRedirect = false;
                    request.Proxy = null;
                    Stopwatch elapsed = Stopwatch.StartNew();
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    using (Stream stream = response.GetResponseStream())
                    using (MemoryStream body = new MemoryStream())
                    {
                        if (response.StatusCode != HttpStatusCode.OK) return false;
                        byte[] buffer = new byte[4096];
                        int count;
                        while ((count = stream.Read(buffer, 0, buffer.Length)) > 0)
                        {
                            if (body.Length + count > RuntimeServiceIdentity.MaxHealthBytes
                                || elapsed.ElapsedMilliseconds > 2500) return false;
                            body.Write(buffer, 0, count);
                        }
                        return RuntimeServiceIdentity.Matches(
                            System.Text.Encoding.UTF8.GetString(body.ToArray()), projectRoot);
                    }
                }
                catch { return false; }
            });
        }

        private static async Task<bool> AreWebUiServicesOnlineAsync(string projectRoot)
        {
            Task<bool> web = IsUrlOnlineAsync(LauncherConstants.WebUiUrl);
            Task<bool> runtime = IsOwnedRuntimeOnlineAsync(projectRoot);
            await Task.WhenAll(web, runtime);
            return web.Result && runtime.Result;
        }

        private static int? TryReadManagedWebUiPid(string projectRoot)
        {
            try
            {
                string runtimeRoot = Path.Combine(projectRoot, "webui", ".runtime");
                string pidPath = Path.Combine(runtimeRoot, "local-manager.pid");
                string statusPath = Path.Combine(runtimeRoot, "local-manager.status.json");
                int pid;
                if (!File.Exists(pidPath)
                    || !Int32.TryParse(File.ReadAllText(pidPath).Trim(), out pid)
                    || pid <= 0
                    || !File.Exists(statusPath))
                {
                    return null;
                }

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                Dictionary<string, object> status = serializer.Deserialize<Dictionary<string, object>>(
                    File.ReadAllText(statusPath));
                if (status == null || !status.ContainsKey("projectRoot")
                    || !RuntimeServiceIdentity.SameProjectPath(Convert.ToString(status["projectRoot"]), projectRoot))
                    return null;
                int supervisorPid = status != null && status.ContainsKey("supervisorPid")
                    ? Convert.ToInt32(status["supervisorPid"])
                    : 0;
                string state = status != null && status.ContainsKey("state")
                    ? Convert.ToString(status["state"])
                    : null;
                if (supervisorPid != pid || !String.Equals(state, "running", StringComparison.OrdinalIgnoreCase))
                {
                    return null;
                }

                using (Process process = Process.GetProcessById(pid))
                {
                    if (process.HasExited || !RuntimeServiceIdentity.SameProjectPath(
                        process.MainModule.FileName, Path.Combine(projectRoot, "_internal", "node", "bin", "node.exe")))
                    {
                        return null;
                    }
                }
                return pid;
            }
            catch
            {
                return null;
            }
        }

        private static string RuntimeDetail(RuntimeComponentValidation runtime)
        {
            if (runtime == null) return "运行时清单未提供校验结果。";
            if (!runtime.Ready) return String.IsNullOrWhiteSpace(runtime.Reason) ? "未安装" : runtime.Reason;
            return String.IsNullOrWhiteSpace(runtime.Version) ? "已通过完整校验" : runtime.Version + " · 已校验";
        }

        private static object RuntimeItem(string id, string label, bool ready, string detail, string path, string source, string link)
        {
            Dictionary<string, object> value = new Dictionary<string, object>
            {
                { "id", id },
                { "label", label },
                { "status", ready ? "installed" : "waiting" },
                { "ready", ready },
                { "detail", detail },
                { "path", path },
                { "source", source }
            };
            if (!String.IsNullOrWhiteSpace(link))
            {
                value["link"] = link;
            }
            return value;
        }

        private static object Step(string id, string label, string status)
        {
            return new Dictionary<string, object>
            {
                { "id", id },
                { "label", label },
                { "status", status }
            };
        }

        private IList<object> GetUiLogs()
        {
            IList<LogEntry> entries = logs.ReadSince(0, 200).Entries;
            List<object> result = new List<object>();
            for (int index = 0; index < entries.Count; index++)
            {
                LogEntry entry = entries[index];
                DateTime timestamp;
                string time = DateTime.TryParse(entry.Timestamp, out timestamp)
                    ? timestamp.ToLocalTime().ToString("HH:mm:ss")
                    : null;
                result.Add(new Dictionary<string, object>
                {
                    { "time", time },
                    { "text", entry.Line },
                    { "line", entry.Line },
                    { "level", entry.Level },
                    { "channel", entry.Channel },
                    { "sequence", entry.Sequence }
                });
            }
            return result;
        }

        private bool TryForwardBootstrapProgress(string line, JavaScriptSerializer serializer)
        {
            try
            {
                Dictionary<string, object> raw = serializer.DeserializeObject(line) as Dictionary<string, object>;
                if (raw == null)
                {
                    return false;
                }
                string message = raw.ContainsKey("message") ? Convert.ToString(raw["message"]) : null;
                string status = raw.ContainsKey("status") ? Convert.ToString(raw["status"]) : null;
                string normalizedStatus = NormalizeBootstrapStatus(status);
                if (!String.IsNullOrWhiteSpace(message))
                {
                    bool failed = String.Equals(normalizedStatus, "error", StringComparison.OrdinalIgnoreCase);
                    logs.Add("bootstrap", message, failed ? "error" : "info");
                }
                Dictionary<string, object> value = new Dictionary<string, object>(raw);
                if (!String.IsNullOrWhiteSpace(normalizedStatus))
                {
                    value["status"] = normalizedStatus;
                }
                string id = raw.ContainsKey("id") ? Convert.ToString(raw["id"]) : null;
                if (String.IsNullOrWhiteSpace(id) && raw.ContainsKey("stage"))
                {
                    id = Convert.ToString(raw["stage"]);
                    value["id"] = id;
                }
                if (String.IsNullOrWhiteSpace(id))
                {
                    return true;
                }
                if (!value.ContainsKey("label") && raw.ContainsKey("message"))
                {
                    value["label"] = Convert.ToString(raw["message"]);
                }
                Action<object> handler = ProgressChanged;
                if (handler != null)
                {
                    handler(value);
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static string TryGetBootstrapFailureMessage(string line, JavaScriptSerializer serializer)
        {
            try
            {
                Dictionary<string, object> raw = serializer.DeserializeObject(line) as Dictionary<string, object>;
                if (raw == null || !raw.ContainsKey("status") || !raw.ContainsKey("message"))
                {
                    return null;
                }
                string status = Convert.ToString(raw["status"]);
                string message = Convert.ToString(raw["message"]);
                return String.Equals(status, "failed", StringComparison.OrdinalIgnoreCase)
                    && !String.IsNullOrWhiteSpace(message)
                    ? message
                    : null;
            }
            catch
            {
                return null;
            }
        }
        private static string NormalizeBootstrapStatus(string status)
        {
            string value = String.IsNullOrWhiteSpace(status) ? String.Empty : status.Trim().ToLowerInvariant();
            switch (value)
            {
                case "ready":
                case "cached":
                case "verified":
                case "resumed":
                case "configured":
                case "space-ok":
                case "selected":
                case "staged":
                case "preserved":
                case "backup-retained":
                    return "installed";
                case "checking":
                case "verifying":
                case "verifying-cache":
                case "probing":
                case "source-failed":
                case "http-failed":
                    return "checking";
                case "downloading":
                case "downloaded":
                case "connecting":
                case "restart":
                    return "downloading";
                case "extracting":
                case "repairing":
                    return "installing";
                case "planned":
                case "rolled-back":
                    return "waiting";
                case "failed":
                case "unavailable":
                case "rollback-failed":
                case "cache-rejected":
                case "hash-mismatch":
                    return "error";
                case "complete":
                    return "complete";
                default:
                    return value.StartsWith("bits-", StringComparison.Ordinal) ? "downloading" : value;
            }
        }

        private static string GetMirrorLabel(string mirror)
        {
            if (String.Equals(mirror, "china", StringComparison.OrdinalIgnoreCase))
            {
                return "国内镜像";
            }
            if (String.Equals(mirror, "official", StringComparison.OrdinalIgnoreCase))
            {
                return "官方源";
            }
            return "自动选择";
        }

        private void ReportProgress(string id, string label, string status, int current, int total)
        {
            Dictionary<string, object> value = new Dictionary<string, object>
            {
                { "id", id },
                { "label", label },
                { "status", status },
                { "current", current },
                { "total", total },
                { "percent", total <= 0 ? 0 : (int)Math.Round((double)current * 100.0 / total) }
            };
            Action<object> handler = ProgressChanged;
            if (handler != null)
            {
                handler(value);
            }
        }

        private static void OpenUrl(string url)
        {
            Uri uri;
            if (!Uri.TryCreate(url, UriKind.Absolute, out uri)
                || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            {
                throw new InvalidOperationException("只允许打开 HTTP 或 HTTPS 地址。");
            }
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = uri.AbsoluteUri;
            startInfo.UseShellExecute = true;
            Process.Start(startInfo);
        }

        private static void EnsureSuccess(CommandResult result, string prefix)
        {
            if (result.Success)
            {
                return;
            }
            string detail = String.IsNullOrWhiteSpace(result.StandardError) ? result.StandardOutput : result.StandardError;
            throw new InvalidOperationException(prefix + "（退出码 " + result.ExitCode + "）" + (String.IsNullOrWhiteSpace(detail) ? String.Empty : "：" + detail));
        }
        private static void EnsureBootstrapSuccess(CommandResult result, string prefix, string failureMessage)
        {
            if (result.Success)
            {
                return;
            }
            if (!String.IsNullOrWhiteSpace(failureMessage))
            {
                throw new InvalidOperationException(prefix + "（退出码 " + result.ExitCode + "）：" + failureMessage);
            }
            EnsureSuccess(result, prefix);
        }
    }
}
