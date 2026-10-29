// Compiled only by cuda-runtime-host.Tests.ps1 with the actual service/RPC sources.
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace Microsoft.Web.WebView2.Core
{
    internal sealed class CoreWebView2WebMessageReceivedEventArgs : EventArgs
    {
        public string Source { get; set; }
        public string WebMessageAsJson { get; set; }
        public string TryGetWebMessageAsString() { return WebMessageAsJson; }
    }
    internal sealed class CoreWebView2
    {
        public event EventHandler<CoreWebView2WebMessageReceivedEventArgs> WebMessageReceived;
        public readonly List<string> Posted = new List<string>();
        public void PostWebMessageAsJson(string value) { Posted.Add(value); }
        public void Raise(string value) { if (WebMessageReceived != null) WebMessageReceived(this, new CoreWebView2WebMessageReceivedEventArgs { Source = "https://launcher.local/", WebMessageAsJson = value }); }
    }
}

namespace CudaRuntimeHostTests
{
    internal sealed class LogEntry { public string Channel { get; set; } public string Line { get; set; } public string Level { get; set; } public string Timestamp { get; set; } public long Sequence { get; set; } }
    internal sealed class LogBuffer
    {
        public event Action<LogEntry> EntryAdded;
        public void Add(string channel, string line, string level) { if (EntryAdded != null) EntryAdded(new LogEntry { Channel=channel, Line=line, Level=level }); }
    }
    internal static class LauncherConstants { public const string VirtualHost = "launcher.local"; }
    internal sealed class Dispatcher { public bool RejectPost { get; set; } public void BeginInvoke(Action action) { if(RejectPost)throw new InvalidOperationException("synthetic dispatcher closed");action(); } }
    internal sealed class MainWindow
    {
        public readonly Dispatcher Dispatcher = new Dispatcher(); public bool IsWindowMaximized;
        public void MinimizeWindow() { } public void ToggleMaximizeWindow() { IsWindowMaximized=!IsWindowMaximized; }
        public void CloseWindow() { } public void DragWindow() { }
    }
    internal static class ProjectLocator
    {
        public static void AssertWritableChildPath(string root, string relative)
        {
            if (relative != ".runtime") throw new InvalidOperationException("unexpected managed path");
        }
    }

    internal sealed class Scenario
    {
        public readonly List<string> Commands = new List<string>();
        public readonly List<Dictionary<string,object>> Events = new List<Dictionary<string,object>>();
        public readonly CudaRuntimeService Service;
        public int StatusCalls, MutationCalls;
        public DateTime Clock = new DateTime(2026,10,29,1,0,0,DateTimeKind.Utc);
        public string Status = State("change-required", false, false);
        public bool HoldStatus, HoldMutation, FailMutation, MalformedStatus;
        public readonly TaskCompletionSource<CommandResult> PendingStatus = new TaskCompletionSource<CommandResult>();
        public readonly TaskCompletionSource<CommandResult> PendingMutation = new TaskCompletionSource<CommandResult>();

        public Scenario()
        {
            Service = new CudaRuntimeService(Execute, delegate(object state) { Events.Add(new Dictionary<string,object>((Dictionary<string,object>)state)); }, delegate { return Clock; });
        }
        public static string State(string status, bool canDelete, bool canRecover)
        {
            return new JavaScriptSerializer().Serialize(new Dictionary<string,object> {
                {"status",status}, {"currentRuntime", "cu128"}, {"targetRuntime","cu126"},
                {"currentLabel","CUDA 12.8"}, {"targetLabel","CUDA 12.6"},
                {"canDelete",canDelete}, {"canRecover",canRecover}, {"canSwitch",true},
                {"transactionId","0123456789abcdef0123456789abcdef"}, {"message","synthetic state"}
            });
        }
        public Task<CommandResult> Execute(string executable,string arguments,string cwd,IDictionary<string,string> env,string channel,Func<string,bool> line)
        {
            Commands.Add(arguments);
            if (!arguments.StartsWith("-I -B -X utf8 ")) throw new InvalidOperationException("non-isolated interpreter");
            if (!executable.EndsWith("_internal\\python_base\\python.exe",StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("non-local interpreter");
            if (arguments.EndsWith(" status")) {
                StatusCalls++;
                if (HoldStatus) { HoldStatus=false; return PendingStatus.Task; }
                return Task.FromResult(new CommandResult { ExitCode=0,StandardOutput=MalformedStatus ? "not json" : Status });
            }
            MutationCalls++;
            if (line != null) {
                line("DFL_CUDA_EVENT|{\"status\":\"downloading\",\"message\":\"正在下载\",\"downloadedBytes\":1024,\"canDelete\":true}");
                line("DFL_CUDA_EVENT|{\"phase\":\"activating\",\"message\":\"正在切换\"}");
            }
            if (HoldMutation) return PendingMutation.Task;
            if (FailMutation) {
                Status=State("error",false,true);
                return Task.FromResult(new CommandResult { ExitCode=7, StandardOutput="",StandardError="synthetic install failure" });
            }
            Status = arguments.Contains(" prepare ") ? State("cleanup-choice",true,false)
                : arguments.Contains(" keep-retained ") ? State("retained",true,false) : State("compatible",false,false);
            return Task.FromResult(new CommandResult { ExitCode=0,StandardOutput="DFL_CUDA_EVENT|{\"status\":\"switching\"}\n"+Status });
        }
    }

    // The marked methods are inserted verbatim from HostController.cs by the
    // test script. Their guard/dispatch code, not a rewritten approximation,
    // runs with process and filesystem boundaries supplied by this fixture.
    internal sealed class HostController
    {
        private readonly CudaRuntimeService cudaRuntime;
        private readonly SemaphoreSlim operationGate = new SemaphoreSlim(1,1);
        private readonly SemaphoreSlim runtimeInspectionGate = new SemaphoreSlim(1,1);
        private string root;
        public readonly LogBuffer Logs = new LogBuffer();
        public event Action<object> ProgressChanged;
        public event Action<object> CudaRuntimeChanged;
        public bool ExternalBusy; public int GuardCalls, StateCalls;
        public HostController(Scenario scenario,string root) { this.cudaRuntime=scenario.Service;this.root=root; }
        private string RequireProjectRoot() { return root; }
        private Task AssertMaintenanceIdleAsync(string value) { GuardCalls++; if (ExternalBusy) throw new InvalidOperationException("synthetic busy project");return Task.FromResult(0); }
        public async Task<object> GetStateAsync() { StateCalls++;return new Dictionary<string,object>{{"cudaRuntime",await cudaRuntime.GetStatusAsync(root,false)}}; }
        public async Task<object> CheckCudaRuntimeAsync(bool force) { return new Dictionary<string,object>{{"cudaRuntime",await cudaRuntime.GetStatusAsync(root,force)}}; }
        public void HoldGate() { operationGate.Wait(); } public void ReleaseGate() { operationGate.Release(); }
        public void SelectRoot(string value) { root=value; }
        public void RaiseCuda(object state) { if(CudaRuntimeChanged!=null)CudaRuntimeChanged(state); }
        public void RaiseProgress(object state) { if(ProgressChanged!=null)ProgressChanged(state); }
        __HOST_METHODS__

        private Task<object> Result() { return Task.FromResult<object>(new Dictionary<string,object>()); }
        public Task<object> RunBootstrapAsync(bool repair) { return Result(); }
        public object ChooseInstallPath(string path) { return null; } public object SetMirror(string value) { return null; }
        public object ToggleMirror() { return null; } public object SetGitNetwork(string a,string b,string c) { return null; }
        public Task<object> CheckUpdatesAsync() { return Result(); } public Task<object> ApplyUpdateAsync() { return Result(); }
        public Task<object> RollbackUpdateAsync() { return Result(); } public Task<object> PrepareFeedbackAsync(string step) { return Result(); }
        public Task<object> StartWebUiAsync() { return Result(); } public Task<object> StopWebUiAsync() { return Result(); }
        public Task<object> OpenLegacyAsync() { return Result(); } public Task<object> StartTerminalAsync() { return Result(); }
        public object OpenExternal(string value) { return null; } public object PollLogs(long since,int limit) { return null; }
    }

    public static class Harness
    {
        private static void CreateTool(string root)
        {
            Directory.CreateDirectory(Path.Combine(root,"_internal","python_base"));Directory.CreateDirectory(Path.Combine(root,"tools"));
            File.WriteAllText(Path.Combine(root,"_internal","python_base","python.exe"),"fixture: never executed");
            File.WriteAllText(Path.Combine(root,"tools","torch-runtime-switch.py"),"# fixture: never executed");
        }
        private static void Require(bool value,string message) { if(!value)throw new InvalidOperationException(message); }
        private static void Error(Action action,string fragment)
        {
            try { action(); } catch(Exception e) { if(e.Message.Contains(fragment))return;throw; }
            throw new InvalidOperationException("Expected failure: "+fragment);
        }
        private static object Dispatch(RpcBridge rpc,string method,Dictionary<string,object> parameters)
        {
            Task<object> task=(Task<object>)typeof(RpcBridge).GetMethod("DispatchAsync",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(rpc,new object[]{method,parameters});
            return task.GetAwaiter().GetResult();
        }
        public static bool Run(string name,string root)
        {
            Scenario s=new Scenario();HostController h=new HostController(s,root);
            if(name=="cross-project-coalesces-and-drains") {
                string other=Path.Combine(root,"other");CreateTool(other);
                TaskCompletionSource<CommandResult> a=new TaskCompletionSource<CommandResult>(),b=new TaskCompletionSource<CommandResult>();
                int callsA=0,callsB=0;
                CudaRuntimeService service=new CudaRuntimeService(delegate(string exe,string args,string cwd,IDictionary<string,string> env,string channel,Func<string,bool> output) {
                    if(cwd==root){callsA++;return a.Task;}callsB++;return b.Task;
                },null,delegate{return s.Clock;});
                Task<Dictionary<string,object>> first=service.GetStatusAsync(root,true),different=service.GetStatusAsync(other,true),again=service.GetStatusAsync(root,true);
                bool same=Object.ReferenceEquals(first,again);Task paused=service.PauseChecksAsync();
                a.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});first.GetAwaiter().GetResult();
                bool drainedTooSoon=paused.IsCompleted;
                b.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});different.GetAwaiter().GetResult();again.GetAwaiter().GetResult();paused.GetAwaiter().GetResult();
                service.ResumeChecks();Dictionary<string,object> stateA=service.GetStatusAsync(root,false).GetAwaiter().GetResult(),stateB=service.GetStatusAsync(other,false).GetAwaiter().GetResult();
                Require(same && callsA==1 && callsB==1,"cross-project checks were duplicated or cache evicted");
                Require(!drainedTooSoon,"pause did not drain every project's query");
                Require((string)stateA["projectDir"]==root && (string)stateB["projectDir"]==other,"project state crossed roots");
            } else if(name=="notification-errors-do-not-lock") {
                int mutations=0;
                CudaRuntimeService service=new CudaRuntimeService(delegate(string exe,string args,string cwd,IDictionary<string,string> env,string channel,Func<string,bool> output) {
                    mutations++;if(output!=null)output("DFL_CUDA_EVENT|{\"status\":\"switching\"}");
                    return Task.FromResult(new CommandResult{ExitCode=0,StandardOutput=Scenario.State("cleanup-choice",true,false)});
                },delegate(object state){throw new InvalidOperationException("synthetic disconnected UI");},delegate{return s.Clock;});
                try{service.PrepareAsync(root,"cu126").GetAwaiter().GetResult();}catch(InvalidOperationException){}
                Require(!service.IsChanging,"notification failure permanently locked runtime operations");
                service.KeepAsync(root,"0123456789abcdef0123456789abcdef").GetAwaiter().GetResult();
                Require(mutations==2,"UI notification prevented authorized operation");
            } else if(name=="missing-tool") {
                Dictionary<string,object> state=s.Service.GetStatusAsync(Path.Combine(root,"missing"),false).GetAwaiter().GetResult();
                Require((string)state["status"]=="unavailable" && s.Commands.Count==0,"missing project launched command");
            } else if(name=="cache-and-force") {
                s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();s.Clock=s.Clock.AddSeconds(10);
                s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();Require(s.StatusCalls==1,"cache not reused");
                s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();Require(s.StatusCalls==2,"force did not refresh");
                s.Clock=s.Clock.AddSeconds(21);s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();Require(s.StatusCalls==3,"expired cache reused");
            } else if(name=="clock-rollback-refreshes") {
                s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();s.Clock=s.Clock.AddHours(-1);
                s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();Require(s.StatusCalls==2,"backward clock kept stale hardware cache");
            } else if(name=="progress-uses-new-transaction") {
                List<Dictionary<string,object>> events=new List<Dictionary<string,object>>();string next="fedcba9876543210fedcba9876543210";
                CudaRuntimeService service=new CudaRuntimeService(delegate(string exe,string args,string cwd,IDictionary<string,string> env,string channel,Func<string,bool> output) {
                    if(output!=null)output("DFL_CUDA_EVENT|{\"status\":\"downloading\",\"transactionId\":\""+next+"\",\"downloadRequired\":false,\"cachedBytes\":100,\"downloadRemainingBytes\":8,\"downloadedBytes\":4}");
                    return Task.FromResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});
                },delegate(object value){events.Add(new Dictionary<string,object>((Dictionary<string,object>)value));},delegate{return s.Clock;});
                service.GetStatusAsync(root,false).GetAwaiter().GetResult();service.PrepareAsync(root,"cu126").GetAwaiter().GetResult();
                Require(events[0]["transactionId"]==null && (string)events[1]["transactionId"]==next && events[1]["downloadRequired"] is bool && !(bool)events[1]["downloadRequired"],"progress retained previous transaction identity");
                Require(Convert.ToInt64(events[1]["cachedBytes"])==100 && Convert.ToInt64(events[1]["downloadRemainingBytes"])==8 && Convert.ToInt64(events[1]["downloadedBytes"])==4,"network progress denominator was lost");
            } else if(name=="historical-cleanup-capability-preserved") {
                s.Status="{\"status\":\"retained\",\"currentRuntime\":\"cu126\",\"retainedRuntimes\":[{\"transactionId\":\"fedcba9876543210fedcba9876543210\",\"deleting\":true,\"canDelete\":true,\"canReuse\":false}]}";
                Dictionary<string,object> state=s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                object[] rows=(object[])state["retainedRuntimes"];Dictionary<string,object> row=(Dictionary<string,object>)rows[0];
                Require(CudaRuntimeService.Flag(row,"canDelete") && !CudaRuntimeService.Flag(row,"canReuse") && !CudaRuntimeService.Flag(state,"canDelete"),"historical cleanup state was flattened or lost");
            } else if(name=="cleanup-retry-needs-transaction") {
                s.Status="{\"status\":\"compatible\",\"canRetryCleanup\":true,\"cleanupWarning\":\"synthetic cleanup busy\"}";
                Dictionary<string,object> invalid=s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                Require(!CudaRuntimeService.Flag(invalid,"canRetryCleanup"),"cleanup retry enabled without its transaction");
                s.Status=Scenario.State("compatible",false,false).TrimEnd('}')+",\"canRetryCleanup\":true,\"cleanupWarning\":\"synthetic cleanup busy\"}";
                Dictionary<string,object> valid=s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                Require(CudaRuntimeService.Flag(valid,"canRetryCleanup") && (string)valid["cleanupWarning"]=="synthetic cleanup busy","cleanup retry or warning was lost");
            } else if(name=="coalesced-status") {
                s.HoldStatus=true;Task<Dictionary<string,object>> first=s.Service.GetStatusAsync(root,true),second=s.Service.GetStatusAsync(root,true);
                Require(Object.ReferenceEquals(first,second) && s.StatusCalls==1,"status duplicated");
                s.PendingStatus.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});first.GetAwaiter().GetResult();
            } else if(name=="pause-drains-query") {
                s.HoldStatus=true;s.Service.GetStatusAsync(root,true);Task pause=s.Service.PauseChecksAsync();
                Require(!pause.IsCompleted,"pause did not await its Python query");s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                Require(s.StatusCalls==1,"paused status created a process");s.PendingStatus.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});
                pause.GetAwaiter().GetResult();s.Service.ResumeChecks();s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();Require(s.StatusCalls==2,"resume did not work");
            } else if(name=="prepare-does-not-delete") {
                Dictionary<string,object> state=s.Service.PrepareAsync(root,"cu126").GetAwaiter().GetResult();
                Require((string)state["status"]=="cleanup-choice" && CudaRuntimeService.Flag(state,"canDelete"),"completion state lost");
                Require(s.MutationCalls==1 && !String.Join("\n",s.Commands).Contains("delete-retained"),"prepare deleted backup");
                Require(s.Events.Count>=3 && !CudaRuntimeService.Flag(s.Events[1],"canDelete"),"progress enabled deletion before success");
                Require((string)s.Events[1]["projectDir"]==Path.GetFullPath(root),"progress lost project identity");
            } else if(name=="operation-status-does-not-probe") {
                s.HoldMutation=true;Task<Dictionary<string,object>> operation=s.Service.PrepareAsync(root,"cu126");
                Dictionary<string,object> state=s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                Require(s.StatusCalls==0 && !CudaRuntimeService.Flag(state,"canDelete") && s.Service.IsChanging,"changing status started a probe or deletion");
                s.PendingMutation.SetResult(new CommandResult{ExitCode=0,StandardOutput=Scenario.State("cleanup-choice",true,false)});
                operation.GetAwaiter().GetResult();Require(!s.Service.IsChanging,"operation did not clear changing state");
            } else if(name=="stale-query-returns-current-progress") {
                s.HoldStatus=true;s.HoldMutation=true;Task<Dictionary<string,object>> query=s.Service.GetStatusAsync(root,true);
                Task<Dictionary<string,object>> mutation=s.Service.PrepareAsync(root,"cu126");
                s.PendingStatus.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});
                Dictionary<string,object> returned=query.GetAwaiter().GetResult();
                s.PendingMutation.SetResult(new CommandResult{ExitCode=0,StandardOutput=Scenario.State("cleanup-choice",true,false)});mutation.GetAwaiter().GetResult();
                Require((string)returned["status"]=="switching" && !CudaRuntimeService.Flag(returned,"canSwitch"),"old query response can overwrite current progress");
            } else if(name=="failure-refreshes-recovery") {
                s.FailMutation=true;Error(delegate{s.Service.PrepareAsync(root,"cu126").GetAwaiter().GetResult();},"synthetic install failure");
                Dictionary<string,object> state=s.Service.GetStatusAsync(root,false).GetAwaiter().GetResult();
                Require(CudaRuntimeService.Flag(state,"canRecover") && !s.Service.IsChanging && s.StatusCalls==1,"failure state not refreshed");
            } else if(name=="delete-requires-confirmation") {
                Error(delegate{s.Service.DeleteAsync(root,"0123456789abcdef0123456789abcdef",false);},"确认删除");Require(s.Commands.Count==0,"unconfirmed deletion executed");
            } else if(name=="invalid-target-and-token") {
                Error(delegate{s.Service.PrepareAsync(root,"cu999");},"受支持");
                Error(delegate{s.Service.KeepAsync(root,"../unsafe");},"记录无效");Require(s.Commands.Count==0,"invalid request executed");
            } else if(name=="keep-delete-recover-distinct") {
                s.Service.KeepAsync(root,"0123456789abcdef0123456789abcdef").GetAwaiter().GetResult();
                Require(s.Commands[0].EndsWith("keep-retained --transaction-id 0123456789abcdef0123456789abcdef"),"keep did not keep");
                s.Service.DeleteAsync(root,"0123456789abcdef0123456789abcdef",true).GetAwaiter().GetResult();
                Require(s.Commands[1].EndsWith("delete-retained --transaction-id 0123456789abcdef0123456789abcdef --confirm-delete"),"delete confirmation lost");
                s.Service.RecoverAsync(root,"0123456789abcdef0123456789abcdef").GetAwaiter().GetResult();Require(s.Commands[2].Contains(" recover --transaction-id 0123456789abcdef0123456789abcdef"),"recovery missing");
            } else if(name=="malformed-status-safe") {
                s.MalformedStatus=true;Dictionary<string,object> state=s.Service.GetStatusAsync(root,true).GetAwaiter().GetResult();
                Require((string)state["status"]=="unavailable" && !CudaRuntimeService.Flag(state,"canSwitch") && !CudaRuntimeService.Flag(state,"canDelete"),"malformed state unsafe");
            } else if(name=="host-gate-refuses-overlap") {
                h.HoldGate();Error(delegate{h.PrepareCudaRuntimeAsync("cu126",root).GetAwaiter().GetResult();},"其他操作");h.ReleaseGate();Require(s.Commands.Count==0,"busy operation queued mutation");
            } else if(name=="host-idle-before-mutation") {
                h.ExternalBusy=true;Error(delegate{h.PrepareCudaRuntimeAsync("cu126",root).GetAwaiter().GetResult();},"busy project");
                Require(h.GuardCalls==1 && s.MutationCalls==0 && !s.Service.IsChanging,"busy project mutated");
            } else if(name=="host-pending-requires-recovery") {
                s.Status=Scenario.State("error",false,true);Error(delegate{h.PrepareCudaRuntimeAsync("cu126",root).GetAwaiter().GetResult();},"先恢复");
                Require(s.MutationCalls==0,"pending transaction overwritten");h.RecoverCudaRuntimeAsync("0123456789abcdef0123456789abcdef",root).GetAwaiter().GetResult();
                Require(s.MutationCalls==1 && s.Commands[s.Commands.Count-1].Contains(" recover "),"explicit recovery failed");
            } else if(name=="host-query-finishes-before-idle") {
                s.HoldStatus=true;Task<object> action=h.PrepareCudaRuntimeAsync("cu126",root);Require(h.GuardCalls==0,"idle guard ran while own query active");
                s.PendingStatus.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});action.GetAwaiter().GetResult();Require(h.GuardCalls==1 && s.MutationCalls==1,"prepare never reached safe phase");
            } else if(name=="host-stale-project-refused") {
                Error(delegate{h.PrepareCudaRuntimeAsync("cu126",Path.Combine(root,"previous-project")).GetAwaiter().GetResult();},"项目已经变化");
                Require(s.Commands.Count==0,"stale UI action launched a command in another project");
            } else if(name=="host-project-change-during-check-refused") {
                s.HoldStatus=true;Task<object> action=h.PrepareCudaRuntimeAsync("cu126",root);
                h.SelectRoot(Path.Combine(root,"new-project"));s.PendingStatus.SetResult(new CommandResult{ExitCode=0,StandardOutput=s.Status});
                Error(delegate{action.GetAwaiter().GetResult();},"项目已经变化");
                Require(s.MutationCalls==0 && !s.Service.IsChanging,"project changed during async safety check but mutation ran");
            } else {
                Microsoft.Web.WebView2.Core.CoreWebView2 view=new Microsoft.Web.WebView2.Core.CoreWebView2();
                MainWindow window=new MainWindow();RpcBridge rpc=new RpcBridge(window,view,h);
                if(name=="rpc-strict-confirmation") {
                    Error(delegate{Dispatch(rpc,"deleteCudaRuntime",new Dictionary<string,object>{{"transactionId","0123456789abcdef0123456789abcdef"},{"confirmed","true"},{"expectedProjectDir",root}});},"布尔值");
                    Error(delegate{Dispatch(rpc,"deleteCudaRuntime",new Dictionary<string,object>{{"transactionId","0123456789abcdef0123456789abcdef"},{"expectedProjectDir",root}});},"确认删除");
                    Require(s.MutationCalls==0,"RPC coerced confirmation");
                } else if(name=="rpc-strict-inputs") {
                    Error(delegate{Dispatch(rpc,"prepareCudaRuntime",new Dictionary<string,object>{{"targetRuntime",123},{"expectedProjectDir",root}});},"有效参数");
                    Error(delegate{Dispatch(rpc,"keepCudaRuntime",new Dictionary<string,object>{{"transactionId","../../foreign"},{"expectedProjectDir",root}});},"记录无效");
                    Require(s.Commands.Count==0,"invalid RPC created process");
                } else if(name=="rpc-project-required") {
                    Error(delegate{Dispatch(rpc,"prepareCudaRuntime",new Dictionary<string,object>{{"targetRuntime","cu126"}});},"expectedProjectDir");
                    Error(delegate{Dispatch(rpc,"keepCudaRuntime",new Dictionary<string,object>{{"transactionId","0123456789abcdef0123456789abcdef"},{"expectedProjectDir",123}});},"expectedProjectDir");
                    Require(s.Commands.Count==0,"missing project identity launched a process");
                } else if(name=="rpc-check-is-lightweight") {
                    Dispatch(rpc,"checkCudaRuntime",new Dictionary<string,object>{{"force",true}});
                    bool changes=(bool)typeof(RpcBridge).GetMethod("ChangesState",BindingFlags.NonPublic|BindingFlags.Static).Invoke(null,new object[]{"checkCudaRuntime"});
                    Require(s.StatusCalls==1 && h.StateCalls==0 && !changes,"poll launched full health check");
                } else if(name=="rpc-explicit-prepare-and-keep") {
                    Dispatch(rpc,"prepareCudaRuntime",new Dictionary<string,object>{{"targetRuntime","cu126"},{"expectedProjectDir",root}});
                    Dispatch(rpc,"keepCudaRuntime",new Dictionary<string,object>{{"transactionId","0123456789abcdef0123456789abcdef"},{"expectedProjectDir",root}});
                    Require(s.MutationCalls==2 && !String.Join("\n",s.Commands).Contains("delete-retained"),"RPC implicit deletion");
                } else if(name=="rpc-progress-and-disposal") {
                    h.RaiseCuda(new Dictionary<string,object>{{"status","downloading"}});Require(view.Posted.Count==1 && view.Posted[0].Contains("cudaRuntime"),"progress not forwarded");
                    rpc.Dispose();h.RaiseCuda(new Dictionary<string,object>{{"status","downloading"}});Require(view.Posted.Count==1,"disposed bridge still subscribed");
                } else if(name=="rpc-disconnected-notification-safe") {
                    window.Dispatcher.RejectPost=true;h.RaiseCuda(new Dictionary<string,object>{{"status","downloading"}});
                    h.Logs.Add("cuda-runtime","synthetic background progress","info");Require(view.Posted.Count==0,"closed view received progress");
                } else throw new InvalidOperationException("Unknown test "+name);
                rpc.Dispose();
            }
            return true;
        }
    }
}
