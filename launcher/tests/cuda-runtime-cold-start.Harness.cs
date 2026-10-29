using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization;
using System.Threading;
using System.Web.Script.Serialization;

namespace DflPtWebUi.Launcher
{
    public static class CudaRuntimeColdStartHarness
    {
        private static void Set(object value,string name,object content)
        {
            value.GetType().GetField(name,BindingFlags.Instance|BindingFlags.NonPublic).SetValue(value,content);
        }
        public static bool Run(string root,string interpreter,string status,bool canRecover,bool verifyProjectChoice=false,string lifecycleScenario=null)
        {
            Directory.CreateDirectory(root);
            foreach(string path in new[]{"webui/package.json","webui/scripts/local-manager.mjs","_internal/DeepFaceLab/me.py",
                "_internal/DeepFaceLab/me_backend/engine.py","_internal/DeepFaceLab/me_backend/network.py","tools/torch-runtime-switch.py"}) {
                string target=Path.Combine(root,path.Replace('/',Path.DirectorySeparatorChar));Directory.CreateDirectory(Path.GetDirectoryName(target));File.WriteAllText(target,"isolated fixture");
            }
            Directory.CreateDirectory(Path.Combine(root,"release"));
            File.WriteAllText(Path.Combine(root,"release/version.json"),"{\"schemaVersion\":1,\"product\":\"DFL-PT-WEBUI\",\"version\":\"fixture\"}");
            string python=Path.Combine(root,"_internal/python_base/python.exe");Directory.CreateDirectory(Path.GetDirectoryName(python));File.Copy(interpreter,python);
            File.WriteAllText(Path.Combine(root,"status.json"),new JavaScriptSerializer().Serialize(new Dictionary<string,object>{
                {"status",status},{"currentRuntime","cu128"},{"targetRuntime","cu126"},{"transactionId","0123456789abcdef0123456789abcdef"},
                {"canRecover",canRecover},{"canDelete",false},{"canSwitch",false},{"message","synthetic pending transaction"}}));
            string payload=Path.Combine(root,"payload");Directory.CreateDirectory(Path.Combine(payload,"bootstrap"));
            File.WriteAllText(Path.Combine(payload,"bootstrap/source-pin.json"),"{}");LauncherPayload.Initialize(payload);
            SettingsStore settings=(SettingsStore)FormatterServices.GetUninitializedObject(typeof(SettingsStore));
            Set(settings,"gate",new object());Set(settings,"serializer",new JavaScriptSerializer());Set(settings,"settingsPath",Path.Combine(root,"settings.json"));
            Set(settings,"current",new LauncherSettings{ProjectRoot=root});
            string logRoot=Path.Combine(root,"fixture-logs");Directory.CreateDirectory(logRoot);
            File.WriteAllText(Path.Combine(logRoot,"fixture.log"),String.Empty);File.WriteAllText(Path.Combine(logRoot,"fixture.errors.log"),String.Empty);
            LogBuffer logs=(LogBuffer)FormatterServices.GetUninitializedObject(typeof(LogBuffer));
            Set(logs,"gate",new object());Set(logs,"entries",new List<LogEntry>());Set(logs,"nextSequence",1L);
            Set(logs,"session","fixture");Set(logs,"fallback",Path.Combine(root,"fallback-logs"));Set(logs,"<DirectoryPath>k__BackingField",logRoot);
            using(HostController host=new HostController(settings,logs)) {
                if(lifecycleScenario=="source-publication") {
                    string token="0123456789abcdef0123456789abcdef",session="source-"+token;
                    string state=Path.Combine(root,".launcher-install"),receipt=Path.Combine(state,session,"publication.json");Directory.CreateDirectory(Path.GetDirectoryName(receipt));
                    File.WriteAllText(Path.Combine(state,"owner.txt"),"DFL-PT-WEBUI install workspace v1");
                    File.WriteAllText(Path.Combine(state,"source-publication.json"),new JavaScriptSerializer().Serialize(new Dictionary<string,object>{{"schemaVersion",1},{"product","DFL-PT-WEBUI"},{"session",session}}));
                    Dictionary<string,object> publication=new Dictionary<string,object>{{"schemaVersion",1},{"product","DFL-PT-WEBUI"},{"source",Path.Combine(state,session,"staging")},{"target",root},{"status","publishing"}};
                    File.WriteAllText(receipt,new JavaScriptSerializer().Serialize(publication));
                    MethodInfo recoverable=typeof(HostController).GetMethod("RecoverableSourceInstallation",BindingFlags.Static|BindingFlags.NonPublic);
                    if(!(bool)recoverable.Invoke(null,new object[]{root}))throw new InvalidOperationException("Interrupted source publication was not allowed to reach recovery");
                    publication["target"]=Path.Combine(root,"other");File.WriteAllText(receipt,new JavaScriptSerializer().Serialize(publication));
                    if((bool)recoverable.Invoke(null,new object[]{root}))throw new InvalidOperationException("A foreign source publication was accepted");
                    return true;
                }
                if(lifecycleScenario!=null && lifecycleScenario.StartsWith("webui-")) {
                    string node=Path.Combine(root,"_internal/node/bin/node.exe");Directory.CreateDirectory(Path.GetDirectoryName(node));File.Copy(interpreter,node);
                    string[] files={"_internal/node/bin/node_modules/corepack/dist/corepack.js","webui/node_modules/vite/bin/vite.js","webui/node_modules/node-pty/package.json","webui/node_modules/old.txt","webui/dist/client/index.html","webui/scripts/prepare-sites-build.mjs","webui/package.json","webui/pnpm-lock.yaml","tools/dist-provenance.mjs"};
                    foreach(string rel in files){string path=Path.Combine(root,rel);Directory.CreateDirectory(Path.GetDirectoryName(path));File.WriteAllText(path,"old fixture");}
                    File.WriteAllText(Path.Combine(root,"_internal/setenv.bat"),"@echo off\r\nset DFL_ACTIVE_PROJECT_ID=default\r\nset WORKSPACE="+Path.Combine(root,"workspace")+"\r\n");
                    if(lifecycleScenario!="webui-install-failure")WebUiDependencies.RecordDependencyInputs(root);
                    if(lifecycleScenario=="webui-install-failure")File.WriteAllText(Path.Combine(root,"fail-install"),"fixture");
                    if(lifecycleScenario=="webui-build-failure")File.WriteAllText(Path.Combine(root,"fail-build"),"fixture");
                    if(lifecycleScenario=="webui-stale-build")File.WriteAllText(Path.Combine(root,"build-outdated"),"fixture");
                    bool failed=false;
                    try{((System.Threading.Tasks.Task)typeof(HostController).GetMethod("BuildWebUiIfPossibleAsync",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(host,new object[]{root,lifecycleScenario!="webui-stale-build"})).GetAwaiter().GetResult();}
                    catch(InvalidOperationException){failed=true;}
                    if(failed!=(lifecycleScenario!="webui-healthy" && lifecycleScenario!="webui-stale-build"))throw new InvalidOperationException("Unexpected preparation failure result");
                    if(File.ReadAllText(Path.Combine(root,"webui/node_modules/old.txt"))!="old fixture")throw new InvalidOperationException("Old dependency tree was not preserved/restored");
                    if(failed && File.ReadAllText(Path.Combine(root,"webui/dist/client/index.html"))!="old fixture")throw new InvalidOperationException("Failed build did not restore old distribution");
                    string calls=File.ReadAllText(Path.Combine(root,"node-calls.txt"));
                    if(lifecycleScenario!="webui-install-failure" && calls.Contains("install"))throw new InvalidOperationException("Healthy dependency tree was unnecessarily reinstalled");
                    if(!failed && !calls.Contains("build"))throw new InvalidOperationException("Explicit rebuild was skipped");
                    if(!failed && !calls.Contains("provenance write"))throw new InvalidOperationException("Build provenance was not verified before publication");
                    return true;
                }
                if(lifecycleScenario!=null && lifecycleScenario.StartsWith("resources-")) {
                    foreach(string name in new[]{"bootstrap.ps1","setup-runtime.ps1","runtime-manifest.json"})
                        File.WriteAllText(Path.Combine(payload,"bootstrap",name),name.EndsWith(".json") ? "{}" : "# synthetic bootstrap");
                    string report=Path.Combine(root,".launcher-install/vision/resources.json");Directory.CreateDirectory(Path.GetDirectoryName(report));
                    File.WriteAllText(report,lifecycleScenario=="resources-pending" ? "{\"schemaVersion\":1,\"complete\":false,\"failed\":[]}" : "{\"schemaVersion\":1,\"complete\":false,\"failed\":\"broken\"}");
                }
                if(lifecycleScenario=="maintenance") Set(host,"maintenanceInProgress",true);
                if(lifecycleScenario=="python-recovery") {
                    string journal=Path.Combine(root,".launcher-install/source/python-repair.json");Directory.CreateDirectory(Path.GetDirectoryName(journal));
                    File.WriteAllText(journal,"{\"schemaVersion\":1,\"phase\":\"installing\"}");
                }
                if(lifecycleScenario=="upgrade-recovery") {
                    File.WriteAllText(Path.Combine(payload,"bootstrap/bootstrap.ps1"),"# isolated bootstrap marker");
                    File.WriteAllText(Path.Combine(payload,"bootstrap/setup-runtime.ps1"),"# isolated setup marker");
                    File.WriteAllText(Path.Combine(payload,"bootstrap/runtime-manifest.json"),"{}");
                    string journal=Path.Combine(root,".launcher-install/upgrade-current/journal.json");Directory.CreateDirectory(Path.GetDirectoryName(journal));File.WriteAllText(journal,"{}");
                    File.WriteAllText(Path.Combine(root,"release/version.json"),"{corrupt interrupted source identity");
                    File.WriteAllText(Path.Combine(payload,"bootstrap/upgrade-project.ps1"),
                        "param([string]$ProjectRoot,[string]$Action,[string]$SourcePinPath)\n"+
                        "if($Action -eq 'status'){Write-Output '{\"status\":\"applied\",\"canRollback\":true,\"message\":\"synthetic interrupted upgrade\"}'}\n"+
                        "if($Action -eq 'rollback'){[IO.File]::WriteAllText((Join-Path $ProjectRoot 'rollback.called'),'fixture recovery');Remove-Item -LiteralPath (Join-Path $ProjectRoot '.launcher-install/upgrade-current/journal.json')}\n");
                }
                if(lifecycleScenario=="operation-dedup") {
                    SemaphoreSlim operation=(SemaphoreSlim)typeof(HostController).GetField("operationGate",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(host);
                    operation.Wait();
                    try {
                        System.Threading.Tasks.Task<object> action=host.RunBootstrapAsync(true);
                        try { if(!action.Wait(1000))throw new InvalidOperationException("A duplicate repair was queued instead of rejected"); }
                        catch(AggregateException error) { if(!(error.InnerException is InvalidOperationException))throw; }
                        if(!action.IsFaulted)throw new InvalidOperationException("Duplicate repair did not fail before side effects");
                    }
                    finally { operation.Release(); }
                    return true;
                }
                if(verifyProjectChoice) {
                    SemaphoreSlim operation=(SemaphoreSlim)typeof(HostController).GetField("operationGate",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(host);
                    bool refused=false;operation.Wait();
                    try { host.ChooseInstallPath(root); }
                    catch(InvalidOperationException error) { refused=error.Message.Contains("等待完成"); }
                    finally { operation.Release(); }
                    if(!refused)throw new InvalidOperationException("Project selection was allowed during a native operation");
                }
                Dictionary<string,object> result=(Dictionary<string,object>)host.GetStateAsync().GetAwaiter().GetResult();
                if(lifecycleScenario!=null) {
                    if(lifecycleScenario.StartsWith("resources-")) {
                        Dictionary<string,object> resources=result["resourcePreparation"] as Dictionary<string,object>;
                        if(resources==null || (bool)resources["complete"] || (string)result["mode"]!="install")throw new InvalidOperationException("Pending resources were hidden");
                        if(lifecycleScenario=="resources-invalid" && !resources.ContainsKey("message"))throw new InvalidOperationException("Invalid resource report did not offer recovery");
                    }
                    if(lifecycleScenario=="maintenance" && (string)result["environmentStatus"]!="changing")throw new InvalidOperationException("Maintenance state imported or validated the runtime");
                    if(lifecycleScenario=="python-recovery" && !(bool)result["runtimeRepairPending"])throw new InvalidOperationException("Interrupted Python replacement has no recovery state");
                    if(lifecycleScenario=="upgrade-recovery") {
                        if(!(bool)result["upgradePending"] || !(bool)result["upgradeCanRollback"])throw new InvalidOperationException("Corrupt source identity hid upgrade recovery");
                        host.RollbackUpdateAsync().GetAwaiter().GetResult();
                        if(!File.Exists(Path.Combine(root,"rollback.called")))throw new InvalidOperationException("Rollback was blocked by damaged project identity");
                    }
                    if(File.Exists(Path.Combine(root,"unexpected-import.txt")))throw new InvalidOperationException("Maintenance or pending replacement imported Torch");
                    return true;
                }
                Dictionary<string,object> cuda=(Dictionary<string,object>)result["cudaRuntime"];
                if((string)result["mode"]!="install" || (bool)cuda["canRecover"]!=canRecover)throw new InvalidOperationException("Cold-start recovery state was lost");
                if(File.Exists(Path.Combine(root,"unexpected-import.txt")))throw new InvalidOperationException("Partial runtime was imported during cold start");
                if((string)result["environmentStatus"]!=(canRecover?"incomplete":"changing"))throw new InvalidOperationException("Incorrect cold-start environment state");
            }
            return true;
        }
    }
}
