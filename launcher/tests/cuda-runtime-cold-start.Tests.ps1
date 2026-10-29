Describe 'CUDA transaction cold start uses no partial runtime' {
    BeforeAll {
        $root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $compile = Join-Path $TestDrive 'full-host'
        New-Item -ItemType Directory -Path $compile | Out-Null
        $framework = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319'
        $compiler = Join-Path $framework 'csc.exe'
        $references = @()
        foreach ($name in @('System.dll','System.Core.dll','System.Drawing.dll','System.Web.dll','System.Web.Extensions.dll','System.Windows.Forms.dll','System.Xaml.dll','WindowsBase.dll','PresentationCore.dll','PresentationFramework.dll')) {
            $candidate = @((Join-Path $framework $name),(Join-Path $framework ('WPF/'+$name))) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
            if (-not $candidate) {
                foreach ($gac in @('GAC_MSIL','GAC_64')) {
                    $folder = Join-Path $env:WINDIR ('Microsoft.NET/assembly/'+$gac+'/'+[IO.Path]::GetFileNameWithoutExtension($name))
                    if (Test-Path -LiteralPath $folder) { $candidate = Get-ChildItem -LiteralPath $folder -Filter $name -Recurse -File | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName }
                    if ($candidate) { break }
                }
            }
            if (-not $candidate) { throw "Missing .NET reference: $name" }
            $references += [string]$candidate
        }
        $sdk = Join-Path $root 'launcher/vendor/webview2/1.0.4129.50/lib/net462'
        $references += (Join-Path $sdk 'Microsoft.Web.WebView2.Core.dll'),(Join-Path $sdk 'Microsoft.Web.WebView2.Wpf.dll')
        $assembly = Join-Path $compile 'FullHostTests.dll'
        $arguments = @('/nologo','/noconfig','/target:library','/platform:x64','/langversion:5','/utf8output',('/out:'+$assembly))
        $arguments += $references | ForEach-Object { '/reference:'+$_ }
        $arguments += Get-ChildItem -LiteralPath (Join-Path $root 'launcher/host') -Filter '*.cs' -File | ForEach-Object FullName
        $arguments += Join-Path $PSScriptRoot 'cuda-runtime-cold-start.Harness.cs'
        & $compiler $arguments
        if ($LASTEXITCODE -ne 0) { throw 'Full actual native host did not compile.' }
        $fake = Join-Path $compile 'FakePython.cs'
        [IO.File]::WriteAllText($fake,@'
using System;using System.IO;using System.Reflection;
class FakePython { static int Main(string[] args) {
 string root=Path.GetFullPath(Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),"..",".."));
 if(Array.IndexOf(args,"status")>=0){Console.WriteLine(File.ReadAllText(Path.Combine(root,"status.json")));return 0;}
 File.WriteAllText(Path.Combine(root,"unexpected-import.txt"),String.Join(" ",args));Console.Error.WriteLine("Unexpected interpreter command");return 99;
} }
'@)
        $script:interpreter = Join-Path $compile 'FakePython.exe'
        & $compiler '/nologo' '/target:exe' '/platform:x64' ('/out:'+$script:interpreter) $fake
        if ($LASTEXITCODE -ne 0) { throw 'Fixture interpreter did not compile.' }
        $fakeNode = Join-Path $compile 'FakeNode.cs'
        [IO.File]::WriteAllText($fakeNode,@'
using System;using System.IO;using System.Reflection;
class FakeNode {
 static void Write(string root,string path,string value){string p=Path.Combine(root,path);Directory.CreateDirectory(Path.GetDirectoryName(p));File.WriteAllText(p,value);}
 static int Main(string[] args){
  string root=Path.GetFullPath(Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),"..","..",".."));
  if(args.Length>0 && args[0]=="-e"){File.AppendAllText(Path.Combine(root,"node-calls.txt"),"probe\n");return 0;}
  if(args.Length>1 && args[0].EndsWith("dist-provenance.mjs")){File.AppendAllText(Path.Combine(root,"node-calls.txt"),"provenance "+args[1]+"\n");return args[1]=="check" && File.Exists(Path.Combine(root,"build-outdated"))?5:0;}
  bool install=Array.IndexOf(args,"pnpm")>=0;
  File.AppendAllText(Path.Combine(root,"node-calls.txt"),install?"install\n":"build\n");
  if(install){Write(root,"webui/node_modules/partial.txt","new partial");return File.Exists(Path.Combine(root,"fail-install"))?3:0;}
  Write(root,"webui/dist/client/index.html","new build");return File.Exists(Path.Combine(root,"fail-build"))?4:0;
 }
}
'@)
        $script:nodeFixture = Join-Path $compile 'FakeNode.exe'
        & $compiler '/nologo' '/target:exe' '/platform:x64' ('/out:'+$script:nodeFixture) $fakeNode
        if ($LASTEXITCODE -ne 0) { throw 'Fixture Node did not compile.' }
        # Load bytes so Windows does not pin the temporary DLL past TestDrive cleanup.
        [Reflection.Assembly]::Load([IO.File]::ReadAllBytes($assembly)) | Out-Null
    }
    It 'offers recovery after a crash without invoking the Torch validator' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'pending-recovery'),$script:interpreter,'error',$true) | Should Be $true
    }
    It 'keeps status visible while another operation is switching without importing its files' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'pending-switch'),$script:interpreter,'switching',$false) | Should Be $true
    }
    It 'refuses project selection while a native operation owns the gate' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'selection-guard'),$script:interpreter,'switching',$false,$true) | Should Be $true
    }
    It 'serves maintenance progress without importing the environment' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'maintenance'),$script:interpreter,'compatible',$false,$false,'maintenance') | Should Be $true
    }
    It 'exposes pending auxiliary resources on cold start' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'resources-pending'),$script:interpreter,'compatible',$false,$false,'resources-pending') | Should Be $true
    }
    It 'offers retry for a malformed resource report without passing it to the UI' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'resources-invalid'),$script:interpreter,'compatible',$false,$false,'resources-invalid') | Should Be $true
    }
    It 'offers Python replacement recovery before any native validation' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'python-recovery'),$script:interpreter,'compatible',$false,$false,'python-recovery') | Should Be $true
    }
    It 'recovers a pending source update even when project identity is corrupt' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'source-recovery'),$script:interpreter,'compatible',$false,$false,'upgrade-recovery') | Should Be $true
    }
    It 'rejects duplicate repair requests before they can queue downloads' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'dedup'),$script:interpreter,'compatible',$false,$false,'operation-dedup') | Should Be $true
    }
    It 'rebuilds with healthy dependencies without reinstalling them' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'webui-healthy'),$script:nodeFixture,'compatible',$false,$false,'webui-healthy') | Should Be $true
    }
    It 'restores the original distribution after an actual host build command fails' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'webui-build-failure'),$script:nodeFixture,'compatible',$false,$false,'webui-build-failure') | Should Be $true
    }
    It 'restores the original dependency tree after an actual host install command fails' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'webui-install-failure'),$script:nodeFixture,'compatible',$false,$false,'webui-install-failure') | Should Be $true
    }
    It 'allows owned interrupted source installation to reach its strict recovery script' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'source-publication'),$script:interpreter,'compatible',$false,$false,'source-publication') | Should Be $true
    }
    It 'rebuilds stale source outputs while reusing the healthy dependency tree' {
        [DflPtWebUi.Launcher.CudaRuntimeColdStartHarness]::Run((Join-Path $TestDrive 'webui-stale-build'),$script:nodeFixture,'compatible',$false,$false,'webui-stale-build') | Should Be $true
    }
}
