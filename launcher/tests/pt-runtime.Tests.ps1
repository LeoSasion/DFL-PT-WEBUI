Describe 'PyTorch runtime migration' {
    BeforeAll {
        $source = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../host/RuntimeManifestValidator.cs') -Raw -Encoding UTF8
        $source = $source.Replace('namespace DflPtWebUi.Launcher','namespace PtRuntimeTests').Replace('internal static class RuntimeManifestValidator','public static class RuntimeManifestValidator').Replace('internal sealed class RuntimeManifestValidation','public sealed class RuntimeManifestValidation').Replace('internal sealed class RuntimeComponentValidation','public sealed class RuntimeComponentValidation')
        Add-Type -TypeDefinition $source -ReferencedAssemblies @('System','System.Core','System.Web.Extensions')
        $script:repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    }
    It 'validates the independent local Python / PyTorch runtime without legacy CUDA components' {
        $result = [PtRuntimeTests.RuntimeManifestValidator]::Validate($script:repositoryRoot,(Join-Path $PSScriptRoot '../runtime-manifest.json'))
        $result.Loaded | Should Be $true
        $result.RequiredComponentsReady | Should Be $true
        $result.Components.ContainsKey('cuda') | Should Be $false
        $result.Components.ContainsKey('cudnn') | Should Be $false
        $result.Components['python'].TargetPath | Should Be (Join-Path $script:repositoryRoot '.venv')
    }
    It 'rejects legacy manifest schemas before checking installed runtimes' {
        $manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../runtime-manifest.json') -Raw | ConvertFrom-Json
        $manifest.schemaVersion = 2
        $target = Join-Path $TestDrive 'legacy-manifest.json'
        [IO.File]::WriteAllText($target,($manifest | ConvertTo-Json -Depth 10))
        $result = [PtRuntimeTests.RuntimeManifestValidator]::Validate($script:repositoryRoot,$target)
        $result.Loaded | Should Be $false
        $result.RequiredComponentsReady | Should Be $false
    }
    It 'accepts slow cold Python startup while retaining the short Node deadline' {
        $directory=Join-Path $TestDrive 'cold-runtime';New-Item -ItemType Directory -Path $directory | Out-Null
        $executable=Join-Path $directory 'delayed-check.exe'
        Add-Type -TypeDefinition 'public class ColdRuntimeProbe { public static void Main(string[] args) { System.Threading.Thread.Sleep(11000); System.Console.WriteLine("COLD_READY"); } }' -OutputType ConsoleApplication -OutputAssembly $executable
        $serializer=New-Object Web.Script.Serialization.JavaScriptSerializer
        $method=[PtRuntimeTests.RuntimeManifestValidator].GetMethod('ValidateComponent',[Reflection.BindingFlags]'NonPublic,Static')
        foreach($id in @('python','node')){
            $component=$serializer.DeserializeObject('{"id":"'+$id+'","required":true,"install":{"relativePath":"cold-runtime"},"validation":{"files":[{"path":"delayed-check.exe","minBytes":32}],"command":{"path":"delayed-check.exe","arguments":[],"outputRegex":"^COLD_READY$"}}}')
            $arguments=New-Object object[] 2
            $arguments[0]=[string]$TestDrive;$arguments[1]=$component
            $result=$method.Invoke($null,$arguments)
            if($id -eq 'python'){$result.Ready | Should Be $true}
            else{$result.Ready | Should Be $false;$result.Reason | Should Match '\u8d85\u65f6'}
        }
    }
    It 'fails closed when FFmpeg is omitted from the new runtime manifest' {
        $manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../runtime-manifest.json') -Raw | ConvertFrom-Json
        $manifest.components = @($manifest.components | Where-Object { $_.id -ne 'ffmpeg' })
        $target = Join-Path $TestDrive 'incomplete-manifest.json'
        [IO.File]::WriteAllText($target,($manifest | ConvertTo-Json -Depth 10))
        $result = [PtRuntimeTests.RuntimeManifestValidator]::Validate($script:repositoryRoot,$target)
        $result.Loaded | Should Be $false
        $result.RequiredComponentsReady | Should Be $false
        $result.Error | Should Match 'ffmpeg'
    }
}
