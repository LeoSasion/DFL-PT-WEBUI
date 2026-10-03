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
