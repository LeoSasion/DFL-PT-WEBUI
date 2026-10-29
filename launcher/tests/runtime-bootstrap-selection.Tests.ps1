Describe 'runtime bootstrap manifest version selection' {
    BeforeAll {
        $source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot '../host/RuntimeBootstrapLocator.cs'))
        $source = $source.Replace('namespace DflPtWebUi.Launcher', 'namespace RuntimeSelectionTests')
        $source = $source.Replace('internal sealed class RuntimeBootstrapResources', 'public sealed class RuntimeBootstrapResources')
        $source = $source.Replace('internal static class RuntimeBootstrapLocator', 'public static class RuntimeBootstrapLocator')
        Add-Type -TypeDefinition $source
        $script:projectManifest = [IO.File]::ReadAllText((Join-Path $PSScriptRoot '../runtime-manifest.json'))
    }

    function New-RuntimeSelectionFixture([string]$Root, [string]$Manifest) {
        New-Item -ItemType Directory -Path $Root -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $Root 'runtime-manifest.json'), $Manifest)
        foreach ($script in @('bootstrap.ps1', 'setup-runtime.ps1')) {
            [IO.File]::WriteAllText((Join-Path $Root $script), '# isolated fixture; never executed')
        }
    }

    It 'uses the new project CUDA manifest ahead of an older embedded manifest' {
        $project = Join-Path $TestDrive 'project-new'
        $launcher = Join-Path $project 'launcher'
        $embedded = Join-Path $TestDrive 'embedded-old'
        New-RuntimeSelectionFixture $launcher $script:projectManifest
        New-RuntimeSelectionFixture $embedded '{"manifestVersion":"2026.10.07.1"}'
        $result = [RuntimeSelectionTests.RuntimeBootstrapLocator]::Resolve($project, $embedded)
        $result.Embedded | Should Be $false
        $result.RootPath | Should Be $launcher
        $result.ManifestPath | Should Be (Join-Path $launcher 'runtime-manifest.json')
        ($script:projectManifest | ConvertFrom-Json).manifestVersion | Should Be '2026.10.29.1'
    }

    It 'retains the newer embedded fallback when project bootstrap is older' {
        $project = Join-Path $TestDrive 'project-old'
        $embedded = Join-Path $TestDrive 'embedded-new'
        New-RuntimeSelectionFixture (Join-Path $project 'launcher') '{"manifestVersion":"2026.10.07.1"}'
        New-RuntimeSelectionFixture $embedded $script:projectManifest
        $result = [RuntimeSelectionTests.RuntimeBootstrapLocator]::Resolve($project, $embedded)
        $result.Embedded | Should Be $true
        $result.RootPath | Should Be $embedded
    }

    It 'does not select an incomplete project bootstrap merely because its manifest is newer' {
        $project = Join-Path $TestDrive 'project-incomplete'
        $launcher = Join-Path $project 'launcher'
        $embedded = Join-Path $TestDrive 'embedded-complete'
        New-RuntimeSelectionFixture $launcher $script:projectManifest
        New-RuntimeSelectionFixture $embedded '{"manifestVersion":"2026.10.07.1"}'
        Remove-Item -LiteralPath (Join-Path $launcher 'setup-runtime.ps1')
        $result = [RuntimeSelectionTests.RuntimeBootstrapLocator]::Resolve($project, $embedded)
        $result.Embedded | Should Be $true
        $result.RootPath | Should Be $embedded
    }
}
