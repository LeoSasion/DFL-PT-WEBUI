Describe 'CUDA runtime host operations' {
    BeforeAll {
        $repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:compileRoot = Join-Path $TestDrive 'compile'
        New-Item -ItemType Directory -Path $script:compileRoot | Out-Null
        function Get-TestHostMethod([string]$Source, [string]$Signature) {
            $start = $Source.IndexOf($Signature, [StringComparison]::Ordinal)
            if ($start -lt 0) { throw "Missing actual host method: $Signature" }
            $open = $Source.IndexOf('{', $start); $depth = 0
            for ($position = $open; $position -lt $Source.Length; $position++) {
                if ($Source[$position] -eq '{') { $depth++ }
                elseif ($Source[$position] -eq '}') { $depth--; if ($depth -eq 0) { return $Source.Substring($start, $position-$start+1) } }
            }
            throw "Unbalanced actual host method: $Signature"
        }
        $hostSource = [IO.File]::ReadAllText((Join-Path $PSScriptRoot '../host/HostController.cs'))
        $methods = @(
            'public Task<object> PrepareCudaRuntimeAsync(', 'public Task<object> KeepCudaRuntimeAsync(',
            'public Task<object> DeleteCudaRuntimeAsync(', 'public Task<object> RecoverCudaRuntimeAsync(',
            'private async Task<object> RunCudaRuntimeOperationAsync(', 'private async Task AssertCudaRuntimeUsableAsync(',
            'private static bool CudaRuntimeNeedsRecovery(', 'private static void AssertExpectedProject('
        ) | ForEach-Object { Get-TestHostMethod $hostSource $_ }
        $harness = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'cuda-runtime-host.Harness.cs')).Replace('__HOST_METHODS__',($methods -join "`n"))
        $harnessPath = Join-Path $script:compileRoot 'Harness.cs'
        [IO.File]::WriteAllText($harnessPath,$harness)
        $sources = @($harnessPath)
        foreach ($name in @('CudaRuntimeService.cs','ProcessRunner.cs','RpcBridge.cs')) {
            $source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot ('../host/'+$name))).Replace('namespace DflPtWebUi.Launcher','namespace CudaRuntimeHostTests')
            $target = Join-Path $script:compileRoot $name
            [IO.File]::WriteAllText($target,$source)
            $sources += $target
        }
        Add-Type -Path $sources -ReferencedAssemblies @('System','System.Core','System.Web.Extensions')
    }

    $cases = @(
        'missing-tool','cache-and-force','coalesced-status','pause-drains-query',
        'cross-project-coalesces-and-drains','notification-errors-do-not-lock',
        'clock-rollback-refreshes','progress-uses-new-transaction','historical-cleanup-capability-preserved',
        'cleanup-retry-needs-transaction',
        'prepare-does-not-delete','operation-status-does-not-probe','stale-query-returns-current-progress','failure-refreshes-recovery','delete-requires-confirmation',
        'invalid-target-and-token','keep-delete-recover-distinct','malformed-status-safe',
        'host-gate-refuses-overlap','host-idle-before-mutation','host-pending-requires-recovery',
        'host-query-finishes-before-idle','rpc-strict-confirmation','rpc-strict-inputs',
        'host-stale-project-refused','host-project-change-during-check-refused',
        'rpc-project-required','rpc-check-is-lightweight','rpc-explicit-prepare-and-keep','rpc-progress-and-disposal','rpc-disconnected-notification-safe'
    )
    foreach ($case in $cases) {
        It $case -TestCases @(@{ CaseName=$case }) {
            param($CaseName)
            $root = Join-Path $TestDrive $CaseName
            New-Item -ItemType Directory -Path (Join-Path $root '_internal/python_base') -Force | Out-Null
            New-Item -ItemType Directory -Path (Join-Path $root 'tools') -Force | Out-Null
            [IO.File]::WriteAllText((Join-Path $root '_internal/python_base/python.exe'),'fixture: never executed')
            [IO.File]::WriteAllText((Join-Path $root 'tools/torch-runtime-switch.py'),'# fixture: never executed')
            [CudaRuntimeHostTests.Harness]::Run($CaseName,$root) | Should Be $true
        }
    }
}
