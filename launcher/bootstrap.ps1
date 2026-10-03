[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [ValidateSet('auto','china','official')][string]$Mirror = 'auto',
    [switch]$Repair, [switch]$GitOnly, [string]$ManifestPath, [switch]$DryRun, [switch]$NoNetwork
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\','/')
if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\','/')) { throw 'ProjectRoot must be an independent project directory.' }
if ([string]::IsNullOrWhiteSpace($ManifestPath)) { $ManifestPath = Join-Path $PSScriptRoot 'runtime-manifest.json' }
function Write-JsonEvent {
    param([string]$Stage,[string]$Id,[string]$Status,[int]$Progress,[string]$Message)
    [Console]::Out.WriteLine((@{ stage=$Stage; id=$Id; status=$Status; progress=$Progress; downloaded=0; total=0; message=$Message } | ConvertTo-Json -Compress))
}
function Resolve-ChildPath {
    param([string]$Base,[string]$Relative)
    if ([string]::IsNullOrWhiteSpace($Relative) -or [IO.Path]::IsPathRooted($Relative)) { throw 'Runtime paths must be relative.' }
    $candidate = [IO.Path]::GetFullPath((Join-Path $Base $Relative))
    if (-not $candidate.StartsWith($Base.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime path escapes the project.' }
    return $candidate
}
try {
    if ($GitOnly) { throw 'Launcher-managed cloning is disabled for this preview. Use install-source.bat or the portable release.' }
    if (-not (Test-Path -LiteralPath (Join-Path $root '_internal\DeepFaceLab\me.py')) -or -not (Test-Path -LiteralPath (Join-Path $root 'webui'))) { throw 'The selected directory is not a DFL-PT-WEBUI project.' }
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$manifest.schemaVersion -ne 3) { throw 'Only the DFL-PT-WEBUI runtime manifest is accepted.' }
    if (-not $DryRun) {
        $setup = Join-Path $PSScriptRoot 'setup-runtime.ps1'
        if ($Repair -and -not $NoNetwork) { & $setup -ProjectRoot $root -InstallDependencies }
        else { & $setup -ProjectRoot $root -NoNetwork }
    }
    $failed = New-Object 'System.Collections.Generic.List[string]'
    foreach ($component in $manifest.components) {
        $target = Resolve-ChildPath $root ([string]$component.install.relativePath)
        $reason = $null
        foreach ($rule in $component.validation.files) {
            $candidate = Resolve-ChildPath $target ([string]$rule.path)
            if ($rule.kind -eq 'directory') {
                if (-not (Test-Path -LiteralPath $candidate -PathType Container)) { $reason = "Missing directory: $candidate"; break }
            } elseif (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { $reason = "Missing file: $candidate"; break }
            elseif ((Get-Item -LiteralPath $candidate).Length -lt [long]$rule.minBytes) { $reason = "Incomplete file: $candidate"; break }
        }
        if (-not $reason -and $component.validation.command) {
            $executable = Resolve-ChildPath $target ([string]$component.validation.command.path)
            $arguments = @($component.validation.command.arguments | ForEach-Object { [string]$_ })
            $output = & $executable @arguments 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0 -or $output.Trim() -notmatch [string]$component.validation.command.outputRegex) { $reason = "Runtime command failed: $($output.Trim())" }
        }
        if ($reason) {
            $status = if ($DryRun) { 'missing' } else { 'failed' }
            Write-JsonEvent 'runtime' ([string]$component.id) $status 0 $reason
            if ($component.required) { $failed.Add($reason) }
        } else { Write-JsonEvent 'runtime' ([string]$component.id) 'ready' 100 "$($component.displayName) validated." }
    }
    if ($failed.Count -gt 0 -and -not $DryRun) { throw ($failed -join [Environment]::NewLine) }
    Write-JsonEvent 'bootstrap' 'complete' 'complete' 100 'Local runtime check complete. CUDA and cuDNN are included in the PyTorch wheel.'
} catch {
    Write-JsonEvent 'bootstrap' 'failed' 'failed' 0 $_.Exception.Message
    exit 1
}
