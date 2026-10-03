[CmdletBinding()]
param(
    [ValidateSet('preview', 'build')][string]$Command = 'preview',
    [ValidateSet('source', 'portable', 'both')][string]$Kind = 'source',
    [ValidateSet('bundled', 'download')][string]$Weights = 'bundled',
    [string]$OutputDirectory = '',
    [string]$Ref = 'HEAD',
    [string]$SourceCommit = '',
    [string]$SourceRepository = '',
    [ValidateRange(1, 1900)][int]$MaxPartMiB = 1900,
    [ValidateSet('stored', 'deflated')][string]$Compression = 'deflated',
    [switch]$WritePlan,
    [switch]$ForcePlan,
    # Recognize the retired API so old automation gets an explicit safe error.
    [string]$RarExe = '',
    [string]$OutputPath = '',
    [string]$PortableNodeModulesPath = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RarExe -or $OutputPath -or $PortableNodeModulesPath) {
    throw 'The recursive RAR/external node_modules API is retired. Use -Command build -Kind source|portable|both -OutputDirectory release-output. Archives contain a Git snapshot and explicit runtime files, never .git or private workspaces.'
}
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repositoryRoot 'release-output' }
$pythonCandidates = @(
    (Join-Path $repositoryRoot '_internal/python_base/python.exe'),
    (Join-Path $repositoryRoot '.venv/Scripts/python.exe')
)
$python = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $python = $candidate; break }
}
if (-not $python) {
    $available = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($available) { $python = $available.Source }
}
if (-not $python) { throw 'Python 3.12 or newer is required to run tools/build-release.py.' }
$releaseArguments = @((Join-Path $PSScriptRoot 'build-release.py'), $Command, '--kind', $Kind,
    '--weights', $Weights, '--ref', $Ref, '--output-dir', $OutputDirectory)
if ($SourceCommit) { $releaseArguments += @('--source-commit', $SourceCommit) }
if ($SourceRepository) { $releaseArguments += @('--source-repository', $SourceRepository) }
if ($Command -eq 'build') {
    if ($WritePlan -or $ForcePlan) { throw '-WritePlan/-ForcePlan are only supported with -Command preview.' }
    $releaseArguments += @('--max-part-mib', [string]$MaxPartMiB, '--compression', $Compression)
} elseif ($WritePlan) {
    $releaseArguments += '--write-plan'
    if ($ForcePlan) { $releaseArguments += '--force-plan' }
} elseif ($ForcePlan) {
    throw '-ForcePlan requires -WritePlan.'
}
& $python @releaseArguments
if ($LASTEXITCODE -ne 0) { throw "Release $Command failed with exit code $LASTEXITCODE." }
