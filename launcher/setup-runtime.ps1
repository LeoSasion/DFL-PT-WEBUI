[CmdletBinding()]
param([string]$ProjectRoot = '', [switch]$InstallDependencies, [switch]$NoNetwork,
    [string]$WheelhousePath = '', [ValidateSet('production','restoration','validation','evaluation','scene')][string]$RuntimeProfile = 'production',
    [ValidateSet('auto','cu126','cu128')][string]$CudaRuntime = 'auto',
    [Parameter(DontShow=$true)][IO.FileStream]$SourceInstallLock,
    [Parameter(DontShow=$true)][string]$SourceInstallToken = '',
    [Parameter(DontShow=$true)][switch]$PreserveVenvConfiguration)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\','/')
if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\','/')) { throw 'ProjectRoot must be an independent project directory.' }
$rootPrefix = $root + [IO.Path]::DirectorySeparatorChar
function Assert-SetupPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime path escapes the project.' }
    $ancestor = $absolute
    while ($ancestor -and ($ancestor -eq $root -or $ancestor.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase))) {
        try {
            if (([IO.File]::GetAttributes($ancestor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Linked runtime path is not supported: $ancestor" }
        } catch [IO.FileNotFoundException] {} catch [IO.DirectoryNotFoundException] {}
        if ($ancestor -eq $root) { break }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $absolute
}
$cache = Assert-SetupPath (Join-Path $root '.launcher-install\source')
$lockPath = Assert-SetupPath (Join-Path $cache 'source-install.lock')
$ownerPath = Assert-SetupPath (Join-Path $cache 'source-install-owner.json')
$basePython = Assert-SetupPath (Join-Path $root '_internal\python_base\python.exe')
$helper = Join-Path $PSScriptRoot 'runtime-install.py'
if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) { $helper = Join-Path $root 'tools\runtime-install.py' }
$runtimeApiPath = Assert-SetupPath (Join-Path $root '_internal\DeepFaceLab\core\gpu_runtime.py')
$guardApiPath = Assert-SetupPath (Join-Path $root '_internal\DeepFaceLab\core\torch_runtime_switch.py')
if (-not (Test-Path -LiteralPath $runtimeApiPath -PathType Leaf) -or -not (Test-Path -LiteralPath $guardApiPath -PathType Leaf) -or
    (Get-Content -LiteralPath $runtimeApiPath -Raw -Encoding UTF8) -notmatch '(?m)^def validate_runtime\([^\r\n]*maintenance_transaction' -or
    -not (Get-Content -LiteralPath $runtimeApiPath -Raw -Encoding UTF8).Contains('--maintenance-transaction') -or
    (Get-Content -LiteralPath $guardApiPath -Raw -Encoding UTF8) -notmatch '(?m)^def assert_runtime_available\([^\r\n]*maintenance_transaction') {
    throw 'This source package predates transactional runtime repair. Update the verified source package before setup or repair.'
}
if (-not (Test-Path -LiteralPath (Join-Path $root '_internal\DeepFaceLab\me.py') -PathType Leaf) -or
    -not (Test-Path -LiteralPath $basePython -PathType Leaf) -or -not (Test-Path -LiteralPath $helper -PathType Leaf)) {
    throw 'The complete source, independent Python base and runtime installer helper are required.'
}
Assert-SetupPath (Join-Path $root '.venv') | Out-Null
if ($WheelhousePath) {
    $WheelhousePath = [IO.Path]::GetFullPath($WheelhousePath)
    if (-not (Test-Path -LiteralPath $WheelhousePath -PathType Container)) { throw 'The read-only wheelhouse directory is missing.' }
}
$lock = $null; $ownsLock = $false; $token = $SourceInstallToken; $savedEnvironment = @{}
try {
    if ($null -ne $SourceInstallLock) {
        if (-not $SourceInstallLock.CanWrite -or [IO.Path]::GetFullPath($SourceInstallLock.Name) -ne $lockPath -or $token -notmatch '^[0-9a-f]{32}$') { throw 'Invalid source installer lock handoff.' }
        $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($owner.token -ne $token -or $owner.pid -ne $PID) { throw 'The source installer lock owner changed.' }
    } else {
        Import-Module (Join-Path $PSHOME 'Modules\CimCmdlets\CimCmdlets.psd1') -Force
        New-Item -ItemType Directory -Path $cache -Force | Out-Null
        try { $lock = [IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
        catch { throw 'Another source/runtime installation is active.' }
        $ownsLock = $true; $token = [Guid]::NewGuid().ToString('N')
        [IO.File]::WriteAllText($ownerPath,(@{schemaVersion=1;token=$token;pid=$PID}|ConvertTo-Json -Compress),(New-Object Text.UTF8Encoding($false)))
        $commandPath = '(?i)(?:^|[\s"''=])' + [regex]::Escape($root) + '(?:[\\/]|(?=[\s"'']|$))'
        $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='node.exe' OR Name='ffmpeg.exe' OR Name='ffprobe.exe'" | Where-Object {
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) -or ($_.CommandLine -and $_.CommandLine -match $commandPath)
        })
        if ($busy.Count) { throw 'Stop WebUI and project tasks before preparing or repairing the Python runtime.' }
    }
    foreach ($entry in @{PYTHONDONTWRITEBYTECODE='1';PYTHONUTF8='1';NN_PROCESS_COUNT='0'}.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key,'Process')
        [Environment]::SetEnvironmentVariable($entry.Key,$entry.Value,'Process')
    }
    $arguments = @('-I','-B',$helper,'--project-root',$root,'--source-lock-token',$token,'--profile',$RuntimeProfile,'--cuda-runtime',$CudaRuntime)
    if ($InstallDependencies) { $arguments += '--install-dependencies' }
    if ($NoNetwork) { $arguments += '--no-network' }
    if ($WheelhousePath) { $arguments += @('--wheelhouse',$WheelhousePath) }
    if ($PreserveVenvConfiguration) { $arguments += '--preserve-configuration' }
    & $basePython @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Python runtime setup failed. Existing environment recovery evidence is retained for retry.' }
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process') }
    if ($ownsLock) {
        try {
            if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
                $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($owner.token -eq $token) { Remove-Item -LiteralPath (Assert-SetupPath $ownerPath) -Force }
            }
        } finally { if ($null -ne $lock) { $lock.Dispose() } }
    }
}
