[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$NoNetwork,
    [string]$ArchiveDirectory = '',
    [string]$WheelhousePath = '',
    [switch]$SkipVisionAssets,
    [switch]$SkipWebuiBuild,
    [switch]$SkipWebuiPreparation
)

# Windows PowerShell 5.1 / PowerShell 7. No global Python, npm or PATH changes.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$rootPrefix = $root + [IO.Path]::DirectorySeparatorChar
$cache = Join-Path $root '.launcher-install\source'
$session = Join-Path $cache ('installing-' + [Guid]::NewGuid().ToString('N'))
$lock = $null
$createdVenv = $false
$createdModules = $false
$createdDist = $false
$succeeded = $false
$savedEnvironment = @{}
$base = Join-Path $root '_internal\python_base'
$basePython = Join-Path $base 'python.exe'
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$nodeRoot = Join-Path $root '_internal\node'
$nodeBin = Join-Path $nodeRoot 'bin'
$node = Join-Path $nodeBin 'node.exe'
$ffmpegRoot = Join-Path $root '_internal\ffmpeg'
$webui = Join-Path $root 'webui'
$modules = Join-Path $webui 'node_modules'
$dist = Join-Path $webui 'dist'
$requirements = Join-Path $root 'requirements.txt'

# Digests are pinned from upstream release metadata / Node SHASUMS256.txt.
$pythonArchive = 'cpython-3.12.14+20260814-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
$pythonUrl = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260814/cpython-3.12.14%2B20260814-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
$pythonSha256 = '89f18f6932917163b74339ebcec2645c8e47ae7f1c5f2ac37f2b4f4cf3beb647'
$nodeVersion = '24.19.0'
$nodeArchive = "node-v$nodeVersion-win-x64.zip"
$nodeUrl = "https://nodejs.org/dist/v$nodeVersion/$nodeArchive"
$nodeSha256 = '57f71ab3652e797d84acddc79c81cc9ff1c6ddb2a1974cdb83f00fee9bff4c73'
$ffmpegArchive = 'ffmpeg-9.0.1-essentials_build.zip'
$ffmpegUrl = 'https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-essentials_build.zip'
$ffmpegSha256 = 'fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9'

function Assert-ProjectPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must stay inside the project: $absolute"
    }
    # Do not follow an old runtime junction or a linked staging directory.
    $ancestor = $absolute
    while ($ancestor -and $ancestor.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Linked runtime / install paths are not supported: $ancestor"
            }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $absolute
}

function Assert-Exit([string]$Operation) {
    if ($LASTEXITCODE -ne 0) { throw "$Operation failed (exit $LASTEXITCODE)." }
}

function Set-InstallEnvironment([string]$Name, [string]$Value) {
    if (-not $savedEnvironment.ContainsKey($Name)) {
        $savedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    }
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
}

function Get-VerifiedArchive([string]$Name, [string]$Url, [string]$Sha256) {
    $candidate = Join-Path $ArchiveDirectory $Name
    $projectCached = Assert-ProjectPath (Join-Path $cache ('archives\' + $Name))
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf) -and (Test-Path -LiteralPath $projectCached -PathType Leaf)) {
        $candidate = $projectCached
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        if ($NoNetwork) { throw "Offline archive missing: $candidate" }
        $download = Assert-ProjectPath (Join-Path $cache ('download-' + [Guid]::NewGuid().ToString('N') + '.partial'))
        Write-Host "[source-install] Downloading $Name from $Url"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $download
            $actual = (Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash
            if ($actual -ne $Sha256) { throw "SHA-256 mismatch for $Name. Expected $Sha256, got $actual." }
            # User-supplied offline directories are read-only inputs. Downloads
            # always go to the project cache, even when that directory is external.
            $candidate = $projectCached
            New-Item -ItemType Directory -Path (Split-Path -Parent $candidate) -Force | Out-Null
            if (Test-Path -LiteralPath $candidate) { throw "Archive appeared during installation; retry: $candidate" }
            Move-Item -LiteralPath $download -Destination $candidate
        } finally {
            if (Test-Path -LiteralPath $download) { Remove-Item -LiteralPath (Assert-ProjectPath $download) -Force }
        }
    }
    $actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) { throw "SHA-256 mismatch for $candidate. Expected $Sha256, got $actual. Nothing was installed from this archive." }
    return $candidate
}

function Publish-Runtime([string]$Source, [string]$Target) {
    $Source = Assert-ProjectPath $Source
    $Target = Assert-ProjectPath $Target
    if (Test-Path -LiteralPath $Target) { throw "Existing runtime will not be replaced: $Target" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
    Get-ChildItem -LiteralPath $Source -File -Recurse | Unblock-File -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $Source -Destination $Target
}

function Assert-BasePython([string]$Executable) {
    Assert-ProjectPath $Executable | Out-Null
    & $Executable -I -c "import sys,struct,venv,ensurepip,pathlib; assert sys.version_info[:2] == (3,12), sys.version; assert struct.calcsize('P') == 8; assert sys.prefix == sys.base_prefix; assert pathlib.Path(sys.base_prefix).resolve() == pathlib.Path(sys.executable).resolve().parent,sys.base_prefix; print('Python base: '+sys.version.split()[0])"
    Assert-Exit 'Project-local Python 3.12 base validation'
}

function Assert-Node([string]$Executable) {
    Assert-ProjectPath $Executable | Out-Null
    $actual = (& $Executable -p 'process.versions.node' | Select-Object -First 1)
    # PowerShell 5.1 can report -1 for Node 24 despite valid stdout.
    if ([string]$actual -ne $nodeVersion) { throw "Expected Node $nodeVersion, got '$actual'. Existing Node will not be replaced." }
}

$pythonValidation = @'
import pathlib,sys,importlib.metadata as metadata,re
root=pathlib.Path(sys.argv[1]).resolve()
assert sys.version_info[:2] == (3,12),sys.version
assert pathlib.Path(sys.base_prefix).resolve() == root/'_internal'/'python_base',sys.base_prefix
assert pathlib.Path(sys.prefix).resolve() == root/'.venv',sys.prefix
import torch,cv2,numpy,scipy
assert pathlib.Path(torch.__file__).resolve().is_relative_to(root/'.venv'),torch.__file__
assert torch.__version__ == '2.9.1+cu128',torch.__version__
assert torch.version.cuda == '12.8',torch.version.cuda
for line in pathlib.Path(sys.argv[2]).read_text(encoding='utf-8').splitlines():
    match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ #]+)',line.strip())
    if match:
        actual=metadata.version(match[1])
        assert actual == match[2],f'{match[1]}: expected {match[2]}, got {actual}'
        assert pathlib.Path(metadata.distribution(match[1]).locate_file('')).resolve().is_relative_to(root/'.venv'),match[1]
print('DFL_PT_RUNTIME_OK|'+sys.version.split()[0]+'|'+torch.__version__)
'@

try {
    if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\','/')) { throw 'ProjectRoot must be an independent source checkout directory.' }
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem -or
        $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw 'This source installer supports Windows x64 (Windows 10/11) only.'
    }
    foreach ($relative in @('_internal\DeepFaceLab\me.py', 'requirements.txt', 'webui\package.json',
        'webui\pnpm-lock.yaml', 'webui\pnpm-workspace.yaml', 'release\version.json', 'tools\prepare-vision-runtime.ps1')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) { throw "Not a complete source checkout: missing $relative" }
    }
    $runtimePaths = @($cache, $session, $base, $venv, $nodeRoot, $ffmpegRoot, $modules, $dist,
        (Join-Path $root '.launcher-install\vision'), (Join-Path $root '_internal\DeepFaceLab\facelib'),
        (Join-Path $root '_internal\vision_models'))
    foreach ($path in $runtimePaths) {
        Assert-ProjectPath $path | Out-Null
    }
    # Use this shell's modules, including when PS5 inherits PS7's module paths.
    foreach ($module in @('Microsoft.PowerShell.Utility', 'Microsoft.PowerShell.Archive', 'CimCmdlets')) {
        Import-Module (Join-Path $PSHOME ("Modules\$module\$module.psd1")) -Force
    }
    # No package installation or rebasing while this checkout is in use.
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='node.exe' OR Name='ffmpeg.exe' OR Name='ffprobe.exe'" |
        Where-Object {
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) -or
            ($_.CommandLine -and $_.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0)
        })
    if ($busy.Count -gt 0) { throw 'Project runtime processes are active. Stop WebUI and all project tasks, then rerun the installer.' }
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $lockPath = Assert-ProjectPath (Join-Path $cache 'source-install.lock')
    try { $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
    catch { throw 'Another source installation is active, or the project cache is not writable.' }
    New-Item -ItemType Directory -Path $session | Out-Null
    if (-not $ArchiveDirectory) { $ArchiveDirectory = Join-Path $cache 'archives' }
    $ArchiveDirectory = [IO.Path]::GetFullPath($ArchiveDirectory)
    if ($WheelhousePath) {
        $WheelhousePath = [IO.Path]::GetFullPath($WheelhousePath)
        if (-not (Test-Path -LiteralPath $WheelhousePath -PathType Container)) { throw "Wheelhouse not found: $WheelhousePath" }
    }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $versions = Get-Content -LiteralPath (Join-Path $root 'release\version.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $package = Get-Content -LiteralPath (Join-Path $webui 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($versions.nodeVersion -ne $nodeVersion -or $package.packageManager -ne ('pnpm@' + $versions.pnpmVersion)) {
        throw 'Runtime pins disagree with release/version.json or webui/package.json; update the source installer pins before installation.'
    }
    if (Test-Path -LiteralPath $base) {
        if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) { throw "Existing Python base is incomplete and will not be replaced: $base" }
        Assert-BasePython $basePython
    } else {
        $archive = Get-VerifiedArchive $pythonArchive $pythonUrl $pythonSha256
        $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
        if (-not (Test-Path -LiteralPath $tar -PathType Leaf)) { throw 'Windows tar.exe is required to extract the standalone Python archive. Install current Windows 10/11 updates.' }
        $expanded = Join-Path $session 'python'
        New-Item -ItemType Directory -Path $expanded | Out-Null
        & $tar -xzf $archive -C $expanded
        Assert-Exit 'Python archive extraction'
        $source = Join-Path $expanded 'python'
        Assert-BasePython (Join-Path $source 'python.exe')
        Publish-Runtime $source $base
    }
    if (Test-Path -LiteralPath $nodeRoot) {
        foreach ($relative in @('node.exe', 'npm.cmd', 'node_modules\corepack\dist\corepack.js')) {
            if (-not (Test-Path -LiteralPath (Join-Path $nodeBin $relative) -PathType Leaf)) { throw "Existing Node is incomplete and will not be replaced: $nodeRoot" }
        }
        Assert-Node $node
    } else {
        $archive = Get-VerifiedArchive $nodeArchive $nodeUrl $nodeSha256
        $expanded = Join-Path $session 'node-extracted'
        Expand-Archive -LiteralPath $archive -DestinationPath $expanded
        $source = Join-Path $expanded "node-v$nodeVersion-win-x64"
        Assert-Node (Join-Path $source 'node.exe')
        $staged = Join-Path $session 'node'
        New-Item -ItemType Directory -Path $staged | Out-Null
        Move-Item -LiteralPath (Assert-ProjectPath $source) -Destination (Assert-ProjectPath (Join-Path $staged 'bin'))
        Publish-Runtime $staged $nodeRoot
    }
    if (Test-Path -LiteralPath $ffmpegRoot) {
        foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
            if (-not (Test-Path -LiteralPath (Join-Path $ffmpegRoot $name) -PathType Leaf)) { throw "Existing FFmpeg is incomplete and will not be replaced: $ffmpegRoot" }
        }
    } else {
        $archive = Get-VerifiedArchive $ffmpegArchive $ffmpegUrl $ffmpegSha256
        $expanded = Join-Path $session 'ffmpeg-extracted'
        Expand-Archive -LiteralPath $archive -DestinationPath $expanded
        $source = Join-Path $expanded 'ffmpeg-9.0.1-essentials_build'
        $staged = Join-Path $session 'ffmpeg'
        New-Item -ItemType Directory -Path $staged | Out-Null
        foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
            Copy-Item -LiteralPath (Join-Path $source ('bin\' + $name)) -Destination $staged
        }
        foreach ($name in @('LICENSE', 'README.txt', 'doc', 'presets')) {
            Copy-Item -LiteralPath (Join-Path $source $name) -Destination $staged -Recurse
        }
        Publish-Runtime $staged $ffmpegRoot
    }
    foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
        $executable = Assert-ProjectPath (Join-Path $ffmpegRoot $name)
        & $executable -version
        Assert-Exit "$name validation"
    }
    if (Test-Path -LiteralPath $venv) {
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Existing .venv is incomplete and will not be replaced: $venv" }
        Assert-ProjectPath $python | Out-Null
        Write-Host '[source-install] Reusing existing .venv without changing pyvenv.cfg or installed packages.'
    } else {
        if ($NoNetwork -and -not $WheelhousePath) { throw 'Creating .venv offline requires -WheelhousePath with torch 2.9.1+cu128 and all requirements / dependencies.' }
        New-Item -ItemType Directory -Path $venv | Out-Null
        $createdVenv = $true
        & $basePython -I -m venv $venv
        Assert-Exit 'Project .venv creation'
        $pipOptions = @('--disable-pip-version-check', '--only-binary=:all:', '--cache-dir', (Join-Path $cache 'pip-cache'))
        if ($WheelhousePath) { $pipOptions += @('--no-index', '--find-links', $WheelhousePath) }
        $torchOptions = $pipOptions + @('torch==2.9.1+cu128')
        $requirementsOptions = $pipOptions + @('-r', $requirements)
        if (-not $WheelhousePath) {
            $torchOptions += @('--index-url', 'https://download.pytorch.org/whl/cu128')
            $requirementsOptions += @('--index-url', 'https://pypi.org/simple')
        }
        & $python -I -m pip --isolated install @torchOptions
        Assert-Exit 'Pinned PyTorch CUDA 12.8 wheel installation'
        & $python -I -m pip --isolated install @requirementsOptions
        Assert-Exit 'Pinned Python requirements installation'
    }
    & $python -I -c $pythonValidation $root $requirements
    Assert-Exit 'Independent Python / pinned requirements validation (existing environments are preserved)'
    & $python -I -m pip --isolated check
    Assert-Exit 'Python dependency consistency check'
    # Keep a fully installed venv if a later, independent asset step fails.
    $createdVenv = $false
    if (-not $SkipVisionAssets) {
        & (Join-Path $root 'tools\prepare-vision-runtime.ps1') -ProjectRoot $root -NoNetwork:$NoNetwork -SkipFFmpeg -PreserveExisting
        if (-not $?) { throw 'Preparing verified helper weights failed.' }
    } else { Write-Host '[source-install] Helper weights explicitly skipped; extraction / enhancement / grouping may be unavailable.' }
    if (-not $SkipWebuiPreparation) {
    Set-InstallEnvironment 'PATH' ($nodeBin + [IO.Path]::PathSeparator + $env:PATH)
    Set-InstallEnvironment 'COREPACK_HOME' (Join-Path $cache 'corepack')
    Set-InstallEnvironment 'COREPACK_ENABLE_NETWORK' $(if ($NoNetwork) { '0' } else { '1' })
    Set-InstallEnvironment 'COREPACK_ENABLE_DOWNLOAD_PROMPT' '0'
    Set-InstallEnvironment 'COREPACK_ENABLE_AUTO_PIN' '0'
    Set-InstallEnvironment 'COREPACK_NPM_REGISTRY' 'https://registry.npmjs.org'
    Set-InstallEnvironment 'npm_config_python' $basePython
    Set-InstallEnvironment 'npm_config_registry' 'https://registry.npmjs.org'
    Set-InstallEnvironment 'npm_config_disturl' 'https://nodejs.org/download/release'
    Set-InstallEnvironment 'npm_config_build_from_source' 'false'
    $corepack = Join-Path $nodeBin 'node_modules\corepack\dist\corepack.js'
    Push-Location $webui
    try {
        if (Test-Path -LiteralPath $modules) {
            Write-Host '[source-install] Reusing existing WebUI node_modules without reinstalling packages.'
        } else {
            # Offline source use accepts a complete preinstalled WebUI environment;
            # native dependency lifecycle scripts may otherwise fetch build inputs.
            if ($NoNetwork) { throw 'Offline WebUI setup requires existing webui/node_modules. Prepare it online in this checkout first, or use the complete portable package.' }
            New-Item -ItemType Directory -Path $modules | Out-Null
            $createdModules = $true
            & $node $corepack pnpm install --frozen-lockfile --registry=https://registry.npmjs.org --store-dir (Join-Path $cache 'pnpm-store')
            Assert-Exit 'Locked WebUI dependency installation (see runtime-README.md for node-pty build prerequisites)'
        }
        & $node -e "for (const name of ['react','vite','ws']) require.resolve(name); require('node-pty'); console.log('WebUI dependencies / node-pty native module ready');"
        Assert-Exit 'WebUI dependency / native module validation'
        $createdModules = $false
        if (-not $SkipWebuiBuild -and -not (Test-Path -LiteralPath (Join-Path $dist 'client\index.html') -PathType Leaf)) {
            if (Test-Path -LiteralPath $dist) { throw 'Existing webui/dist is incomplete and will not be replaced. Move it aside before explicitly rebuilding.' }
            New-Item -ItemType Directory -Path $dist | Out-Null
            $createdDist = $true
            # Run the locked build with local Node; no Corepack download is needed
            # when reusing an offline node_modules installation.
            & $node (Join-Path $modules 'vite\bin\vite.js') build --configLoader runner
            Assert-Exit 'WebUI production build'
            & $node (Join-Path $webui 'scripts\prepare-sites-build.mjs')
            Assert-Exit 'WebUI build preparation'
            & $node (Join-Path $root 'tools\dist-provenance.mjs') write
            Assert-Exit 'WebUI build provenance'
        }
    } finally { Pop-Location }
    } else {
        Write-Host '[source-install] WebUI dependency preparation and build delegated to launcher repair.'
    }
    $succeeded = $true
    Write-Host '[source-install] Source runtime ready. Start the root WebUI BAT when you want to run the application.'
} catch {
    Write-Error -Message ("Source installation stopped: " + $_.Exception.Message) -ErrorAction Continue
    exit 1
} finally {
    # Only remove directories created by this invocation; never an existing env.
    $cleanup = @($session)
    if (-not $succeeded) {
        if ($createdVenv) { $cleanup += $venv }
        if ($createdModules) { $cleanup += $modules }
        if ($createdDist) { $cleanup += $dist }
    }
    foreach ($path in $cleanup) {
        if (Test-Path -LiteralPath $path) {
            try { Remove-Item -LiteralPath (Assert-ProjectPath $path) -Recurse -Force }
            catch { Write-Warning "Could not clean installer-owned directory $path. Move it aside before retrying. $($_.Exception.Message)" }
        }
    }
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
    if ($null -ne $lock) { $lock.Dispose() }
}
