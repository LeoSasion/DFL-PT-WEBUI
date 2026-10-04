[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$SkipUiBuild,
    [switch]$NoDownload
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$WebView2Version = "1.0.4129.50"
$WebView2Sha256 = "D3934F482D484B89FB4825DF720C710664E1143A1E90F7B3A60794EF33F473D2"
$PackageName = "microsoft.web.webview2.$WebView2Version.nupkg"
$PackageUrl = "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/$WebView2Version/$PackageName"

$LauncherRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $LauncherRoot ".."))
$HostRoot = Join-Path $LauncherRoot "host"
$VendorRoot = Join-Path $LauncherRoot "vendor"
$PackagePath = Join-Path $VendorRoot $PackageName
$SdkRoot = Join-Path $VendorRoot ("webview2\" + $WebView2Version)
$FrameworkRoot = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319"
$Compiler = Join-Path $FrameworkRoot "csc.exe"
$ManifestPath = Join-Path $HostRoot "app.manifest"
$IconPath = Join-Path $HostRoot "brand-mark.ico"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $LauncherRoot "bin"
} else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
}

function Write-Step {
    param([string]$Text)
    Write-Host "[launcher] $Text" -ForegroundColor Cyan
}

function Test-StrictChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Parent
    )

    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    if ([String]::Equals($candidateFull, $parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    return $candidateFull.StartsWith(
        $parentFull + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)
}

function Assert-PrivateStagingPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $full = [IO.Path]::GetFullPath($Path)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not (Test-StrictChildPath -Candidate $full -Parent $LauncherRoot) -and
        -not (Test-StrictChildPath -Candidate $full -Parent $tempRoot)) {
        throw "Unsafe launcher build staging path: $full"
    }
    return $full
}

function Remove-PrivateStagingDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $full = Assert-PrivateStagingPath -Path $Path
    if (Test-Path -LiteralPath $full -PathType Leaf) {
        throw "Launcher build staging path unexpectedly became a file: $full"
    }
    if (Test-Path -LiteralPath $full -PathType Container) {
        $item = Get-Item -LiteralPath $full -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to remove a reparse-point staging directory: $full"
        }
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}

function Publish-LauncherExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$StagedExecutable,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $stagedFull = [IO.Path]::GetFullPath($StagedExecutable)
    if (-not (Test-Path -LiteralPath $stagedFull -PathType Leaf)) {
        throw "Staged launcher executable is missing: $stagedFull"
    }
    $destinationFull = [IO.Path]::GetFullPath($Destination)
    $destinationDirectory = Split-Path -Parent $destinationFull
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $destinationFull -PathType Container) {
        throw "Launcher destination is a directory: $destinationFull"
    }
    if (Test-Path -LiteralPath $destinationFull -PathType Leaf) {
        $destinationItem = Get-Item -LiteralPath $destinationFull -Force
        if ($destinationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to replace a reparse-point launcher destination: $destinationFull"
        }
    }

    $publishingPath = Join-Path $destinationDirectory (
        [IO.Path]::GetFileName($destinationFull) + ".publishing-" + [Guid]::NewGuid().ToString("N") + ".tmp")
    $backupPath = Join-Path $destinationDirectory (
        [IO.Path]::GetFileName($destinationFull) + ".backup-" + [Guid]::NewGuid().ToString("N") + ".tmp")
    try {
        Copy-Item -LiteralPath $stagedFull -Destination $publishingPath
        $stagedHash = (Get-FileHash -LiteralPath $stagedFull -Algorithm SHA256).Hash
        $publishingHash = (Get-FileHash -LiteralPath $publishingPath -Algorithm SHA256).Hash
        if (-not [String]::Equals($stagedHash, $publishingHash, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Launcher executable hash changed during publication."
        }
        if (Test-Path -LiteralPath $destinationFull -PathType Leaf) {
            [IO.File]::Replace($publishingPath, $destinationFull, $backupPath, $true)
        } else {
            [IO.File]::Move($publishingPath, $destinationFull)
        }
    } finally {
        if (Test-Path -LiteralPath $publishingPath -PathType Leaf) {
            Remove-Item -LiteralPath $publishingPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
            Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-FrameworkReference {
    param([Parameter(Mandatory = $true)][string]$Name)

    $referenceRoots = @()
    if (-not [String]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
        $referenceRoots += Join-Path ${env:ProgramFiles(x86)} "Reference Assemblies\Microsoft\Framework\.NETFramework\v4.8"
    }
    if (-not [String]::IsNullOrWhiteSpace($env:ProgramFiles)) {
        $referenceRoots += Join-Path $env:ProgramFiles "Reference Assemblies\Microsoft\Framework\.NETFramework\v4.8"
    }
    $referenceRoots += $FrameworkRoot
    $referenceRoots += Join-Path $FrameworkRoot "WPF"

    foreach ($root in $referenceRoots) {
        $candidate = Join-Path $root $Name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }

    $assemblyName = [IO.Path]::GetFileNameWithoutExtension($Name)
    $gacRoots = @(
        (Join-Path $env:WINDIR "Microsoft.NET\assembly\GAC_MSIL"),
        (Join-Path $env:WINDIR "Microsoft.NET\assembly\GAC_64")
    )
    foreach ($gacRoot in $gacRoots) {
        $assemblyRoot = Join-Path $gacRoot $assemblyName
        if (-not (Test-Path -LiteralPath $assemblyRoot -PathType Container)) {
            continue
        }
        $candidate = Get-ChildItem -LiteralPath $assemblyRoot -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($null -ne $candidate) {
            return $candidate.FullName
        }
    }

    throw ".NET Framework reference assembly was not found: $Name"
}

function Assert-PackageHash {
    param([string]$Path)
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $WebView2Sha256) {
        throw "WebView2 SDK SHA-256 mismatch. Expected $WebView2Sha256, got $actual."
    }
}

function Find-Node {
    $candidates = @(
        (Join-Path $RepositoryRoot "_internal\node\bin\node.exe"),
        (Join-Path $env:ProgramFiles "nodejs\node.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $fromPath = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($null -ne $fromPath) {
        return $fromPath.Source
    }
    return $null
}

$outputTarget = $OutputDirectory.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$protectedOutputTargets = @(
    ([IO.Path]::GetPathRoot($OutputDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)),
    ($env:WINDIR.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)),
    ($env:ProgramFiles.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)),
    (${env:ProgramFiles(x86)}.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)),
    ($RepositoryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)),
    ($LauncherRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar))
)
foreach ($protectedTarget in $protectedOutputTargets) {
    if (-not [String]::IsNullOrWhiteSpace($protectedTarget) -and
        [String]::Equals($outputTarget, $protectedTarget, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe launcher output directory: $OutputDirectory"
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "The DFL-PT-WEBUI launcher requires 64-bit Windows."
}
if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw ".NET Framework 4.8 x64 compiler was not found: $Compiler"
}
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf) -or -not (Test-Path -LiteralPath $IconPath -PathType Leaf)) {
    throw "Launcher manifest or icon is missing under $HostRoot."
}

New-Item -ItemType Directory -Path $VendorRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
    if ($NoDownload) {
        throw "WebView2 SDK package is missing and -NoDownload was specified: $PackagePath"
    }
    Write-Step "Downloading official WebView2 SDK $WebView2Version..."
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $PackageUrl -OutFile $PackagePath
}

Write-Step "Verifying WebView2 SDK SHA-256..."
Assert-PackageHash -Path $PackagePath

$CoreAssembly = Join-Path $SdkRoot "lib\net462\Microsoft.Web.WebView2.Core.dll"
$WpfAssembly = Join-Path $SdkRoot "lib\net462\Microsoft.Web.WebView2.Wpf.dll"
$Loader = Join-Path $SdkRoot "runtimes\win-x64\native\WebView2Loader.dll"
if (-not (Test-Path -LiteralPath $CoreAssembly -PathType Leaf) -or
    -not (Test-Path -LiteralPath $WpfAssembly -PathType Leaf) -or
    -not (Test-Path -LiteralPath $Loader -PathType Leaf)) {
    Write-Step "Extracting WebView2 SDK..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $staging = Join-Path $VendorRoot ("webview2-extract-" + [Guid]::NewGuid().ToString("N"))
    try {
        [IO.Compression.ZipFile]::ExtractToDirectory($PackagePath, $staging)
        $sdkParent = Split-Path -Parent $SdkRoot
        New-Item -ItemType Directory -Path $sdkParent -Force | Out-Null
        if (Test-Path -LiteralPath $SdkRoot) {
            Remove-Item -LiteralPath $SdkRoot -Recurse -Force
        }
        Move-Item -LiteralPath $staging -Destination $SdkRoot
        $staging = $null
    } finally {
        if ($null -ne $staging -and (Test-Path -LiteralPath $staging)) {
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

$UiRuntimePackages = @(
    [PSCustomObject]@{ Name = 'react'; Version = '19.2.0' },
    [PSCustomObject]@{ Name = 'react-dom'; Version = '19.2.0' },
    [PSCustomObject]@{ Name = 'scheduler'; Version = '0.27.0' },
    [PSCustomObject]@{ Name = '@tabler/icons-react'; Version = '3.35.0' },
    [PSCustomObject]@{ Name = '@xterm/xterm'; Version = '6.0.0' },
    [PSCustomObject]@{ Name = '@xterm/addon-fit'; Version = '0.11.0' }
)
# npm lockfiles include a packages[""] root entry. Windows PowerShell 5.1's
# PSCustomObject JSON converter cannot represent an empty property name.
Add-Type -AssemblyName System.Web.Extensions
$uiLockReader = New-Object System.Web.Script.Serialization.JavaScriptSerializer
$uiLockReader.MaxJsonLength = 8388608
$uiLock = $uiLockReader.DeserializeObject([IO.File]::ReadAllText((Join-Path $LauncherRoot 'ui\package-lock.json')))
$uiRuntimeLicenseSources = @()
$uiRuntimeInputFiles = @()
foreach ($runtimePackage in $UiRuntimePackages) {
    $packageRoot = Join-Path (Join-Path $LauncherRoot 'ui\node_modules') $runtimePackage.Name
    $packageManifest = Join-Path $packageRoot 'package.json'
    $license = Join-Path $packageRoot 'LICENSE'
    if (-not (Test-Path -LiteralPath $packageManifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $license -PathType Leaf)) {
        throw "Missing required bundled UI package or license: $($runtimePackage.Name)."
    }
    $metadata = Get-Content -LiteralPath $packageManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    $locked = $uiLock['packages'][('node_modules/' + $runtimePackage.Name)]
    if ($null -eq $locked -or $metadata.name -ne $runtimePackage.Name -or
        $metadata.version -ne $runtimePackage.Version -or $metadata.license -ne 'MIT' -or
        $locked['version'] -ne $runtimePackage.Version -or $locked['license'] -ne 'MIT') {
        throw "Bundled UI package does not match its pinned MIT dependency: $($runtimePackage.Name)."
    }
    $licenseText = Get-Content -LiteralPath $license -Raw -Encoding UTF8
    if ($licenseText.Length -lt 700 -or $licenseText.Length -gt 65536 -or
        $licenseText.Substring(0, [Math]::Min(2048, $licenseText.Length)) -notmatch '(?i)copyright\s+\(c\)' -or
        -not $licenseText.Contains('Permission is hereby granted, free of charge') -or
        -not $licenseText.Contains('THE SOFTWARE IS PROVIDED "AS IS"')) {
        throw "Bundled UI license is missing its MIT text or copyright notice: $($runtimePackage.Name)."
    }
    $uiRuntimeInputFiles += (Get-Item -LiteralPath $packageManifest), (Get-Item -LiteralPath $license)
    $uiRuntimeLicenseSources += [PSCustomObject]@{
        RelativePath = 'licenses/UI/' + $runtimePackage.Name + '.txt'
        FilePath = $license
    }
}

function Get-LauncherBuildSources {
    $buildSourceFiles = @(Get-ChildItem -LiteralPath $HostRoot -File) + @(
        (Get-Item -LiteralPath $PSCommandPath),
        (Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\package.json')),
        (Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\package-lock.json')),
        (Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\vite.config.mjs')),
        (Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\index.html')),
        (Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\scripts\prepare-sites-build.mjs')))
    $buildSourceFiles += $uiRuntimeInputFiles
    if (Test-Path -LiteralPath (Join-Path $LauncherRoot 'ui\.npmrc')) {
        $buildSourceFiles += Get-Item -LiteralPath (Join-Path $LauncherRoot 'ui\.npmrc')
    }
    foreach ($sourceRoot in @((Join-Path $LauncherRoot 'ui\src'), (Join-Path $LauncherRoot 'ui\public'))) {
        if (Test-Path -LiteralPath $sourceRoot) {
            $buildSourceFiles += Get-ChildItem -LiteralPath $sourceRoot -File -Recurse
        }
    }
    return @($buildSourceFiles | Sort-Object FullName -Unique | ForEach-Object {
        [PSCustomObject]@{
            path = $_.FullName.Substring($RepositoryRoot.TrimEnd('\', '/').Length + 1).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    })
}
# Bind both the UI compiler and the host compiler to the same source snapshot.
# A reused dist can still be built for local diagnostics, but its receipt must
# never claim that this invocation compiled the corresponding UI sources.
$buildSources = Get-LauncherBuildSources
$buildSourceSnapshot = $buildSources | ConvertTo-Json -Depth 4 -Compress
$uiBuildPerformed = $false
if (-not $SkipUiBuild) {
    $UiRoot = Join-Path $LauncherRoot "ui"
    $ViteEntry = Join-Path $UiRoot "node_modules\vite\bin\vite.js"
    $Node = Find-Node
    if ($null -eq $Node) {
        throw "Node.js was not found. Install the project-local runtime or use -SkipUiBuild with an existing UI build."
    }
    if (-not (Test-Path -LiteralPath $ViteEntry -PathType Leaf)) {
        throw "Launcher UI dependencies are missing: $ViteEntry"
    }
    Write-Step "Building launcher UI..."
    Push-Location $UiRoot
    try {
        & $Node $ViteEntry build --configLoader runner
        if ($LASTEXITCODE -ne 0) {
            throw "Launcher UI build failed with exit code $LASTEXITCODE."
        }
        if ((Get-LauncherBuildSources | ConvertTo-Json -Depth 4 -Compress) -ne $buildSourceSnapshot) {
            throw 'Launcher sources changed during the UI build; the previous executable was preserved.'
        }
        $uiBuildPerformed = $true
    } finally {
        Pop-Location
    }
}

$UiBuild = Join-Path $LauncherRoot "ui\dist\client"
if (-not (Test-Path -LiteralPath (Join-Path $UiBuild "index.html") -PathType Leaf)) {
    throw "Launcher UI build is missing: $UiBuild\index.html"
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$OutputExe = Join-Path $OutputDirectory "DFL-PT-WEBUI.Launcher.exe"
$BootstrapScript = Join-Path $LauncherRoot "bootstrap.ps1"
$RuntimeManifest = Join-Path $LauncherRoot "runtime-manifest.json"
$RuntimeSetup = Join-Path $LauncherRoot "setup-runtime.ps1"
$ProjectSetup = Join-Path $LauncherRoot "install-project.ps1"
$SourceSetup = Join-Path $LauncherRoot "install-source.ps1"
$TerminalBridgeEntry = Join-Path $LauncherRoot "server\index.mjs"
$TerminalBridgeCore = Join-Path $LauncherRoot "server\terminal-bridge.mjs"
if (-not (Test-Path -LiteralPath $BootstrapScript -PathType Leaf) -or
    -not (Test-Path -LiteralPath $RuntimeManifest -PathType Leaf) -or
    -not (Test-Path -LiteralPath $RuntimeSetup -PathType Leaf) -or
    -not (Test-Path -LiteralPath $ProjectSetup -PathType Leaf) -or
    -not (Test-Path -LiteralPath $SourceSetup -PathType Leaf)) {
    throw "Launcher local runtime setup files are incomplete."
}
if (-not (Test-Path -LiteralPath $TerminalBridgeEntry -PathType Leaf) -or
    -not (Test-Path -LiteralPath $TerminalBridgeCore -PathType Leaf)) {
    throw "Launcher terminal bridge files are incomplete."
}

$sourceFiles = @(Get-ChildItem -LiteralPath $HostRoot -Filter *.cs -File | Sort-Object Name | ForEach-Object FullName)
if ($sourceFiles.Count -eq 0) {
    throw "No C# source files were found under $HostRoot."
}

$references = @(
    (Resolve-FrameworkReference "System.dll"),
    (Resolve-FrameworkReference "System.Core.dll"),
    (Resolve-FrameworkReference "System.Drawing.dll"),
    (Resolve-FrameworkReference "System.Web.dll"),
    (Resolve-FrameworkReference "System.Web.Extensions.dll"),
    (Resolve-FrameworkReference "System.Windows.Forms.dll"),
    (Resolve-FrameworkReference "System.Xaml.dll"),
    (Resolve-FrameworkReference "WindowsBase.dll"),
    (Resolve-FrameworkReference "PresentationCore.dll"),
    (Resolve-FrameworkReference "PresentationFramework.dll"),
    $CoreAssembly,
    $WpfAssembly
)

$payloadSources = @(
    [PSCustomObject]@{ RelativePath = "Microsoft.Web.WebView2.Core.dll"; FilePath = $CoreAssembly },
    [PSCustomObject]@{ RelativePath = "Microsoft.Web.WebView2.Wpf.dll"; FilePath = $WpfAssembly },
    [PSCustomObject]@{ RelativePath = "WebView2Loader.dll"; FilePath = $Loader },
    [PSCustomObject]@{ RelativePath = "bootstrap/bootstrap.ps1"; FilePath = $BootstrapScript },
    [PSCustomObject]@{ RelativePath = "bootstrap/runtime-manifest.json"; FilePath = $RuntimeManifest },
    [PSCustomObject]@{ RelativePath = "bootstrap/setup-runtime.ps1"; FilePath = $RuntimeSetup },
    [PSCustomObject]@{ RelativePath = "bootstrap/install-project.ps1"; FilePath = $ProjectSetup },
    [PSCustomObject]@{ RelativePath = "bootstrap/install-source.ps1"; FilePath = $SourceSetup },
    [PSCustomObject]@{ RelativePath = "licenses/WebView2/LICENSE.txt"; FilePath = (Join-Path $SdkRoot "LICENSE.txt") },
    [PSCustomObject]@{ RelativePath = "licenses/WebView2/NOTICE.txt"; FilePath = (Join-Path $SdkRoot "NOTICE.txt") },
    [PSCustomObject]@{ RelativePath = "terminal/index.mjs"; FilePath = $TerminalBridgeEntry },
    [PSCustomObject]@{ RelativePath = "terminal/terminal-bridge.mjs"; FilePath = $TerminalBridgeCore }
)
$payloadSources += $uiRuntimeLicenseSources
$uiPrefixLength = $UiBuild.TrimEnd('\', '/').Length + 1
foreach ($uiFile in Get-ChildItem -LiteralPath $UiBuild -Recurse -File | Sort-Object FullName) {
    $uiRelative = $uiFile.FullName.Substring($uiPrefixLength).Replace('\', '/')
    $payloadSources += [PSCustomObject]@{
        RelativePath = "ui/" + $uiRelative
        FilePath = $uiFile.FullName
    }
}

$payloadEntries = @()
$resourceIndex = 0
foreach ($source in $payloadSources | Sort-Object RelativePath) {
    if ($source.RelativePath.IndexOf("`t") -ge 0 -or $source.RelativePath.IndexOf("`n") -ge 0 -or
        -not (Test-Path -LiteralPath $source.FilePath -PathType Leaf)) {
        throw "Invalid launcher payload source: $($source.RelativePath)"
    }
    $file = Get-Item -LiteralPath $source.FilePath
    $payloadEntries += [PSCustomObject]@{
        RelativePath = $source.RelativePath
        FilePath = $file.FullName
        Length = [Int64]$file.Length
        Sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        ResourceName = "DflPtWebUi.Launcher.Payload.File." + $resourceIndex.ToString("D4")
    }
    $resourceIndex++
}

$buildMaterial = ($payloadEntries | ForEach-Object {
    $_.RelativePath + "`t" + $_.Length.ToString([Globalization.CultureInfo]::InvariantCulture) + "`t" + $_.Sha256
}) -join "`n"
$buildHasher = [Security.Cryptography.SHA256]::Create()
try {
    $buildHashBytes = $buildHasher.ComputeHash((New-Object Text.UTF8Encoding($false)).GetBytes($buildMaterial))
} finally {
    $buildHasher.Dispose()
}
$BuildId = ([BitConverter]::ToString($buildHashBytes).Replace("-", "").ToLowerInvariant()).Substring(0, 24)

$payloadTempRoot = Assert-PrivateStagingPath -Path (Join-Path ([IO.Path]::GetTempPath()) ("dflsn-launcher-build-" + [Guid]::NewGuid().ToString("N")))
New-Item -ItemType Directory -Path $payloadTempRoot -Force | Out-Null
$payloadTempRoot = Assert-PrivateStagingPath -Path ((Resolve-Path -LiteralPath $payloadTempRoot).Path)
$PayloadManifestPath = Join-Path $payloadTempRoot "payload-manifest.txt"
$StagedOutputExe = Join-Path $payloadTempRoot "DFL-PT-WEBUI.Launcher.exe"
try {
    $manifestLines = New-Object 'System.Collections.Generic.List[string]'
    $manifestLines.Add("DFLSN_PAYLOAD_V1`t$BuildId")
    foreach ($entry in $payloadEntries) {
        $manifestLines.Add(
            "F`t" + $entry.RelativePath + "`t" +
            $entry.Length.ToString([Globalization.CultureInfo]::InvariantCulture) + "`t" +
            $entry.Sha256 + "`t" + $entry.ResourceName)
    }
    [IO.File]::WriteAllLines($PayloadManifestPath, $manifestLines, (New-Object Text.UTF8Encoding($false)))

    $compilerArguments = @(
        "/nologo",
        # References are resolved above. csc.rsp would additionally load the
        # runtime assemblies and conflict with SDK reference assemblies on CI.
        "/noconfig",
        "/target:winexe",
        "/platform:x64",
        "/optimize+",
        "/debug-",
        "/langversion:5",
        "/utf8output",
        "/out:$StagedOutputExe",
        "/win32manifest:$ManifestPath",
        "/win32icon:$IconPath",
        "/resource:$PayloadManifestPath,DflPtWebUi.Launcher.Payload.Manifest"
    )
    $compilerArguments += $references | ForEach-Object { "/reference:$_" }
    $compilerArguments += $payloadEntries | ForEach-Object { "/resource:$($_.FilePath),$($_.ResourceName)" }
    $compilerArguments += $sourceFiles


    Write-Step "Compiling single-file .NET Framework 4.8 WPF launcher..."
    & $Compiler $compilerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "C# compilation failed with exit code $LASTEXITCODE."
    }
    if ((Get-LauncherBuildSources | ConvertTo-Json -Depth 4 -Compress) -ne $buildSourceSnapshot) {
        throw 'Launcher sources changed during compilation; the previous executable was preserved.'
    }
    Publish-LauncherExecutable -StagedExecutable $StagedOutputExe -Destination $OutputExe
} finally {
    Remove-PrivateStagingDirectory -Path $payloadTempRoot
}

if (-not (Test-Path -LiteralPath $OutputExe -PathType Leaf)) {
    throw "Launcher output executable was not published: $OutputExe"
}

# Record compiler inputs separately from embedded UI/runtime resources so the
# standalone release verifier can reject an executable built from stale host code.
$buildReceipt = [PSCustomObject]@{
    schemaVersion = 1
    product = 'DFL-PT-WEBUI Launcher'
    executable = [PSCustomObject]@{
        file = [IO.Path]::GetFileName($OutputExe)
        sha256 = (Get-FileHash -LiteralPath $OutputExe -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes = (Get-Item -LiteralPath $OutputExe).Length
    }
    payloadBuildId = $BuildId
    uiBuildPerformed = $uiBuildPerformed
    sourceSnapshotBeforeUiBuild = $true
    sources = $buildSources
}
[IO.File]::WriteAllText((Join-Path $OutputDirectory 'launcher-build.json'),
    ($buildReceipt | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
if ([String]::Equals($OutputDirectory.TrimEnd('\', '/'), (Join-Path $LauncherRoot 'bin').TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
    Publish-LauncherExecutable -StagedExecutable $OutputExe -Destination (Join-Path $RepositoryRoot 'DFL-PT-WEBUI.exe')
}

Write-Step "Built single EXE: $OutputExe"
Write-Step "Payload build-id: $BuildId"
