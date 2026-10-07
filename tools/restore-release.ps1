[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ArchiveIndex,
    [Parameter(Mandatory = $true)][string]$Destination
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1') -Force
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Management/Microsoft.PowerShell.Management.psd1') -Force
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Archive/Microsoft.PowerShell.Archive.psd1') -Force

# This script reads the published manifest/index and checksums, joins ordinary
# binary parts when needed, then expands the ZIP. It does not run the program.
$indexPath = (Resolve-Path -LiteralPath $ArchiveIndex).Path
$partsDirectory = [IO.Path]::GetFullPath((Split-Path -Parent $indexPath)).TrimEnd('\', '/')
$destinationPath = [IO.Path]::GetFullPath($Destination)
$temporaryArchive = $null

function Get-ArtifactPath([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name) -or $Name -match '[/\\:]' -or
        $Name -eq '.' -or $Name -eq '..' -or [IO.Path]::GetFileName($Name) -ne $Name) {
        throw "Unsafe artifact filename: $Name"
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $partsDirectory $Name))
    if (-not $candidate.StartsWith($partsDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact escaped the release directory: $Name"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Missing release artifact: $Name"
    }
    $item = Get-Item -LiteralPath $candidate -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Release artifacts must be ordinary files: $Name"
    }
    return $candidate
}

function Assert-Hash([string]$Path, [string]$Expected) {
    if ($Expected -cnotmatch '^[0-9a-f]{64}$') { throw "Invalid SHA-256 metadata: $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -cne $Expected) { throw "SHA-256 mismatch: $Path" }
}

$checksumPath = Get-ArtifactPath 'SHA256SUMS'
$checksums = @{}
foreach ($line in (Get-Content -LiteralPath $checksumPath -Encoding UTF8)) {
    if ($line -notmatch '^([0-9a-f]{64})  ([^/\\:]+)$') { throw 'Malformed SHA256SUMS line' }
    $filename = $Matches[2]
    if ($checksums.ContainsKey($filename)) { throw "Duplicate SHA256SUMS entry: $filename" }
    $checksums[$filename] = $Matches[1]
}
$indexName = Split-Path -Leaf $indexPath
if (-not $checksums.ContainsKey($indexName)) { throw "SHA256SUMS does not cover $indexName" }
Assert-Hash $indexPath $checksums[$indexName]
$index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($index.schemaVersion -ne 1 -or @($index.parts).Count -eq 0) { throw 'Unsupported or empty archive index' }
if ($index.archiveFile -match '[/\\:]' -or -not $index.archiveFile.EndsWith('.zip', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Invalid archive filename'
}
$manifestPath = Get-ArtifactPath $index.manifestFile
if (-not $checksums.ContainsKey([string]$index.manifestFile) -or $checksums[[string]$index.manifestFile] -cne $index.manifestSha256) {
    throw 'Manifest is not covered consistently by SHA256SUMS'
}
Assert-Hash $manifestPath $index.manifestSha256

$partPaths = @()
$partNames = @{}
$totalSize = [long]0
foreach ($part in @($index.parts)) {
    $partName = [string]$part.file
    if ($partNames.ContainsKey($partName)) { throw "Duplicate archive part: $partName" }
    $partNames[$partName] = $true
    $partPath = Get-ArtifactPath $partName
    $size = [long](Get-Item -LiteralPath $partPath).Length
    if ($size -ne [long]$part.size -or $size -le 0) { throw "Part size mismatch: $partName" }
    if (-not $checksums.ContainsKey($partName) -or $checksums[$partName] -cne $part.sha256) {
        throw "Archive part is not covered consistently by SHA256SUMS: $partName"
    }
    Assert-Hash $partPath $part.sha256
    $partPaths += $partPath
    $totalSize += $size
}
if ($totalSize -ne [long]$index.archiveSize) { throw 'Combined archive size mismatch' }
if ((Test-Path -LiteralPath $destinationPath) -and
    (-not (Test-Path -LiteralPath $destinationPath -PathType Container) -or
     @(Get-ChildItem -LiteralPath $destinationPath -Force).Count -ne 0)) {
    throw 'Destination must be a new or empty directory; existing files are never overwritten'
}

try {
    if ($partPaths.Count -eq 1 -and (Split-Path -Leaf $partPaths[0]) -eq $index.archiveFile) {
        $zipPath = $partPaths[0]
    } else {
        $temporaryArchive = Join-Path $partsDirectory ('.release-reassembly-' + [Guid]::NewGuid().ToString('N') + '.zip')
        $zipPath = $temporaryArchive
        $outStream = [IO.File]::Open($zipPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            foreach ($partPath in $partPaths) {
                $inStream = [IO.File]::OpenRead($partPath)
                try { $inStream.CopyTo($outStream, 1048576) } finally { $inStream.Dispose() }
            }
        } finally { $outStream.Dispose() }
    }
    Assert-Hash $zipPath $index.archiveSha256
    # Verify exact ZIP paths and embedded metadata before extracting anything.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $archiveRoot = [string]$manifest.archiveRoot
    if ($archiveRoot -notmatch '^[A-Za-z0-9._-]+$' -or $archiveRoot -eq '.' -or $archiveRoot -eq '..') {
        throw 'Invalid archive root in manifest'
    }
    $expected = @{}
    foreach ($file in @($manifest.files)) {
        $relativeName = [string]$file.path
        if ($relativeName -match '(^/|\\|:|(^|/)\.\.(/|$)|(^|/)\.(/|$))') { throw "Unsafe manifest path: $relativeName" }
        $entryName = $archiveRoot + '/' + $relativeName
        if ($expected.ContainsKey($entryName)) { throw "Duplicate manifest path: $entryName" }
        $expected[$entryName] = [long]$file.size
    }
    $expected[$archiveRoot + '/RELEASE-MANIFEST.json'] = [long](Get-Item -LiteralPath $manifestPath).Length
    $expected[$archiveRoot + '/PAYLOAD-SHA256SUMS'] = [long]-1
    $zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        if ($zip.Entries.Count -ne $expected.Count) { throw 'ZIP entry count differs from the file manifest' }
        $seen = @{}
        foreach ($entry in $zip.Entries) {
            $entryName = [string]$entry.FullName
            if (-not $expected.ContainsKey($entryName) -or $seen.ContainsKey($entryName)) { throw "Unexpected or duplicate ZIP path: $entryName" }
            $seen[$entryName] = $true
            if ($expected[$entryName] -ge 0 -and [long]$entry.Length -ne $expected[$entryName]) { throw "ZIP file size differs: $entryName" }
        }
        $embeddedEntry = $zip.GetEntry($archiveRoot + '/RELEASE-MANIFEST.json')
        $reader = [IO.StreamReader]::new($embeddedEntry.Open(), [Text.Encoding]::UTF8)
        try { $embeddedText = $reader.ReadToEnd() } finally { $reader.Dispose() }
        if ($embeddedText -cne [IO.File]::ReadAllText($manifestPath, [Text.Encoding]::UTF8)) { throw 'Embedded/external manifests differ' }
        if (-not (Test-Path -LiteralPath $destinationPath -PathType Container)) {
            New-Item -ItemType Directory -Path $destinationPath | Out-Null
        }
        $destinationPrefix = $destinationPath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        $extracted = 0
        foreach ($entry in $zip.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $destinationPath ($entry.FullName.Replace('/', [IO.Path]::DirectorySeparatorChar))))
            if (-not $target.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "ZIP path escaped the destination: $($entry.FullName)"
            }
            $parent = Split-Path -Parent $target
            if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            # Stream ZIP64 contents into new files. Never load an entire large
            # archive/entry into memory or overwrite files created meanwhile.
            $fileStream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $entryStream = $entry.Open()
                try { $entryStream.CopyTo($fileStream, 1048576) } finally { $entryStream.Dispose() }
            } finally { $fileStream.Dispose() }
            if ([long](Get-Item -LiteralPath $target).Length -ne [long]$entry.Length) {
                throw "Extracted file size differs: $($entry.FullName)"
            }
            $extracted++
            if ($extracted % 5000 -eq 0) { Write-Host "Expanded $extracted / $($zip.Entries.Count) files" }
        }
    } finally { $zip.Dispose() }
    Write-Host "Verified and expanded to: $destinationPath"
} finally {
    if ($temporaryArchive -and (Test-Path -LiteralPath $temporaryArchive -PathType Leaf)) {
        $cleanupPath = [IO.Path]::GetFullPath($temporaryArchive)
        if (-not $cleanupPath.StartsWith($partsDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            -not (Split-Path -Leaf $cleanupPath).StartsWith('.release-reassembly-', [StringComparison]::Ordinal)) {
            throw 'Refusing to remove an unexpected reassembly file'
        }
        Remove-Item -LiteralPath $cleanupPath -Force
    }
}
