[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [switch]$NoNetwork,
    [switch]$SkipFFmpeg,
    [switch]$PreserveExisting,
    [string]$ResourcePackPath = '',
    [string]$VisionCacheRoot = '',
    [switch]$IncludeRestoration,
    [switch]$IncludeScene
)
$ErrorActionPreference = 'Stop'
# A Node process launched from PowerShell 7 can pass incompatible module paths
# to Windows PowerShell 5.1. Load the running shell's own bundled modules.
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1') -Force
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Archive/Microsoft.PowerShell.Archive.psd1') -Force
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$visionRoot = [IO.Path]::GetFullPath($ProjectRoot)
$cacheRoot = Join-Path $visionRoot '.launcher-install/vision'
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

function Resolve-VisionPath([string]$Relative) {
    $full = [IO.Path]::GetFullPath((Join-Path $visionRoot $Relative))
    if (-not $full.StartsWith($visionRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Visual dependency path escaped the project'
    }
    return $full
}
function Test-Hash([string]$File, [string]$Sha256) {
    return (Test-Path -LiteralPath $File -PathType Leaf) -and ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -eq $Sha256)
}
function Get-Verified([string]$Name, [string]$Url, [string]$Sha256) {
    $cached = Join-Path $cacheRoot $Name
    if (Test-Hash $cached $Sha256) { return $cached }
    if ($NoNetwork) { throw "Offline helper archive / weight missing or checksum mismatch: $cached" }
    $partial = $cached + '.' + [Guid]::NewGuid().ToString('N') + '.partial'
    Write-Host "Downloading $Name"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
    if (-not (Test-Hash $partial $Sha256)) { throw "SHA-256 mismatch: $Name" }
    Move-Item -LiteralPath $partial -Destination $cached -Force
    return $cached
}
function Install-Verified([string]$Source, [string]$Relative, [string]$Sha256) {
    $target = Resolve-VisionPath $Relative
    if (Test-Hash $target $Sha256) { return }
    if ($PreserveExisting -and (Test-Path -LiteralPath $target)) {
        throw "Existing helper asset checksum differs and will not be replaced: $target"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    $pending = $target + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    Copy-Item -LiteralPath $Source -Destination $pending
    if (-not (Test-Hash $pending $Sha256)) { throw "Copy verification failed: $Relative" }
    if (Test-Path -LiteralPath $target) {
        Move-Item -LiteralPath $target -Destination ($target + '.backup-' + [Guid]::NewGuid().ToString('N'))
    }
    Move-Item -LiteralPath $pending -Destination $target
    Write-Host "Ready: $Relative"
}
$dflRevision = 'e4b7543ffa1d73b26fce1e31852727f658ba490c'
$models = @(
    @{ Name='S3FD.npy'; Hash='b4894ecfba8e6461eb1a69490f76184620c816605238ff0ba3216f1143a06c29'; Repository='iperov/DeepFaceLab'; Revision=$dflRevision },
    @{ Name='2DFAN.npy'; Hash='ca2dc7f0b2aa146842e6de2119fedff1142188b7cff5ab702564952d6cba4624'; Repository='iperov/DeepFaceLab'; Revision=$dflRevision },
    @{ Name='3DFAN.npy'; Hash='b50d2faf0fd4d6503aba9d19365e7aff06f2ea96bac37fc5f8f25a191a0a63a9'; Repository='iperov/DeepFaceLab'; Revision=$dflRevision }
)
foreach ($model in $models) {
    $source = Resolve-VisionPath ('_internal/DeepFaceLab/facelib/' + $model.Name)
    if ($PreserveExisting -and (Test-Path -LiteralPath $source) -and -not (Test-Hash $source $model.Hash)) {
        throw "Existing helper weight checksum differs and will not be replaced: $source"
    }
    if (-not (Test-Hash $source $model.Hash)) {
        $source = Get-Verified $model.Name "https://raw.githubusercontent.com/$($model.Repository)/$($model.Revision)/facelib/$($model.Name)" $model.Hash
    }
    Install-Verified $source "_internal/DeepFaceLab/facelib/$($model.Name)" $model.Hash
}
# This fixed trained inference model is also physically present in both release
# archives. Git clones fetch the identical, hash-pinned converted asset.
$genericManifest = Resolve-VisionPath 'release/generic-xseg.json'
$generic = Get-Content -LiteralPath $genericManifest -Raw | ConvertFrom-Json
if ($generic.schemaVersion -ne 1 -or $generic.licenseStatus -ne 'verified') {
    throw 'Generic XSeg source/license record is incomplete'
}
$genericTarget = Resolve-VisionPath $generic.converted.path
if ($PreserveExisting -and (Test-Path -LiteralPath $genericTarget) -and -not (Test-Hash $genericTarget $generic.converted.sha256)) {
    throw "Existing generic XSeg weight checksum differs and will not be replaced: $genericTarget"
}
$genericSource = $genericTarget
if (-not (Test-Hash $genericSource $generic.converted.sha256)) {
    $genericSource = Get-Verified 'XSeg_256.pth' $generic.converted.downloadUrl $generic.converted.sha256
}
Install-Verified $genericSource $generic.converted.path $generic.converted.sha256
Install-Verified (Resolve-VisionPath $generic.metadata.source) $generic.metadata.path $generic.metadata.sha256
Install-Verified $genericManifest '_internal/model_generic_xseg/SOURCE.json' (Get-FileHash -LiteralPath $genericManifest -Algorithm SHA256).Hash
foreach ($record in @($generic.license, $generic.summary)) {
    Install-Verified (Resolve-VisionPath $record.source) $record.path $record.sha256
}
$sface = '_internal/vision_models/face_recognition_sface_2021dec.onnx'
$sfaceHash = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
Install-Verified (Resolve-VisionPath 'tools/licenses/SFace-Apache-2.0.txt') '_internal/vision_models/SFace-LICENSE.txt' 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
if (-not (Test-Hash (Resolve-VisionPath $sface) $sfaceHash)) {
    if ($PreserveExisting -and (Test-Path -LiteralPath (Resolve-VisionPath $sface))) {
        throw "Existing SFace weight checksum differs and will not be replaced: $sface"
    }
    $source = Get-Verified 'sface-2021dec.onnx' 'https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' $sfaceHash
    Install-Verified $source $sface $sfaceHash
}
$ffmpegHash = '72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3'
$ffprobeHash = '19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f'
$ffmpegBinariesReady = (Test-Hash (Resolve-VisionPath '_internal/ffmpeg/ffmpeg.exe') $ffmpegHash) -and
                      (Test-Hash (Resolve-VisionPath '_internal/ffmpeg/ffprobe.exe') $ffprobeHash)
$ffmpegDocumentationReady = $true
foreach ($relative in @('LICENSE', 'README.txt', 'doc/ffmpeg.html',
        'presets/libvpx-1080p.ffpreset', 'presets/libvpx-1080p50_60.ffpreset',
        'presets/libvpx-360p.ffpreset', 'presets/libvpx-720p.ffpreset', 'presets/libvpx-720p50_60.ffpreset')) {
    if (-not (Test-Path -LiteralPath (Resolve-VisionPath ('_internal/ffmpeg/' + $relative)) -PathType Leaf)) {
        $ffmpegDocumentationReady = $false
    }
}
if (-not $SkipFFmpeg -and -not ($ffmpegBinariesReady -and $ffmpegDocumentationReady)) {
    if ($PreserveExisting) {
        foreach ($entry in @(
            @{ Name='ffmpeg.exe'; Hash=$ffmpegHash },
            @{ Name='ffprobe.exe'; Hash=$ffprobeHash }
        )) {
            $target = Resolve-VisionPath ('_internal/ffmpeg/' + $entry.Name)
            if ((Test-Path -LiteralPath $target) -and -not (Test-Hash $target $entry.Hash)) {
                throw "Existing FFmpeg binary checksum differs and will not be replaced: $target"
            }
        }
    }
    $archive = Get-Verified 'ffmpeg-9.0.1-essentials_build.zip' 'https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-essentials_build.zip' 'fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9'
    $expanded = Join-Path $cacheRoot ('ffmpeg-' + [Guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $archive -DestinationPath $expanded
    $distribution = Join-Path $expanded 'ffmpeg-9.0.1-essentials_build'
    Install-Verified (Join-Path $distribution 'bin/ffmpeg.exe') '_internal/ffmpeg/ffmpeg.exe' $ffmpegHash
    Install-Verified (Join-Path $distribution 'bin/ffprobe.exe') '_internal/ffmpeg/ffprobe.exe' $ffprobeHash
    # Install verified distribution resources one file at a time. Matching
    # existing files are retained; PreserveExisting rejects differing files.
    $documentation = @((Get-Item -LiteralPath (Join-Path $distribution 'LICENSE')),
                       (Get-Item -LiteralPath (Join-Path $distribution 'README.txt')))
    $documentation += @(Get-ChildItem -LiteralPath (Join-Path $distribution 'doc') -File -Recurse)
    $documentation += @(Get-ChildItem -LiteralPath (Join-Path $distribution 'presets') -File -Recurse)
    foreach ($file in $documentation) {
        $relative = $file.FullName.Substring($distribution.Length + 1)
        $resourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        Install-Verified $file.FullName ('_internal/ffmpeg/' + $relative) $resourceHash
    }
}
if ($SkipFFmpeg) { Write-Host 'Helper weights verified: S3FD, 2DFAN, 3DFAN, generic XSeg, SFace. FFmpeg left to the caller.' }
else { Write-Host 'Visual dependencies verified: FFmpeg, S3FD, 2DFAN, 3DFAN, generic XSeg, SFace.' }
# Resource acquisition is an installation step; inference never downloads.
$resourcePython = Resolve-VisionPath '.venv/Scripts/python.exe'
$resourceTool = Resolve-VisionPath 'tools/prepare-production-vision.py'
$resourceProfiles = @('production')
if ($IncludeRestoration) { $resourceProfiles += 'restoration' }
if ($IncludeScene) { $resourceProfiles += 'scene' }
if ($ResourcePackPath) {
    & $resourcePython -I $resourceTool install-pack --project-root $visionRoot --output $ResourcePackPath --profiles @resourceProfiles
} else {
    $resourceOptions = @('install','--project-root',$visionRoot,'--profiles') + $resourceProfiles
    if (-not $VisionCacheRoot) { $VisionCacheRoot = Join-Path $visionRoot 'workspace/.vision-models' }
    $resourceOptions += @('--cache-root',$VisionCacheRoot)
    if (-not $NoNetwork) { $resourceOptions += '--allow-download' }
    & $resourcePython -I $resourceTool @resourceOptions
}
if ($LASTEXITCODE -ne 0) { throw 'Production vision resources incomplete. Supply a verified separate ResourcePackPath (TUFA official weights require explicit acquisition); no fallback or automatic inference download is permitted.' }
