[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$SourceArchivePath = '',
    [string]$SourcePinPath = '',
    [switch]$NoNetwork,
    [ValidateRange(1,536870912)][long]$MaxArchiveBytes = 536870912,
    [ValidateRange(1,2147483648)][long]$MaxExpandedBytes = 2147483648,
    [ValidateRange(1,268435456)][long]$MaxEntryBytes = 268435456,
    [ValidateRange(1,50000)][int]$MaxEntries = 50000
)

# Source acquisition only. install-source.ps1 installs the project-local runtimes.
# No Git dependency, no branch history import, and no replacement of user files.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Read-ProjectSourcePin([string]$Path) {
    if (-not $Path) { $Path = Join-Path $PSScriptRoot 'source-pin.json' }
    Assert-ProjectInstallPlainPath $Path | Out-Null
    $pin = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($pin.schemaVersion -ne 1 -or $pin.product -cne 'DFL-PT-WEBUI' -or
        $pin.sourceCommit -cnotmatch '^[0-9a-f]{40}$' -or $pin.archiveSha256 -cnotmatch '^[0-9a-f]{64}$' -or
        $pin.archiveRoot -cne ('DFL-PT-WEBUI-' + $pin.sourceCommit)) {
        throw '发行版源码快照未固定或来源记录无效；请下载已验证的新版启动器。'
    }
    return $pin
}

function Write-ProjectInstallationRecord([string]$Root, [object]$Pin) {
    $version = Get-Content -LiteralPath (Join-Path $Root 'release/version.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($version.version -cne $Pin.applicationVersion) { throw '源码应用版本与固定来源记录不一致。' }
    $launcherVersion = if ($version.PSObject.Properties.Name -contains 'launcherVersion') { $version.launcherVersion } else { 'unknown' }
    $record = @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion=$version.version;launcherVersion=$launcherVersion;
        installationSource='official-source';sourceCommit=$Pin.sourceCommit;archiveSha256=$Pin.archiveSha256;installedAt=[DateTime]::UtcNow.ToString('o')}
    [IO.File]::WriteAllText((Resolve-ProjectInstallChild $Root 'release/installation.json'), ($record | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
}

function Write-ProjectInstallEvent([string]$Id, [string]$Status, [int]$Progress, [string]$Message) {
    [Console]::Out.WriteLine((@{ stage='source'; id=$Id; status=$Status; progress=$Progress;
        downloaded=0; total=0; message=$Message } | ConvertTo-Json -Compress))
}

function Assert-ProjectInstallPlainPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $ancestor = $absolute
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "安装路径不能包含链接或重解析点：$ancestor"
            }
        }
        $parent = Split-Path -Parent $ancestor
        if ($parent -eq $ancestor) { break }
        $ancestor = $parent
    }
    return $absolute
}

function Resolve-ProjectInstallChild([string]$Base, [string]$Relative) {
    if ([string]::IsNullOrWhiteSpace($Relative) -or [IO.Path]::IsPathRooted($Relative)) {
        throw '安装文件必须使用项目内的相对路径。'
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $Base $Relative))
    if (-not $candidate.StartsWith($Base.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) { throw '安装文件路径越出了指定目录。' }
    return (Assert-ProjectInstallPlainPath $candidate)
}

function Assert-ProjectInstallIdentity([string]$Path) {
    foreach ($relative in @(
        'release/version.json', 'requirements.txt', '_internal/DeepFaceLab/me.py',
        '_internal/DeepFaceLab/me_backend/engine.py', '_internal/DeepFaceLab/me_backend/network.py',
        '_internal/DeepFaceLab/core/leras/nn.py', 'webui/package.json', 'webui/pnpm-lock.yaml',
        'webui/pnpm-workspace.yaml', 'webui/scripts/local-manager.mjs', 'webui/server/index.mjs',
        'launcher/install-source.ps1', 'launcher/setup-runtime.ps1', 'launcher/runtime-manifest.json',
        'tools/prepare-vision-runtime.ps1', 'tools/dist-provenance.mjs', 'install-source.bat')) {
        $file = Resolve-ProjectInstallChild $Path $relative
        if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -eq 0) {
            throw "不是完整的 DFL-PT-WEBUI 源码，缺少 $relative。"
        }
        if ((Get-Item -LiteralPath $file).Length -gt 8388608) { throw "项目身份文件过大：$relative" }
    }
    $version = Get-Content -LiteralPath (Join-Path $Path 'release/version.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($version.product -cne 'DFL-PT-WEBUI' -or
        ($version.schemaVersion -isnot [int] -and $version.schemaVersion -isnot [long]) -or
        $version.schemaVersion -ne 1 -or [string]::IsNullOrWhiteSpace([string]$version.version)) {
        throw '仅接受 DFL-PT-WEBUI schema 1 项目；原版 DFL-WEBUI 不适用。'
    }
    $manifest = Get-Content -LiteralPath (Join-Path $Path 'launcher/runtime-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schemaVersion -ne 3) { throw 'PT 运行环境清单版本不正确。' }
    $package = Get-Content -LiteralPath (Join-Path $Path 'webui/package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($package.type -ne 'module' -or $package.packageManager -notmatch '^pnpm@') {
        throw '缺少原生 PT WebUI 的包管理配置。'
    }
    $entry = Get-Content -LiteralPath (Join-Path $Path '_internal/DeepFaceLab/me.py') -Raw -Encoding UTF8
    $engine = Get-Content -LiteralPath (Join-Path $Path '_internal/DeepFaceLab/me_backend/engine.py') -Raw -Encoding UTF8
    $network = Get-Content -LiteralPath (Join-Path $Path '_internal/DeepFaceLab/me_backend/network.py') -Raw -Encoding UTF8
    $requirements = Get-Content -LiteralPath (Join-Path $Path 'requirements.txt') -Raw -Encoding UTF8
    if ($entry -notmatch '(?m)^import torch\s*$' -or $entry -notmatch 'me_backend\.engine import MEEngine' -or
        $engine -notmatch 'class MEEngine\b' -or $engine -notmatch '(?m)^import torch\s*$' -or
        $network -notmatch 'class MENetwork\(torch\.nn\.Module\)' -or
        ($entry + "`n" + $engine + "`n" + $network) -match '(?im)^\s*(import tensorflow\b|from tensorflow\b)' -or
        $requirements -match '(?im)^\s*(tensorflow|tensorflow-gpu)(\b|[=<>])' -or
        (Test-Path -LiteralPath (Join-Path $Path '_internal/DeepFaceLab_old')) -or
        (Test-Path -LiteralPath (Join-Path $Path '_internal/python_common'))) {
        throw '项目必须保留 ME/PyTorch 后端，不能安装原版 TensorFlow 项目。'
    }
}

function Assert-ProjectInstallTarget([string]$Path) {
    Assert-ProjectInstallPlainPath $Path | Out-Null
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw '安装目标必须是独立目录。' }
    $children = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($children.Count -eq 0) { return }
    if ($children.Count -ne 1 -or $children[0].Name -cne '.launcher-install' -or -not $children[0].PSIsContainer) {
        throw '安装目标包含已有文件，不能覆盖；请使用独立的空目录。'
    }
    $state = Resolve-ProjectInstallChild $Path '.launcher-install'
    $owner = Resolve-ProjectInstallChild $state 'owner.txt'
    if (-not (Test-Path -LiteralPath $owner -PathType Leaf) -or (Get-Item -LiteralPath $owner).Length -gt 128 -or
        [IO.File]::ReadAllText($owner) -cne 'DFL-PT-WEBUI install workspace v1') {
        throw '安装缓存不属于 DFL-PT-WEBUI，不能继续。'
    }
}

function Get-ProjectInstallArchive([string]$Destination, [string]$OfflineArchive, [bool]$Offline, [long]$Limit, [object]$Pin=$null) {
    if (-not [string]::IsNullOrWhiteSpace($OfflineArchive)) {
        $inputPath = Assert-ProjectInstallPlainPath $OfflineArchive
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf) -or
            (Get-Item -LiteralPath $inputPath).Length -gt $Limit) { throw '离线源码包不存在或超过大小上限。' }
        Copy-Item -LiteralPath $inputPath -Destination $Destination
        if ((Get-Item -LiteralPath $Destination).Length -gt $Limit) { throw '源码包超过大小上限。' }
        return
    }
    if ($Offline) { throw '离线模式需要提供 SourceArchivePath 源码 ZIP。' }
    if (-not $Pin) { throw '在线安装必须指定固定源码快照。' }
    # HttpClient disables redirects and applies a finite timeout. Never accept an
    # alternate source host from downloaded content, command-line input or Git.
    Add-Type -AssemblyName System.Net.Http
    $previousTls = [Net.ServicePointManager]::SecurityProtocol
    $handler = New-Object Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    if ($handler.PSObject.Properties.Name -contains 'SslProtocols') {
        $handler.SslProtocols = [Security.Authentication.SslProtocols]::Tls12
    }
    $client = New-Object Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromMinutes(10)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('DFL-PT-WEBUI-Launcher/1.0')
    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $response = $client.GetAsync(('https://codeload.github.com/LeoSasion/DFL-PT-WEBUI/zip/' + $Pin.sourceCommit),
            [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if ([int]$response.StatusCode -ne 200) { throw "官方源码下载失败（HTTP $([int]$response.StatusCode)）。" }
        if ($response.Content.Headers.ContentLength -and $response.Content.Headers.ContentLength -gt $Limit) {
            throw '官方源码包超过大小上限。'
        }
        $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $outputStream = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $buffer = New-Object byte[] 65536
        $total = [long]0
        $timer = [Diagnostics.Stopwatch]::StartNew()
        while ($true) {
            if ($timer.Elapsed.TotalMinutes -gt 10) { throw '官方源码下载超时，请稍后重试。' }
            # Bound each body read too; ResponseHeadersRead's client timeout
            # otherwise stops covering the transfer after headers arrive.
            $readTask = $inputStream.ReadAsync($buffer, 0, $buffer.Length)
            if (-not $readTask.Wait(30000)) { throw '官方源码下载无响应，请稍后重试。' }
            $read = $readTask.GetAwaiter().GetResult()
            if ($read -eq 0) { break }
            $total += $read
            if ($total -gt $Limit) { throw '官方源码包超过大小上限。' }
            $outputStream.Write($buffer, 0, $read)
        }
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
        $client.Dispose()
        $handler.Dispose()
        [Net.ServicePointManager]::SecurityProtocol = $previousTls
    }
}

function Expand-ProjectInstallArchive([string]$ZipPath, [string]$Destination, [long]$ExpandedLimit, [long]$EntryLimit, [int]$EntryCountLimit, [string]$ArchiveRoot='DFL-PT-WEBUI-main') {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        if ($zip.Entries.Count -eq 0 -or $zip.Entries.Count -gt $EntryCountLimit) { throw '源码包文件数量不符合上限。' }
        $paths = New-Object 'System.Collections.Generic.Dictionary[string,bool]' ([StringComparer]::OrdinalIgnoreCase)
        $entries = New-Object 'System.Collections.Generic.List[object]'
        $expanded = [long]0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName
            # Use strict portable names. Windows alternate streams, device
            # names, trailing dots/spaces and case aliases must fail closed.
            if ($name.Length -gt 512 -or $name -match '[\\:\x00-\x1f<>"|?*]' -or $name.StartsWith('/')) {
                throw "源码包含不安全路径：$name"
            }
            $directory = $name.EndsWith('/')
            $trimmed = $name.TrimEnd('/')
            $parts = @($trimmed.Split('/'))
            foreach ($part in $parts) {
                if ([string]::IsNullOrWhiteSpace($part) -or $part -eq '.' -or $part -eq '..' -or
                    $part.EndsWith('.') -or $part.EndsWith(' ') -or
                    $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') {
                    throw "源码包含不安全路径：$name"
                }
            }
            if ($parts[0] -cne $ArchiveRoot) { throw '源码包根目录与固定发行来源不一致。' }
            $unixMode = ([long]$entry.ExternalAttributes -shr 16) -band 0xf000
            if (($entry.ExternalAttributes -band 0x400) -ne 0 -or
                ($unixMode -ne 0 -and $unixMode -ne 0x8000 -and $unixMode -ne 0x4000) -or
                ($unixMode -eq 0x4000 -and -not $directory) -or
                ($unixMode -eq 0x8000 -and $directory)) { throw "源码包不能包含链接或特殊文件：$name" }
            if ($parts.Count -eq 1) {
                if (-not $directory) { throw '源码包根目录不是文件夹。' }
                continue
            }
            $relative = ($parts[1..($parts.Count-1)] -join '/')
            if ($parts[1] -ieq '.launcher-install' -or $parts[1] -ieq '.git') { throw '源码包不能包含安装缓存或 Git 私有历史。' }
            if ($entry.Length -gt $EntryLimit -or ($directory -and $entry.Length -ne 0)) { throw '源码包单个文件超过大小上限。' }
            $expanded += $entry.Length
            if ($expanded -gt $ExpandedLimit) { throw '源码包解压后超过大小上限。' }
            if ($paths.ContainsKey($relative)) { throw "源码包含重复文件名：$relative" }
            $paths.Add($relative, $directory)
            $entries.Add(@{ entry=$entry; relative=$relative; directory=$directory })
        }
        # Validate the complete graph before extracting a byte: a file cannot
        # also be the parent of another entry, regardless of ZIP entry order.
        foreach ($relative in $paths.Keys) {
            $parent = $relative
            while ($parent.Contains('/')) {
                $parent = $parent.Substring(0, $parent.LastIndexOf('/'))
                if ($paths.ContainsKey($parent) -and -not $paths[$parent]) { throw "源码包文件与目录冲突：$parent" }
            }
        }
        New-Item -ItemType Directory -Path $Destination | Out-Null
        $buffer = New-Object byte[] 65536
        foreach ($record in $entries) {
            $target = Resolve-ProjectInstallChild $Destination $record.relative
            if ($record.directory) { New-Item -ItemType Directory -Path $target -Force | Out-Null; continue }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Assert-ProjectInstallPlainPath $target | Out-Null
            $inputStream = $record.entry.Open()
            $outputStream = $null
            try {
                $outputStream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                $written = [long]0
                while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $written += $read
                    if ($written -gt $record.entry.Length -or $written -gt $EntryLimit) { throw '源码包实际文件长度超过声明或上限。' }
                    $outputStream.Write($buffer, 0, $read)
                }
                if ($written -ne $record.entry.Length) { throw '源码包文件长度不完整。' }
            } finally {
                if ($outputStream) { $outputStream.Dispose() }
                $inputStream.Dispose()
            }
        }
    } finally { $zip.Dispose() }
}

function Move-ProjectInstallEntry([string]$Source, [string]$Destination) {
    Assert-ProjectInstallPlainPath $Source | Out-Null
    Assert-ProjectInstallPlainPath $Destination | Out-Null
    # .NET Move rejects an already existing destination atomically; Move-Item
    # can instead nest a source under an unexpectedly created destination dir.
    if ([IO.Directory]::Exists($Source)) { [IO.Directory]::Move($Source, $Destination) }
    else { [IO.File]::Move($Source, $Destination) }
}

function Publish-ProjectInstallSource([string]$Source, [string]$Target) {
    Assert-ProjectInstallTarget $Target
    $children = @(Get-ChildItem -LiteralPath $Source -Force)
    if ($children.Count -eq 0) { throw '没有可发布的源码。' }
    foreach ($child in $children) {
        $destination = Resolve-ProjectInstallChild $Target $child.Name
        if (Test-Path -LiteralPath $destination) { throw "源码不能覆盖已有内容：$($child.Name)" }
    }
    $moved = New-Object 'System.Collections.Generic.List[string]'
    try {
        foreach ($child in $children) {
            $sourcePath = Resolve-ProjectInstallChild $Source $child.Name
            $destination = Resolve-ProjectInstallChild $Target $child.Name
            if (Test-Path -LiteralPath $destination) { throw "安装期间出现同名文件：$($child.Name)" }
            Move-ProjectInstallEntry $sourcePath $destination
            $moved.Add($child.Name)
        }
        Assert-ProjectInstallIdentity $Target
    } catch {
        $reason = $_.Exception.Message
        $rollbackFailed = New-Object 'System.Collections.Generic.List[string]'
        for ($index = $moved.Count - 1; $index -ge 0; $index--) {
            try {
                $from = Resolve-ProjectInstallChild $Target $moved[$index]
                $to = Resolve-ProjectInstallChild $Source $moved[$index]
                if (Test-Path -LiteralPath $to) { throw '回滚目标已被占用。' }
                Move-ProjectInstallEntry $from $to
            } catch { $rollbackFailed.Add($moved[$index]) }
        }
        if ($rollbackFailed.Count -gt 0) {
            $exception = New-Object IO.IOException("源码发布失败：$reason；以下新文件未能回滚，请保留安装缓存后重试：$($rollbackFailed -join ', ')")
            $exception.Data['PreserveInstallSession'] = $true
            throw $exception
        }
        throw "源码发布失败，已回滚：$reason"
    }
}

function Remove-ProjectInstallSession([string]$State, [string]$Session) {
    $sessionName = Split-Path -Leaf $Session
    if ($sessionName -notmatch '^source-[0-9a-f]{32}$' -or
        [IO.Path]::GetFullPath((Split-Path -Parent $Session)) -cne [IO.Path]::GetFullPath($State)) {
        throw '拒绝清理未验证的安装缓存。'
    }
    $owner = Resolve-ProjectInstallChild $State 'owner.txt'
    if ((Get-Item -LiteralPath $owner).Length -gt 128 -or
        [IO.File]::ReadAllText($owner) -cne 'DFL-PT-WEBUI install workspace v1') { throw '安装缓存归属不正确。' }
    Assert-ProjectInstallPlainPath $Session | Out-Null
    if (Test-Path -LiteralPath $Session) {
        $directories = New-Object 'System.Collections.Generic.Queue[string]'
        $directories.Enqueue($Session)
        while ($directories.Count -gt 0) {
            $directory = $directories.Dequeue()
            foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw '安装缓存含链接，拒绝递归清理。' }
                if ($item.PSIsContainer) { $directories.Enqueue($item.FullName) }
            }
        }
        Remove-Item -LiteralPath $Session -Recurse -Force -ErrorAction Stop
    }
}

function Invoke-ProjectSourceInstall {
    [CmdletBinding()]
    param([string]$ProjectRoot, [string]$SourceArchivePath='', [string]$SourcePinPath='', [switch]$NoNetwork,
        [ValidateRange(1,536870912)][long]$MaxArchiveBytes=536870912,
        [ValidateRange(1,2147483648)][long]$MaxExpandedBytes=2147483648,
        [ValidateRange(1,268435456)][long]$MaxEntryBytes=268435456,
        [ValidateRange(1,50000)][int]$MaxEntries=50000)
    $lock = $null
    $session = $null
    $state = $null
    $keepSession = $false
    try {
        if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { throw '必须提供独立的 ProjectRoot 安装目录。' }
        $root = (Assert-ProjectInstallPlainPath $ProjectRoot).TrimEnd('\','/')
        if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\','/')) { throw '不能将源码安装到磁盘根目录。' }
        foreach ($protected in @([Environment]::GetFolderPath('Windows'), [Environment]::GetFolderPath('UserProfile'),
            [Environment]::GetFolderPath('ProgramFiles'), [Environment]::GetFolderPath('ProgramFilesX86'))) {
            if ($protected -and $root -ieq $protected.TrimEnd('\','/')) { throw '请使用系统目录以外的独立安装子目录。' }
        }
        if ((Test-Path -LiteralPath (Join-Path $root 'release/version.json')) -or
            (Test-Path -LiteralPath (Join-Path $root '_internal/DeepFaceLab/me.py'))) {
            Assert-ProjectInstallIdentity $root
            Write-ProjectInstallEvent 'complete' 'complete' 100 '已验证现有 DFL-PT-WEBUI 源码，无需重新下载。'
            return 0
        }
        Assert-ProjectInstallTarget $root
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        $state = Resolve-ProjectInstallChild $root '.launcher-install'
        if (-not (Test-Path -LiteralPath $state)) {
            New-Item -ItemType Directory -Path $state | Out-Null
            [IO.File]::WriteAllText((Join-Path $state 'owner.txt'), 'DFL-PT-WEBUI install workspace v1', (New-Object Text.UTF8Encoding($false)))
        }
        $lockPath = Resolve-ProjectInstallChild $state 'project-source.lock'
        try { $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
        catch { throw '已有源码安装正在运行，或安装目录不可写。' }
        Assert-ProjectInstallTarget $root
        $session = Resolve-ProjectInstallChild $state ('source-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $session | Out-Null
        $archive = Resolve-ProjectInstallChild $session 'source.zip'
        $pin = if ($SourcePinPath -or -not $NoNetwork) { Read-ProjectSourcePin $SourcePinPath } else { $null }
        Write-ProjectInstallEvent 'download' 'running' 5 '正在获取 DFL-PT-WEBUI 固定版本源码。'
        Get-ProjectInstallArchive $archive $SourceArchivePath ([bool]$NoNetwork) $MaxArchiveBytes $pin
        if ($pin -and (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -cne $pin.archiveSha256) { throw '源码包 SHA-256 与固定发行来源不一致。' }
        Write-ProjectInstallEvent 'verify' 'running' 35 '正在检查源码包路径、大小与完整性。'
        $staging = Resolve-ProjectInstallChild $session 'staging'
        $archiveRoot = if ($pin) { $pin.archiveRoot } else { 'DFL-PT-WEBUI-main' }
        Expand-ProjectInstallArchive $archive $staging $MaxExpandedBytes $MaxEntryBytes $MaxEntries $archiveRoot
        Assert-ProjectInstallIdentity $staging
        if ($pin) { Write-ProjectInstallationRecord $staging $pin }
        Write-ProjectInstallEvent 'publish' 'running' 80 '正在将已验证的 PT 源码放入独立安装目录。'
        Publish-ProjectInstallSource $staging $root
        Write-ProjectInstallEvent 'complete' 'complete' 100 'DFL-PT-WEBUI 源码已就绪，可以继续安装本项目运行环境。'
        return 0
    } catch {
        $keepSession = [bool]$_.Exception.Data['PreserveInstallSession']
        Write-ProjectInstallEvent 'failed' 'failed' 0 $_.Exception.Message
        return 1
    } finally {
        if ($lock) { $lock.Dispose() }
        if ($session -and -not $keepSession) {
            try { Remove-ProjectInstallSession $state $session }
            catch { Write-ProjectInstallEvent 'cleanup' 'warning' 0 ('安装缓存已保留：' + $_.Exception.Message) }
        }
    }
}

# Dot sourcing exposes the bounded operations to offline Pester tests without
# exiting the test runner; normal host execution always returns an explicit code.
if ($MyInvocation.InvocationName -ne '.') {
    [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
    if (-not $SourcePinPath) { $SourcePinPath = Join-Path $PSScriptRoot 'source-pin.json' }
    $installExitCode = Invoke-ProjectSourceInstall -ProjectRoot $ProjectRoot -SourceArchivePath $SourceArchivePath -SourcePinPath $SourcePinPath -NoNetwork:$NoNetwork `
        -MaxArchiveBytes $MaxArchiveBytes -MaxExpandedBytes $MaxExpandedBytes -MaxEntryBytes $MaxEntryBytes -MaxEntries $MaxEntries
    exit $installExitCode
}
