[CmdletBinding()]
param([string]$ProjectRoot, [ValidateSet('apply','rollback','complete','guard')][string]$Action='apply',
    [string]$SourcePinPath='', [string]$SourceArchivePath='')
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'install-project.ps1') -ProjectRoot $ProjectRoot -SourcePinPath $SourcePinPath -SourceArchivePath $SourceArchivePath

# A source update never replaces runtimes, datasets, models, credentials, or
# local configuration. The retained transaction includes every replaced byte.
function Test-ProtectedUpgradePath([string]$Relative) {
    return $Relative -match '^(?i:workspace|workspaces|\.git|\.venv|\.launcher-install|release-output)(/|$)' -or
        $Relative -match '^(?i:_internal/(node|python_base|git|ffmpeg|config\.txt)|webui/(node_modules|\.runtime|\.env[^/]*))(/|$)' -or
        $Relative -match '(?i)(^|/)(\.env[^/]*|DFL-PT-WEBUI\.exe|DFL-PT-WEBUI\.Launcher\.exe)$'
}

function Assert-UpgradeIdle([string]$Root) {
    $prefix = $Root.TrimEnd('\','/') + '\'
    foreach ($process in Get-CimInstance Win32_Process -ErrorAction Stop) {
        $ownedExecutable = $process.ExecutablePath -and $process.ExecutablePath.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)
        $ownedCommand = $process.CommandLine -and ($process.CommandLine.IndexOf($prefix,[StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $process.CommandLine.IndexOf($prefix.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase) -ge 0)
        if (($ownedExecutable -or $ownedCommand) -and $process.Name -match '^(?i:python(?:w)?|node|ffmpeg)\.exe$') {
            throw '项目进程或任务正在运行，请先结束任务和 WebUI 再修复或升级。'
        }
    }
}

function Get-UpgradeFiles([string]$Root) {
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    $pending.Enqueue($Root)
    while ($pending.Count) {
        $directory = $pending.Dequeue()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            Assert-ProjectInstallPlainPath $item.FullName | Out-Null
            if ($item.PSIsContainer) { $pending.Enqueue($item.FullName) }
            else { $item.FullName.Substring($Root.TrimEnd('\','/').Length+1).Replace('\','/') }
        }
    }
}

function Invoke-ProjectUpgrade {
    param([string]$Root, [string]$Action, [string]$PinPath, [string]$ArchivePath)
    $Root = (Assert-ProjectInstallPlainPath $Root).TrimEnd('\','/')
    Assert-UpgradeIdle $Root
    if ($Action -eq 'guard') { return }
    Assert-ProjectInstallIdentity $Root
    # Git working copies keep their history and branch intact. Source ZIP
    # upgrades are intended for installed archives; Git users use reviewed refs.
    if (Test-Path -LiteralPath (Join-Path $Root '.git')) { throw 'Git 工作副本请通过固定公开提交升级；启动器不会覆盖 Git 源码或私有历史。' }
    $state = Resolve-ProjectInstallChild $Root '.launcher-install'
    if (-not (Test-Path -LiteralPath $state)) {
        New-Item -ItemType Directory -Path $state | Out-Null
        [IO.File]::WriteAllText((Join-Path $state 'owner.txt'),'DFL-PT-WEBUI install workspace v1')
    }
    $owner = Resolve-ProjectInstallChild $state 'owner.txt'
    if ([IO.File]::ReadAllText($owner) -cne 'DFL-PT-WEBUI install workspace v1') { throw '安装缓存归属不正确。' }
    $lock = [IO.File]::Open((Resolve-ProjectInstallChild $state 'project-source.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    try {
        $transaction = Resolve-ProjectInstallChild $state 'upgrade-current'
        $journalPath = Resolve-ProjectInstallChild $transaction 'journal.json'
        if ($Action -eq 'complete') {
            if (-not (Test-Path -LiteralPath $journalPath)) { throw '没有待完成的升级事务。' }
            $journal = Get-Content -LiteralPath $journalPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($journal.status -ne 'applied') { throw '升级事务未通过应用检查。' }
            $history = Resolve-ProjectInstallChild $state ('upgrade-backup-' + [DateTime]::UtcNow.ToString('yyyyMMddHHmmss') + '-' + [Guid]::NewGuid().ToString('N'))
            Move-ProjectInstallEntry $transaction $history
            Write-ProjectInstallEvent 'upgrade' 'complete' 100 '升级完成；原源码、构建和配置备份已保留在安装缓存中。'
            return
        }
        if ($Action -eq 'rollback') {
            if (-not (Test-Path -LiteralPath $journalPath)) { throw '没有可回退的升级事务。' }
            $journal = Get-Content -LiteralPath $journalPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($journal.schemaVersion -ne 1 -or $journal.product -cne 'DFL-PT-WEBUI' -or @($journal.files).Count -gt 50000) { throw '升级事务标识不正确。' }
            # Validate the complete restore set before moving a working build or
            # dependency tree. A damaged backup must leave the current app alone.
            foreach ($file in $journal.files) {
                if (Test-ProtectedUpgradePath $file.path) { throw '升级事务包含受保护路径。' }
                Resolve-ProjectInstallChild $Root $file.path | Out-Null
                if ($file.existed) {
                    $backup = Resolve-ProjectInstallChild $transaction ('backup/' + $file.path)
                    if (-not (Test-Path -LiteralPath $backup -PathType Leaf) -or
                        (Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.sha256) {
                        throw '回退备份校验失败，请保留安装缓存；当前源码与依赖未移动。'
                    }
                }
            }
            foreach ($reserved in @('failed-dist','failed-node_modules')) {
                if (Test-Path -LiteralPath (Resolve-ProjectInstallChild $transaction $reserved)) { throw '回退暂存目录已存在，请保留缓存并检查之前的恢复。' }
            }
            $dependencies = Resolve-ProjectInstallChild $Root 'webui/node_modules'
            $originalDependencies = Resolve-ProjectInstallChild $transaction 'original-node_modules'
            if (Test-Path -LiteralPath $originalDependencies) {
                if (Test-Path -LiteralPath $dependencies) { Move-ProjectInstallEntry $dependencies (Resolve-ProjectInstallChild $transaction 'failed-node_modules') }
                Move-ProjectInstallEntry $originalDependencies $dependencies
            } elseif ($journal.dependenciesMoved -and (Test-Path -LiteralPath $dependencies)) {
                Move-ProjectInstallEntry $dependencies (Resolve-ProjectInstallChild $transaction 'failed-node_modules')
            }
            $dist = Resolve-ProjectInstallChild $Root 'webui/dist'
            if (Test-Path -LiteralPath $dist) { Move-ProjectInstallEntry $dist (Resolve-ProjectInstallChild $transaction 'failed-dist') }
            foreach ($file in $journal.files) {
                if (Test-ProtectedUpgradePath $file.path) { throw '升级事务包含受保护路径。' }
                $destination = Resolve-ProjectInstallChild $Root $file.path
                if ($file.existed) {
                    $backup = Resolve-ProjectInstallChild $transaction ('backup/' + $file.path)
                    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
                    Copy-Item -LiteralPath $backup -Destination $destination -Force
                } elseif (Test-Path -LiteralPath $destination -PathType Leaf) { Remove-Item -LiteralPath $destination -Force }
            }
            $journal.status = 'rolled-back'
            [IO.File]::WriteAllText($journalPath,($journal | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
            Move-ProjectInstallEntry $transaction (Resolve-ProjectInstallChild $state ('upgrade-rollback-' + [Guid]::NewGuid().ToString('N')))
            Write-ProjectInstallEvent 'upgrade' 'rolled-back' 100 '升级失败，原源码与构建已回退；工作区、模型、配置和运行环境均已保留。'
            return
        }
        if (Test-Path -LiteralPath $transaction) { throw '存在未完成的升级，先执行回退恢复。' }
        $pin = Read-ProjectSourcePin $PinPath
        New-Item -ItemType Directory -Path $transaction | Out-Null
        $archive = Resolve-ProjectInstallChild $transaction 'source.zip'
        Get-ProjectInstallArchive $archive $ArchivePath $false 536870912 $pin
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -cne $pin.archiveSha256) { throw '升级源码包 SHA-256 校验失败。' }
        $staging = Resolve-ProjectInstallChild $transaction 'staging'
        Expand-ProjectInstallArchive $archive $staging 2147483648 268435456 50000 $pin.archiveRoot
        Assert-ProjectInstallIdentity $staging
        Write-ProjectInstallationRecord $staging $pin
        $newFiles = @(Get-UpgradeFiles $staging)
        foreach ($relative in $newFiles) { if (Test-ProtectedUpgradePath $relative) { throw '升级包包含受保护的本地数据或运行环境。' } }
        $allFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($relative in $newFiles) { $allFiles.Add($relative) | Out-Null }
        $obsolete = @()
        $oldInventory = Resolve-ProjectInstallChild $Root 'release/source-files.json'
        if (Test-Path -LiteralPath $oldInventory -PathType Leaf) {
            $inventory = Get-Content -LiteralPath $oldInventory -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($inventory.product -cne 'DFL-PT-WEBUI' -or @($inventory.files).Count -gt 50000) { throw '旧源码清单无效，已暂停升级。' }
            foreach ($record in $inventory.files) {
                $relative = [string]$record.path
                if (Test-ProtectedUpgradePath $relative) { throw '旧源码清单包含受保护路径。' }
                $oldFile = Resolve-ProjectInstallChild $Root $relative
                if (-not $allFiles.Contains($relative) -and (Test-Path -LiteralPath $oldFile -PathType Leaf)) {
                    $allFiles.Add($relative) | Out-Null
                    $obsolete += $relative
                }
            }
        }
        # Include build outputs so a failed rebuild returns to the previous UI.
        if (Test-Path -LiteralPath (Join-Path $Root 'webui/dist')) {
            foreach ($relative in Get-UpgradeFiles (Join-Path $Root 'webui/dist')) { $allFiles.Add('webui/dist/' + $relative) | Out-Null }
        }
        $records = @()
        foreach ($relative in $allFiles) {
            $destination = Resolve-ProjectInstallChild $Root $relative
            if (Test-Path -LiteralPath $destination -PathType Container) { throw '新源码文件与现有目录冲突。' }
            $exists = Test-Path -LiteralPath $destination -PathType Leaf
            $digest = $null
            if ($exists) {
                $backup = Resolve-ProjectInstallChild $transaction ('backup/' + $relative)
                New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
                Copy-Item -LiteralPath $destination -Destination $backup
                $digest = (Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            $records += @{path=$relative;existed=$exists;sha256=$digest}
        }
        $journal = @{schemaVersion=1;product='DFL-PT-WEBUI';status='prepared';sourceCommit=$pin.sourceCommit;files=$records;dependenciesMoved=$false}
        [IO.File]::WriteAllText($journalPath,($journal | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
        Assert-UpgradeIdle $Root
        $dependencies = Resolve-ProjectInstallChild $Root 'webui/node_modules'
        if (Test-Path -LiteralPath $dependencies) { Move-ProjectInstallEntry $dependencies (Resolve-ProjectInstallChild $transaction 'original-node_modules') }
        $journal.dependenciesMoved = $true
        [IO.File]::WriteAllText($journalPath,($journal | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
        foreach ($relative in $newFiles) {
            $source = Resolve-ProjectInstallChild $staging $relative
            $destination = Resolve-ProjectInstallChild $Root $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
        foreach ($relative in $obsolete) { Remove-Item -LiteralPath (Resolve-ProjectInstallChild $Root $relative) -Force }
        $journal.status = 'applied'
        [IO.File]::WriteAllText($journalPath,($journal | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
        Write-ProjectInstallEvent 'upgrade' 'complete' 80 '固定版本源码已应用；正在等待构建与健康检查。'
    } catch {
        # Download/path rejection happens before mutations. Keep its evidence in
        # a separate session so a retry cannot mistake it for a resumable update.
        if ($Action -eq 'apply' -and (Test-Path -LiteralPath $transaction) -and -not (Test-Path -LiteralPath $journalPath)) {
            Move-ProjectInstallEntry $transaction (Resolve-ProjectInstallChild $state ('upgrade-rejected-' + [Guid]::NewGuid().ToString('N')))
        }
        throw
    } finally { $lock.Dispose() }
}

if ($MyInvocation.InvocationName -ne '.') {
    try { Invoke-ProjectUpgrade $ProjectRoot $Action $SourcePinPath $SourceArchivePath; exit 0 }
    catch { Write-ProjectInstallEvent 'upgrade' 'failed' 0 $_.Exception.Message; exit 1 }
}
