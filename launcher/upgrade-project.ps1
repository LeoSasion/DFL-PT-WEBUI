[CmdletBinding()]
param([string]$ProjectRoot, [ValidateSet('apply','rollback','complete','guard','status')][string]$Action='apply',
    [string]$SourcePinPath='', [string]$SourceArchivePath='')
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'install-project.ps1') -ProjectRoot $ProjectRoot -SourcePinPath $SourcePinPath -SourceArchivePath $SourceArchivePath

# A source update never replaces runtimes, datasets, models, credentials, or
# local configuration. The retained transaction includes every replaced byte.
function Test-ProtectedUpgradePath([string]$Relative) {
    if([string]::IsNullOrWhiteSpace($Relative) -or $Relative.Length -gt 512 -or $Relative -match '[\\:\x00-\x1f<>"|?*]'){return $true}
    foreach($part in $Relative.Split('/')){
        if([string]::IsNullOrWhiteSpace($part) -or $part -in @('.','..') -or $part.EndsWith('.') -or $part.EndsWith(' ') -or
            $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)'){return $true}
    }
    return $Relative -match '^(?i:workspace|workspaces|\.git|\.venv|\.runtime|\.launcher-install|release-output)(/|$)' -or
        $Relative -match '^(?i:_internal/(node|python_base|git|ffmpeg|config\.txt|vision_models|model_generic_xseg)|webui/(node_modules|\.runtime|\.env[^/]*))(/|$)' -or
        $Relative -match '^(?i:_internal/DeepFaceLab/(pretrained/.*\.(pt|pth|safetensors)|facelib/[^/]+\.(npy|pth)))$' -or
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

function Get-UpgradeProperty([object]$Value,[string]$Name,$Default=$null) {
    if($Value -is [Collections.IDictionary]){if($Value.Contains($Name)){return $Value[$Name]};return $Default}
    if($Value.PSObject.Properties.Name -contains $Name){return $Value.$Name}
    return $Default
}
function Test-ProjectSourceContentHash([string]$Path,[string]$Expected) {
    if($Expected -cnotmatch '^[0-9a-f]{64}$'){return $false}
    $bytes=[IO.File]::ReadAllBytes($Path)
    $hasher=[Security.Cryptography.SHA256]::Create()
    try {
        $raw=[BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-','').ToLowerInvariant()
        if($raw -ceq $Expected){return $true}
        # Match release/source-integrity.mjs: UTF-8 text with no NUL in
        # its first 8000 bytes uses canonical LF, including Windows scripts.
        for($i=0;$i -lt [Math]::Min(8000,$bytes.Length);$i++){if($bytes[$i] -eq 0){return $false}}
        try{$text=(New-Object Text.UTF8Encoding($false,$true)).GetString($bytes)}catch{return $false}
        $canonical=[Text.Encoding]::UTF8.GetBytes($text.Replace("`r`n","`n"))
        return [BitConverter]::ToString($hasher.ComputeHash($canonical)).Replace('-','').ToLowerInvariant() -ceq $Expected
    } finally {$hasher.Dispose()}
}
function Get-ProjectNumericVersion([string]$Value) {
    if($Value -match '^(\d+\.\d+\.\d+)(?:-[a-zA-Z0-9.-]+)?$'){
        $parsed=$null
        if([Version]::TryParse($Matches[1],[ref]$parsed)){return $parsed}
    }
    return $null
}
function Set-UpgradeProperty([object]$Value,[string]$Name,$Content) {
    if($Value -is [Collections.IDictionary]){$Value[$Name]=$Content}
    else{$Value | Add-Member -MemberType NoteProperty -Name $Name -Value $Content -Force}
}
function Read-ProjectUpgradeJournal([string]$Root,[string]$Transaction) {
    $path=Resolve-ProjectInstallChild $Transaction 'journal.json'
    $journal=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    if($journal.schemaVersion -ne 1 -or $journal.product -cne 'DFL-PT-WEBUI' -or $journal.sourceCommit -cnotmatch '^[0-9a-f]{40}$' -or
        $journal.status -notin @('prepared','applying','applied','rolling-back','rolled-back','completing','completed') -or @($journal.files).Count -eq 0 -or @($journal.files).Count -gt 50000){throw '升级事务标识不正确。'}
    $seen=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    if($journal.dependenciesMoved -isnot [bool]){throw '升级依赖移动记录不正确。'}
    if($journal.PSObject.Properties.Name -contains 'dependenciesExisted' -and $journal.dependenciesExisted -isnot [bool]){throw '升级依赖归属记录不正确。'}
    $rollback=Get-UpgradeProperty $journal 'rollback'
    if($rollback -and ($rollback.dependencies -notin @('pending','intent','done') -or $rollback.dist -notin @('pending','intent','done'))){throw '升级回退阶段记录不正确。'}
    foreach($file in $journal.files){
        $relative=[string]$file.path
        if(-not $relative -or $relative -match '[\\:\x00-\x1f<>"|?*]' -or $relative.Split('/') -contains '..' -or $relative.Split('/') -contains '.' -or
            $relative.Split('/') -contains '' -or
            -not $seen.Add($relative) -or (Test-ProtectedUpgradePath $relative) -or $file.existed -isnot [bool]){throw '升级事务包含无效或受保护路径。'}
        Resolve-ProjectInstallChild $Root $relative | Out-Null
        if($file.existed){
            $backup=Resolve-ProjectInstallChild $Transaction ('backup/'+$relative)
            if($file.sha256 -cnotmatch '^[0-9a-f]{64}$' -or -not (Test-Path -LiteralPath $backup -PathType Leaf) -or
                (Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.sha256){throw '回退备份校验失败，请保留安装缓存；当前源码与依赖未移动。'}
        }
        if($file.PSObject.Properties.Name -notcontains 'newSha256'){
            $staged=Resolve-ProjectInstallChild $Transaction ('staging/'+$relative)
            Set-UpgradeProperty $file 'newSha256' $(if(Test-Path -LiteralPath $staged -PathType Leaf){(Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null})
        }
        if($file.newSha256 -and $file.newSha256 -cnotmatch '^[0-9a-f]{64}$'){throw '升级发布校验记录不正确。'}
    }
    return $journal
}
function Get-ProjectUpgradeTransaction([string]$State) {
    $current=Resolve-ProjectInstallChild $State 'upgrade-current'
    if(Test-Path -LiteralPath (Resolve-ProjectInstallChild $current 'journal.json')){return $current}
    $pointerPath=Resolve-ProjectInstallChild $State 'upgrade-last.json'
    if(Test-Path -LiteralPath $pointerPath -PathType Leaf){
        $pointer=Get-Content -LiteralPath $pointerPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if($pointer.schemaVersion -ne 1 -or $pointer.product -cne 'DFL-PT-WEBUI' -or $pointer.history -cnotmatch '^upgrade-(backup|rollback)-[0-9a-f]{32}$' -or
            $pointer.status -notin @('completing','complete','rolled-back')){throw '升级完成记录不正确。'}
        if($pointer.status -eq 'completing'){return (Resolve-ProjectInstallChild $State $pointer.history)}
    }
    return $null
}
function Write-ProjectUpgradePointer([string]$State,[string]$Status,[string]$History) {
    Write-ProjectInstallJson (Resolve-ProjectInstallChild $State 'upgrade-last.json') @{schemaVersion=1;product='DFL-PT-WEBUI';status=$Status;history=$History}
}
function Read-ProjectUpgradePreparation([string]$Transaction) {
    $path=Resolve-ProjectInstallChild $Transaction 'preparation.json'
    if(-not (Test-Path -LiteralPath $path -PathType Leaf)){throw '升级暂存目录缺少归属记录，已保留供检查。'}
    $record=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    if($record.schemaVersion -ne 1 -or $record.product -cne 'DFL-PT-WEBUI' -or $record.sourceCommit -cnotmatch '^[0-9a-f]{40}$' -or $record.archiveSha256 -cnotmatch '^[0-9a-f]{64}$'){throw '升级准备记录不正确。'}
    return $record
}
function Publish-ProjectUpgradeFile([string]$Source,[string]$Destination,[string]$Transaction,[string]$Relative) {
    $temporary=Resolve-ProjectInstallChild $Transaction ('publish/'+$Relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $temporary) -Force | Out-Null
    $digest=(Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
    if(Test-Path -LiteralPath $temporary){
        if(-not (Test-Path -LiteralPath $temporary -PathType Leaf) -or (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -cne $digest){throw '升级临时文件已变化，已保留供检查。'}
    }else{Copy-Item -LiteralPath $Source -Destination $temporary}
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Assert-ProjectInstallPlainPath $Destination | Out-Null
    if(Test-Path -LiteralPath $Destination -PathType Leaf){[IO.File]::Replace($temporary,$Destination,[System.Management.Automation.Language.NullString]::Value)}
    else{[IO.File]::Move($temporary,$Destination)}
}
function Restore-ProjectUpgrade([string]$Root,[string]$State,[string]$Transaction,[object]$Journal) {
    $journalPath=Resolve-ProjectInstallChild $Transaction 'journal.json'
    # New files may be removed only while they still match our publication.
    # Changed pre-existing source is retained separately before restoring it.
    foreach($file in $Journal.files){
        if($file.path -like 'webui/dist/*'){continue}
        $destination=Resolve-ProjectInstallChild $Root $file.path
        if(Test-Path -LiteralPath $destination -PathType Container){throw '当前源码路径已变为目录，回退已停止。'}
        if(Test-Path -LiteralPath $destination -PathType Leaf){
            $current=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
            if(-not $file.existed -and (-not $file.newSha256 -or $current -cne $file.newSha256)){throw '新源码已被修改，回退不会删除这些文件；请先保留修改并检查。'}
            if($file.existed -and $current -cne $file.sha256 -and $current -cne $file.newSha256){
                $retained=Resolve-ProjectInstallChild $Transaction ('failed-source/'+$file.path)
                New-Item -ItemType Directory -Path (Split-Path -Parent $retained) -Force | Out-Null
                if(Test-Path -LiteralPath $retained){if((Get-FileHash -LiteralPath $retained -Algorithm SHA256).Hash.ToLowerInvariant() -cne $current){throw '回退保留文件已存在且不同，当前修改未覆盖。'}}
                else{Copy-Item -LiteralPath $destination -Destination $retained}
            }
        }
    }
    $rollback=Get-UpgradeProperty $Journal 'rollback'
    if(-not $rollback){
        foreach($name in @('failed-dist','failed-node_modules')){if(Test-Path -LiteralPath (Resolve-ProjectInstallChild $Transaction $name)){throw '未知回退暂存目录已存在，请保留缓存后检查。'}}
        $rollback=[pscustomobject]@{dependencies='pending';dist='pending'};Set-UpgradeProperty $Journal 'rollback' $rollback
    }
    $Journal.status='rolling-back';Write-ProjectInstallJson $journalPath $Journal
    $dependencies=Resolve-ProjectInstallChild $Root 'webui/node_modules'
    $original=Resolve-ProjectInstallChild $Transaction 'original-node_modules'
    $failed=Resolve-ProjectInstallChild $Transaction 'failed-node_modules'
    if($rollback.dependencies -ne 'done'){
        if($rollback.dependencies -notin @('pending','intent')){throw '依赖回退阶段无效。'}
        $rollback.dependencies='intent';Write-ProjectInstallJson $journalPath $Journal
        if(Test-Path -LiteralPath $original){
            if(Test-Path -LiteralPath $dependencies){if(Test-Path -LiteralPath $failed){throw '依赖回退出现冲突，所有目录均已保留。'};Move-ProjectInstallEntry $dependencies $failed}
            Move-ProjectInstallEntry $original $dependencies
        }elseif(-not (Get-UpgradeProperty $Journal 'dependenciesExisted' $false) -and $Journal.dependenciesMoved -and (Test-Path -LiteralPath $dependencies)){
            if(Test-Path -LiteralPath $failed){throw '依赖回退出现冲突，所有目录均已保留。'};Move-ProjectInstallEntry $dependencies $failed
        }
        $rollback.dependencies='done';Write-ProjectInstallJson $journalPath $Journal
    }
    if($rollback.dist -ne 'done'){
        if($rollback.dist -notin @('pending','intent')){throw '构建回退阶段无效。'}
        $rollback.dist='intent';Write-ProjectInstallJson $journalPath $Journal
        $dist=Resolve-ProjectInstallChild $Root 'webui/dist';$failedDist=Resolve-ProjectInstallChild $Transaction 'failed-dist'
        if(Test-Path -LiteralPath $dist){if(Test-Path -LiteralPath $failedDist){throw '构建回退出现冲突，所有目录均已保留。'};Move-ProjectInstallEntry $dist $failedDist}
        $rollback.dist='done';Write-ProjectInstallJson $journalPath $Journal
    }
    foreach($file in $Journal.files){
        $destination=Resolve-ProjectInstallChild $Root $file.path
        if($file.existed){Publish-ProjectUpgradeFile (Resolve-ProjectInstallChild $Transaction ('backup/'+$file.path)) $destination $Transaction ('rollback/'+$file.path)}
        elseif(Test-Path -LiteralPath $destination -PathType Leaf){
            if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.newSha256){throw '新源码已变化，回退未删除。'}
            Remove-Item -LiteralPath $destination -Force
        }
    }
    Assert-ProjectInstallIdentity $Root
    $Journal.status='rolled-back';Write-ProjectInstallJson $journalPath $Journal
    $history='upgrade-rollback-'+[Guid]::NewGuid().ToString('N')
    Write-ProjectUpgradePointer $State 'rolled-back' $history
    Move-ProjectInstallEntry $Transaction (Resolve-ProjectInstallChild $State $history)
    try{Write-ProjectInstallEvent 'upgrade' 'rolled-back' 100 '原源码与构建已回退；失败内容和用户修改已保留在安装缓存中。'}catch{}
}

function Complete-ProjectUpgrade([string]$Root,[string]$State,[string]$Transaction,[object]$Journal) {
    if($Journal.status -notin @('applied','completing','completed')){throw '升级事务未通过应用检查。'}
    Assert-ProjectInstallIdentity $Root
    foreach($file in $Journal.files){
        if($file.path -like 'webui/dist/*'){continue}
        $publishedUpgradePath=Resolve-ProjectInstallChild $Root $file.path
        if($file.newSha256){
            if(-not (Test-Path -LiteralPath $publishedUpgradePath -PathType Leaf) -or
                (Get-FileHash -LiteralPath $publishedUpgradePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.newSha256){
                throw '升级应用后的源码或版本记录已变化，不能确认完成；回退备份已保留。'
            }
        }elseif(Test-Path -LiteralPath $publishedUpgradePath){throw '已移除的旧源码重新出现，不能确认完成；回退备份已保留。'}
    }
    $history=Get-UpgradeProperty $Journal 'archiveName'
    if(-not $history){$history='upgrade-backup-'+[Guid]::NewGuid().ToString('N');Set-UpgradeProperty $Journal 'archiveName' $history}
    if($history -cnotmatch '^upgrade-backup-[0-9a-f]{32}$'){throw '升级归档路径不正确。'}
    $Journal.status='completing';Write-ProjectInstallJson (Resolve-ProjectInstallChild $Transaction 'journal.json') $Journal
    Write-ProjectUpgradePointer $State 'completing' $history
    $archived=Resolve-ProjectInstallChild $State $history
    if($Transaction -ine $archived){Move-ProjectInstallEntry $Transaction $archived}
    $Journal.status='completed';Write-ProjectInstallJson (Resolve-ProjectInstallChild $archived 'journal.json') $Journal
    Write-ProjectUpgradePointer $State 'complete' $history
    try{Write-ProjectInstallEvent 'upgrade' 'complete' 100 '升级完成；原源码、构建和配置备份已保留在安装缓存中。'}catch{}
}

function Get-ProjectUpgradeStatus([string]$Root) {
    $state=Resolve-ProjectInstallChild $Root '.launcher-install'
    if(-not (Test-Path -LiteralPath $state)){return @{status='none';canRollback=$false;canComplete=$false;message='没有未完成的源码升级。'}}
    $owner=Resolve-ProjectInstallChild $state 'owner.txt'
    if(-not (Test-Path -LiteralPath $owner -PathType Leaf) -or [IO.File]::ReadAllText($owner) -cne 'DFL-PT-WEBUI install workspace v1'){throw '安装缓存归属不正确。'}
    $transaction=Get-ProjectUpgradeTransaction $state
    if(-not $transaction){
        $preparing=Resolve-ProjectInstallChild $state 'upgrade-current'
        if(Test-Path -LiteralPath $preparing){Read-ProjectUpgradePreparation $preparing | Out-Null;return @{status='preparing';canRollback=$false;canComplete=$false;message='源码准备曾中断，可重新执行升级；原源码尚未修改。'}}
        return @{status='none';canRollback=$false;canComplete=$false;message='没有未完成的源码升级。'}
    }
    $journal=Read-ProjectUpgradeJournal $Root $transaction
    return @{status=$journal.status;canRollback=($journal.status -in @('prepared','applying','applied','rolling-back','rolled-back','completing'));canComplete=($journal.status -in @('applied','completing','completed'));message='检测到源码升级事务，请先完成或回退恢复。'}
}

function Invoke-ProjectUpgrade {
    param([string]$Root, [string]$Action, [string]$PinPath, [string]$ArchivePath)
    $Root = (Assert-ProjectInstallPlainPath $Root).TrimEnd('\','/')
    if($Action -eq 'status'){return (Get-ProjectUpgradeStatus $Root)}
    Assert-UpgradeIdle $Root
    if ($Action -eq 'guard') { return }
    if($Action -notin @('rollback','complete')){Assert-ProjectInstallIdentity $Root}
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
    $lifecycleLocks=New-Object 'System.Collections.Generic.List[IDisposable]'
    $ownsJournal=$false
    try {
        $transaction = Resolve-ProjectInstallChild $state 'upgrade-current'
        $journalPath = Resolve-ProjectInstallChild $transaction 'journal.json'
        # Match the runtime installer and WebUI preparation lock order. A
        # second launcher must not publish source while either flow is active.
        foreach($relative in @('source/source-install.lock','webui-preparation/operation.lock')){
            $path=Resolve-ProjectInstallChild $state $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
            Assert-ProjectInstallPlainPath $path | Out-Null
            try{$lifecycleLocks.Add([IO.File]::Open($path,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None))}
            catch{throw '运行环境安装或 WebUI 准备正在进行，请等待完成后再升级或回退。'}
        }
        if ($Action -eq 'complete') {
            $transaction=Get-ProjectUpgradeTransaction $state
            if(-not $transaction){
                $last=Resolve-ProjectInstallChild $state 'upgrade-last.json'
                if((Test-Path -LiteralPath $last) -and (Get-Content -LiteralPath $last -Raw | ConvertFrom-Json).status -eq 'complete'){return}
                throw '没有待完成的升级事务。'
            }
            Complete-ProjectUpgrade $Root $state $transaction (Read-ProjectUpgradeJournal $Root $transaction)
            return
        }
        if ($Action -eq 'rollback') {
            $transaction=Get-ProjectUpgradeTransaction $state
            if(-not $transaction){
                $last=Resolve-ProjectInstallChild $state 'upgrade-last.json'
                if((Test-Path -LiteralPath $last) -and (Get-Content -LiteralPath $last -Raw | ConvertFrom-Json).status -eq 'rolled-back'){return}
                throw '没有可回退的升级事务。'
            }
            Restore-ProjectUpgrade $Root $state $transaction (Read-ProjectUpgradeJournal $Root $transaction)
            return
        }
        if(Get-ProjectUpgradeTransaction $state){throw '存在未完成的升级，先完成或回退恢复。'}
        if (Test-Path -LiteralPath $transaction) {
            Read-ProjectUpgradePreparation $transaction | Out-Null
            Move-ProjectInstallEntry $transaction (Resolve-ProjectInstallChild $state ('upgrade-rejected-'+[Guid]::NewGuid().ToString('N')))
        }
        $pin = Read-ProjectSourcePin $PinPath
        $currentVersion=Get-ProjectNumericVersion ([string](Get-Content -LiteralPath (Join-Path $Root 'release/version.json') -Raw -Encoding UTF8|ConvertFrom-Json).version)
        $targetVersion=Get-ProjectNumericVersion ([string]$pin.applicationVersion)
        if($currentVersion -and $targetVersion -and $targetVersion -lt $currentVersion){
            throw '目标应用版本旧于当前版本，请使用匹配的新版启动器；不会执行源码降级。'
        }
        New-Item -ItemType Directory -Path $transaction | Out-Null
        Write-ProjectInstallJson (Resolve-ProjectInstallChild $transaction 'preparation.json') @{schemaVersion=1;product='DFL-PT-WEBUI';sourceCommit=$pin.sourceCommit;archiveSha256=$pin.archiveSha256}
        $archive = Get-VerifiedProjectInstallArchive $state $transaction $ArchivePath $false 536870912 $pin
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
                    if(-not (Test-ProjectSourceContentHash $oldFile $record.sha256)){throw '旧源码文件已被修改，升级不会删除这些文件。'}
                    $allFiles.Add($relative) | Out-Null
                    $obsolete += $relative
                }
            }
        }
        # Include build outputs so a failed rebuild returns to the previous UI.
        if (Test-Path -LiteralPath (Join-Path $Root 'webui/dist')) {
            foreach ($relative in Get-UpgradeFiles (Join-Path $Root 'webui/dist')) { $allFiles.Add('webui/dist/' + $relative) | Out-Null }
        }
        if($allFiles.Count -gt 50000){throw '升级恢复清单超过文件数量上限，原源码未修改。'}
        $records=New-Object 'System.Collections.Generic.List[object]'
        $recordByPath=New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
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
            $newPath=Resolve-ProjectInstallChild $staging $relative
            $newDigest=if(Test-Path -LiteralPath $newPath -PathType Leaf){(Get-FileHash -LiteralPath $newPath -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null}
            $record=@{path=$relative;existed=$exists;sha256=$digest;newSha256=$newDigest}
            $records.Add($record);$recordByPath.Add($relative,$record)
        }
        $journal = @{schemaVersion=1;product='DFL-PT-WEBUI';status='prepared';sourceCommit=$pin.sourceCommit;files=$records.ToArray();dependenciesMoved=$false;dependenciesExisted=[bool](Test-Path -LiteralPath (Resolve-ProjectInstallChild $Root 'webui/node_modules'))}
        Write-ProjectInstallJson $journalPath $journal
        $ownsJournal=$true
        Assert-UpgradeIdle $Root
        $dependencies = Resolve-ProjectInstallChild $Root 'webui/node_modules'
        if (Test-Path -LiteralPath $dependencies) { Move-ProjectInstallEntry $dependencies (Resolve-ProjectInstallChild $transaction 'original-node_modules') }
        $journal.dependenciesMoved = $true
        $journal.status='applying';Write-ProjectInstallJson $journalPath $journal
        foreach ($relative in $newFiles) {
            $source = Resolve-ProjectInstallChild $staging $relative
            $destination = Resolve-ProjectInstallChild $Root $relative
            $record=$recordByPath[$relative]
            if($record.existed){if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -cne $record.sha256){throw '备份后源码已被修改，升级未覆盖该修改。'}}
            elseif(Test-Path -LiteralPath $destination){throw '发布期间出现未知同名文件，升级未覆盖。'}
            Publish-ProjectUpgradeFile $source $destination $transaction $relative
        }
        foreach ($relative in $obsolete) { Remove-Item -LiteralPath (Resolve-ProjectInstallChild $Root $relative) -Force }
        $journal.status = 'applied'
        Write-ProjectInstallJson $journalPath $journal
        Write-ProjectInstallEvent 'upgrade' 'complete' 80 '固定版本源码已应用；正在等待构建与健康检查。'
    } catch {
        $failure=$_
        if($Action -eq 'apply' -and $ownsJournal -and (Test-Path -LiteralPath $journalPath)){
            try{Restore-ProjectUpgrade $Root $state $transaction (Read-ProjectUpgradeJournal $Root $transaction)}
            catch{throw ($failure.Exception.Message+'；自动回退未完成，请重试回退：'+$_.Exception.Message)}
        }
        # Download/path rejection happens before mutations. Keep its evidence in
        # a separate session so a retry cannot mistake it for a resumable update.
        if ($Action -eq 'apply' -and (Test-Path -LiteralPath $transaction) -and -not (Test-Path -LiteralPath $journalPath)) {
            Move-ProjectInstallEntry $transaction (Resolve-ProjectInstallChild $state ('upgrade-rejected-' + [Guid]::NewGuid().ToString('N')))
        }
        throw $failure
    } finally {
        for($index=$lifecycleLocks.Count-1;$index -ge 0;$index--){$lifecycleLocks[$index].Dispose()}
        $lock.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    [Console]::OutputEncoding=New-Object Text.UTF8Encoding($false)
    try { $result=Invoke-ProjectUpgrade $ProjectRoot $Action $SourcePinPath $SourceArchivePath;if($Action -eq 'status'){[Console]::Out.WriteLine(($result | ConvertTo-Json -Compress))};exit 0 }
    catch {
        if($Action -eq 'status'){[Console]::Out.WriteLine((@{status='error';canRollback=$false;canComplete=$false;message=$_.Exception.Message} | ConvertTo-Json -Compress))}
        else{Write-ProjectInstallEvent 'upgrade' 'failed' 0 $_.Exception.Message}
        exit 1
    }
}
