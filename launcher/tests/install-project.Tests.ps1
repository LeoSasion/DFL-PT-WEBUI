$script:projectInstallerPath = Join-Path $PSScriptRoot '../install-project.ps1'
. $script:projectInstallerPath
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function New-ProjectSourceFixture([string]$Archive, [string]$Missing='', [object[]]$Extra=@(), [string]$Product='DFL-PT-WEBUI', [string]$ArchiveRoot='DFL-PT-WEBUI-main') {
    $files = [ordered]@{
        'release/version.json' = ('{"schemaVersion":1,"version":"fixture","product":"' + $Product + '"}')
        'requirements.txt' = 'numpy==2.2.6'
        '_internal/DeepFaceLab/me.py' = "import torch`nfrom me_backend.engine import MEEngine`n"
        '_internal/DeepFaceLab/me_backend/engine.py' = "import torch`nclass MEEngine: pass`n"
        '_internal/DeepFaceLab/me_backend/network.py' = "import torch`nclass MENetwork(torch.nn.Module): pass`n"
        '_internal/DeepFaceLab/core/leras/nn.py' = '# PyTorch fixture'
        'webui/package.json' = '{"type":"module","packageManager":"pnpm@11.19.0"}'
        'webui/pnpm-lock.yaml' = 'lockfileVersion: 9.0'
        'webui/pnpm-workspace.yaml' = 'packages: []'
        'webui/scripts/local-manager.mjs' = 'export const fixture = true'
        'webui/server/index.mjs' = 'export const fixture = true'
        'launcher/install-source.ps1' = '# source install fixture'
        'launcher/setup-runtime.ps1' = '# setup fixture'
        'launcher/runtime-manifest.json' = '{"schemaVersion":3}'
        'tools/prepare-vision-runtime.ps1' = '# vision fixture'
        'tools/dist-provenance.mjs' = 'export const fixture = true'
        'install-source.bat' = '@echo off'
        'README.md' = 'Fixture source; no user media or dependencies.'
    }
    $zip = [IO.Compression.ZipFile]::Open($Archive, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($name in $files.Keys) {
            if ($name -eq $Missing) { continue }
            $entry = $zip.CreateEntry($ArchiveRoot + '/' + $name)
            $stream = $entry.Open()
            try {
                $bytes = [Text.Encoding]::UTF8.GetBytes($files[$name])
                $stream.Write($bytes, 0, $bytes.Length)
            } finally { $stream.Dispose() }
        }
        foreach ($record in $Extra) {
            $entry = $zip.CreateEntry([string]$record.name)
            if ($record.ContainsKey('attributes')) { $entry.ExternalAttributes = [int]$record.attributes }
            $stream = $entry.Open()
            try {
                $bytes = [Text.Encoding]::UTF8.GetBytes([string]$record.content)
                $stream.Write($bytes, 0, $bytes.Length)
            } finally { $stream.Dispose() }
        }
    } finally { $zip.Dispose() }
    return $Archive
}

function New-OwnedProjectInstallTarget([string]$Path) {
    $state = Join-Path $Path '.launcher-install'
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $state 'owner.txt'), 'DFL-PT-WEBUI install workspace v1')
    return $state
}

function New-ProjectUpgradeFixture([string]$Base, [string]$Label, [object[]]$Extra=@()) {
    $target = Join-Path $Base ($Label+'-target')
    $oldArchive = New-ProjectSourceFixture (Join-Path $Base ($Label+'-old.zip'))
    if ((Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $oldArchive -NoNetwork) -ne 0) { throw 'Synthetic source install failed.' }
    $sourceRoot = 'DFL-PT-WEBUI-' + ('a'*40)
    $archive = New-ProjectSourceFixture (Join-Path $Base ($Label+'-new.zip')) -ArchiveRoot $sourceRoot -Extra $Extra
    $pin = Join-Path $Base ($Label+'-pin.json')
    @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=$sourceRoot;archiveSha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $pin -Encoding UTF8
    return @{target=$target;archive=$archive;pin=$pin}
}

Describe 'fixed source upgrade transactions' {
    BeforeEach {
        . (Join-Path $PSScriptRoot '../upgrade-project.ps1')
        Mock Assert-UpgradeIdle { }
    }
    It 'rejects Windows aliases in inventories and recovery paths' {
        foreach($relative in @('.runtime./torch-variants/switch.json','workspace./model/tae.pt','_internal/python_base./Lib/site-packages/torch','webui/.env.local.','folder/../README.md','folder//file.txt','NUL.txt')){
            Test-ProtectedUpgradePath $relative | Should Be $true
        }
    }
    It 'backs up source and dependencies, preserves user data and restores after a failed build' {
        $target = Join-Path $TestDrive 'upgrade-target'
        $oldArchive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-old.zip')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $oldArchive -NoNetwork | Should Be 0
        [IO.File]::WriteAllText((Join-Path $target 'README.md'),'old source')
        foreach ($relative in @('workspace/model/checkpoint.dat','workspaces/project/media.png','_internal/config.txt','webui/.env.local','webui/node_modules/original.txt','webui/dist/client/index.html')) {
            $file = Join-Path $target $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
            [IO.File]::WriteAllText($file,'original ' + $relative)
        }
        $sourceRoot = 'DFL-PT-WEBUI-' + ('a' * 40)
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-new.zip') -ArchiveRoot $sourceRoot
        $pin = Join-Path $TestDrive 'upgrade-pin.json'
        @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=$sourceRoot;archiveSha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $pin -Encoding UTF8
        Invoke-ProjectUpgrade $target 'apply' $pin $archive
        [IO.File]::ReadAllText((Join-Path $target 'README.md')) | Should Match 'Fixture source'
        Test-Path (Join-Path $target '.launcher-install/upgrade-current/original-node_modules/original.txt') | Should Be $true
        Test-Path (Join-Path $target 'webui/node_modules') | Should Be $false
        $built = Join-Path $target 'webui/dist/client/failed-new.js'
        [IO.File]::WriteAllText($built,'new build artifact')
        Invoke-ProjectUpgrade $target 'rollback' $pin ''
        [IO.File]::ReadAllText((Join-Path $target 'README.md')) | Should Be 'old source'
        Test-Path $built | Should Be $false
        foreach ($relative in @('workspace/model/checkpoint.dat','workspaces/project/media.png','_internal/config.txt','webui/.env.local','webui/node_modules/original.txt','webui/dist/client/index.html')) {
            [IO.File]::ReadAllText((Join-Path $target $relative)) | Should Be ('original ' + $relative)
        }
        Test-Path (Join-Path $target 'release/installation.json') | Should Be $false
    }
    It 'rejects a wrong archive digest before source mutation and leaves retries possible' {
        $target = Join-Path $TestDrive 'upgrade-hash-target'
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-hash.zip')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        $pin = Join-Path $TestDrive 'wrong-pin.json'
        @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=('DFL-PT-WEBUI-'+('a'*40));archiveSha256=('0'*64)} | ConvertTo-Json | Set-Content -LiteralPath $pin -Encoding UTF8
        { Invoke-ProjectUpgrade $target 'apply' $pin $archive } | Should Throw
        Test-Path (Join-Path $target '.launcher-install/upgrade-current') | Should Be $false
        @(Get-ChildItem (Join-Path $target '.launcher-install') -Filter 'upgrade-rejected-*').Count | Should Be 1
        [IO.File]::ReadAllText((Join-Path $target 'README.md')) | Should Match 'Fixture source'
    }
    It 'refuses damaged rollback backups before moving current build or dependencies' {
        $target = Join-Path $TestDrive 'upgrade-corrupt-target'
        $oldArchive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-corrupt-old.zip')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $oldArchive -NoNetwork | Should Be 0
        foreach ($relative in @('webui/node_modules/original.txt','webui/dist/client/index.html')) {
            $file = Join-Path $target $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
            [IO.File]::WriteAllText($file,'original')
        }
        $sourceRoot = 'DFL-PT-WEBUI-' + ('a' * 40)
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-corrupt-new.zip') -ArchiveRoot $sourceRoot
        $pin = Join-Path $TestDrive 'upgrade-corrupt-pin.json'
        @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=$sourceRoot;archiveSha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $pin -Encoding UTF8
        Invoke-ProjectUpgrade $target 'apply' $pin $archive
        $currentDependencies = Join-Path $target 'webui/node_modules/current.txt'
        New-Item -ItemType Directory -Path (Split-Path -Parent $currentDependencies) -Force | Out-Null
        [IO.File]::WriteAllText($currentDependencies,'current dependencies')
        $currentBuild = Join-Path $target 'webui/dist/client/index.html'
        [IO.File]::WriteAllText($currentBuild,'current build')
        [IO.File]::WriteAllText((Join-Path $target '.launcher-install/upgrade-current/backup/README.md'),'corrupted')
        { Invoke-ProjectUpgrade $target 'rollback' $pin '' } | Should Throw
        [IO.File]::ReadAllText($currentDependencies) | Should Be 'current dependencies'
        [IO.File]::ReadAllText($currentBuild) | Should Be 'current build'
        Test-Path (Join-Path $target '.launcher-install/upgrade-current/failed-dist') | Should Be $false
    }
    It 'blocks a project process before obtaining or changing source' {
        $target = Join-Path $TestDrive 'upgrade-busy-target'
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'upgrade-busy.zip')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        Mock Assert-UpgradeIdle { throw 'running task' }
        { Invoke-ProjectUpgrade $target 'apply' 'missing-pin' 'missing.zip' } | Should Throw 'running task'
        Test-Path (Join-Path $target '.launcher-install/upgrade-current') | Should Be $false
    }
    It 'recovers a damaged current identity using the verified rollback receipt' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'damaged-identity'
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        [IO.File]::WriteAllText((Join-Path $fixture.target 'release/version.json'),'{interrupted source write')
        { Invoke-ProjectUpgrade $fixture.target 'rollback' $fixture.pin '' } | Should Not Throw
        Assert-ProjectInstallIdentity $fixture.target
    }
    It 'resumes rollback after current dependencies were retained but original restoration failed' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'rollback-resume'
        $old=Join-Path $fixture.target 'webui/node_modules/original.txt'
        New-Item -ItemType Directory -Path (Split-Path -Parent $old) -Force | Out-Null
        [IO.File]::WriteAllText($old,'original dependency')
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        $current=Join-Path $fixture.target 'webui/node_modules/current.txt'
        New-Item -ItemType Directory -Path (Split-Path -Parent $current) -Force | Out-Null
        [IO.File]::WriteAllText($current,'new dependency')
        $original=Join-Path $fixture.target '.launcher-install/upgrade-current/original-node_modules'
        $script:rollbackInterrupted=$false
        Mock Move-ProjectInstallEntry {
            if(-not $script:rollbackInterrupted){$script:rollbackInterrupted=$true;throw 'synthetic restoration interrupted'}
            [IO.Directory]::Move($Source,$Destination)
        } -ParameterFilter {$Source -eq $original}
        { Invoke-ProjectUpgrade $fixture.target 'rollback' $fixture.pin '' } | Should Throw
        { Invoke-ProjectUpgrade $fixture.target 'rollback' $fixture.pin '' } | Should Not Throw
        [IO.File]::ReadAllText($old) | Should Be 'original dependency'
    }
    It 'refuses to remove a newly published source file edited after upgrade' {
        $name='added-by-upgrade.txt'
        $fixture=New-ProjectUpgradeFixture $TestDrive 'changed-new-file' -Extra @(@{name=('DFL-PT-WEBUI-'+('a'*40)+'/'+$name);content='official new source'})
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        $changed=Join-Path $fixture.target $name
        [IO.File]::WriteAllText($changed,'user edited this after upgrade')
        { Invoke-ProjectUpgrade $fixture.target 'rollback' $fixture.pin '' } | Should Throw
        [IO.File]::ReadAllText($changed) | Should Be 'user edited this after upgrade'
    }
    It 'rejects runtime transaction data in a source upgrade before publishing source' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'runtime-protection' -Extra @(@{name=('DFL-PT-WEBUI-'+('a'*40)+'/.runtime/torch-variants/switch.json');content='must not publish'})
        { Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive } | Should Throw
        Test-Path -LiteralPath (Join-Path $fixture.target '.runtime/torch-variants/switch.json') | Should Be $false
    }
    It 'does not reset a Git working copy or its edited source' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'git-protection'
        New-Item -ItemType Directory -Path (Join-Path $fixture.target '.git') | Out-Null
        $edited=Join-Path $fixture.target 'README.md';[IO.File]::WriteAllText($edited,'dirty source remains')
        { Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive } | Should Throw 'Git'
        [IO.File]::ReadAllText($edited) | Should Be 'dirty source remains'
    }
    It 'reads recovery status without importing or requiring the current source identity' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'status-identity'
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        $version=Join-Path $fixture.target 'release/version.json';[IO.File]::WriteAllText($version,'broken identity')
        $status=Invoke-ProjectUpgrade $fixture.target 'status' '' ''
        $status.status | Should Be 'applied';$status.canRollback | Should Be $true
        [IO.File]::ReadAllText($version) | Should Be 'broken identity'
    }
    It 'automatically restores source after a publication failure and allows repeated rollback' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'auto-rollback'
        [IO.File]::WriteAllText((Join-Path $fixture.target 'README.md'),'original source')
        Mock Publish-ProjectUpgradeFile {throw 'synthetic source publication failure'} -ParameterFilter {$Relative -eq 'README.md' -and $Transaction -like '*auto-rollback-target*'}
        {Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive} | Should Throw 'synthetic'
        [IO.File]::ReadAllText((Join-Path $fixture.target 'README.md')) | Should Be 'original source'
        {Invoke-ProjectUpgrade $fixture.target 'rollback' '' ''} | Should Not Throw
        (Invoke-ProjectUpgrade $fixture.target 'status' '' '').canRollback | Should Be $false
    }
    It 'completes idempotently across both sides of the archive rename' {
        foreach($afterMove in @($false,$true)){
            $fixture=New-ProjectUpgradeFixture $TestDrive ('complete-'+$afterMove)
            Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
            $current=Join-Path $fixture.target '.launcher-install/upgrade-current';$script:archiveInterrupted=$false
            Mock Move-ProjectInstallEntry {
                if(-not $script:archiveInterrupted){$script:archiveInterrupted=$true;if($afterMove){[IO.Directory]::Move($Source,$Destination)};throw 'synthetic archive interruption'}
                [IO.Directory]::Move($Source,$Destination)
            } -ParameterFilter {$Source -eq $current -and (Split-Path -Leaf $Destination) -like 'upgrade-backup-*'}
            {Invoke-ProjectUpgrade $fixture.target 'complete' '' ''} | Should Throw 'synthetic'
            (Invoke-ProjectUpgrade $fixture.target 'status' '' '').canComplete | Should Be $true
            {Invoke-ProjectUpgrade $fixture.target 'complete' '' ''} | Should Not Throw
            {Invoke-ProjectUpgrade $fixture.target 'complete' '' ''} | Should Not Throw
            (Invoke-ProjectUpgrade $fixture.target 'status' '' '').status | Should Be 'none'
        }
    }
    It 'keeps a completed upgrade committed when the notification consumer disconnects' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'complete-notification'
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        Mock Write-ProjectInstallEvent {throw 'synthetic disconnected consumer'} -ParameterFilter {$Id -eq 'upgrade' -and $Progress -eq 100 -and $Status -eq 'complete'}
        {Invoke-ProjectUpgrade $fixture.target 'complete' '' ''} | Should Not Throw
        (Invoke-ProjectUpgrade $fixture.target 'status' '' '').status | Should Be 'none'
        Assert-ProjectInstallIdentity $fixture.target
    }
    It 'resumes rollback after the current build was renamed before its done receipt' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'dist-intent'
        $dist=Join-Path $fixture.target 'webui/dist';New-Item -ItemType Directory -Path $dist -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $dist 'original.txt'),'old build')
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        $script:distInterrupted=$false
        Mock Move-ProjectInstallEntry {[IO.Directory]::Move($Source,$Destination);if(-not $script:distInterrupted){$script:distInterrupted=$true;throw 'synthetic dist intent interruption'}} -ParameterFilter {$Source -eq $dist}
        {Invoke-ProjectUpgrade $fixture.target 'rollback' '' ''} | Should Throw 'synthetic'
        {Invoke-ProjectUpgrade $fixture.target 'rollback' '' ''} | Should Not Throw
        [IO.File]::ReadAllText((Join-Path $dist 'original.txt')) | Should Be 'old build'
    }
    It 'preserves interrupted preparation evidence and retries without a source journal' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'prepare-intent'
        $current=Join-Path $fixture.target '.launcher-install/upgrade-current';New-Item -ItemType Directory -Path $current | Out-Null
        $pin=Get-Content -LiteralPath $fixture.pin -Raw | ConvertFrom-Json
        Write-ProjectInstallJson (Join-Path $current 'preparation.json') @{schemaVersion=1;product='DFL-PT-WEBUI';sourceCommit=$pin.sourceCommit;archiveSha256=$pin.archiveSha256}
        [IO.File]::WriteAllText((Join-Path $current 'unknown-evidence.txt'),'preserve unknown')
        (Invoke-ProjectUpgrade $fixture.target 'status' '' '').status | Should Be 'preparing'
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        $retained=@(Get-ChildItem -LiteralPath (Join-Path $fixture.target '.launcher-install') -Filter 'upgrade-rejected-*')
        [IO.File]::ReadAllText((Join-Path $retained[0].FullName 'unknown-evidence.txt')) | Should Be 'preserve unknown'
    }
    It 'refuses source mutation while runtime installation or WebUI preparation owns its lock' {
        foreach($relative in @('source/source-install.lock','webui-preparation/operation.lock')){
            $fixture=New-ProjectUpgradeFixture $TestDrive ('cross-lock-'+$relative.Replace('/','-').Replace('.','-'))
            $path=Join-Path (Join-Path $fixture.target '.launcher-install') $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
            $held=[IO.File]::Open($path,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
            try{{Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive} | Should Throw}
            finally{$held.Dispose()}
            Test-Path -LiteralPath (Join-Path $fixture.target '.launcher-install/upgrade-current/journal.json') | Should Be $false
        }
    }
    It 'does not delete an obsolete source file changed since its ownership inventory' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'changed-obsolete'
        $obsolete=Join-Path $fixture.target 'obsolete-owned.txt';[IO.File]::WriteAllText($obsolete,'old official source')
        $digest=(Get-FileHash -LiteralPath $obsolete -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-ProjectInstallJson (Join-Path $fixture.target 'release/source-files.json') @{schemaVersion=1;product='DFL-PT-WEBUI';files=@(@{path='obsolete-owned.txt';sha256=$digest})}
        [IO.File]::WriteAllText($obsolete,'user modified obsolete source')
        {Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive} | Should Throw '修改'
        [IO.File]::ReadAllText($obsolete) | Should Be 'user modified obsolete source'
    }
    It 'returns a single UTF8 status JSON object for both valid and rejected recovery evidence' {
        $fixture=New-ProjectUpgradeFixture $TestDrive 'status-protocol'
        Invoke-ProjectUpgrade $fixture.target 'apply' $fixture.pin $fixture.archive
        foreach($damaged in @($false,$true)){
            if($damaged){[IO.File]::WriteAllText((Join-Path $fixture.target '.launcher-install/upgrade-current/backup/README.md'),'damaged backup')}
            $start=New-Object Diagnostics.ProcessStartInfo
            $start.FileName=[Diagnostics.Process]::GetCurrentProcess().MainModule.FileName;$start.UseShellExecute=$false;$start.CreateNoWindow=$true
            $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true;$start.StandardOutputEncoding=[Text.Encoding]::UTF8;$start.StandardErrorEncoding=[Text.Encoding]::UTF8
            $start.Arguments='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../upgrade-project.ps1'))+'" -ProjectRoot "'+$fixture.target+'" -Action status'
            $process=[Diagnostics.Process]::Start($start)
            try{
                $output=$process.StandardOutput.ReadToEndAsync();$errors=$process.StandardError.ReadToEndAsync();$process.WaitForExit(30000) | Should Be $true
                $status=$output.GetAwaiter().GetResult() | ConvertFrom-Json
                $errors.GetAwaiter().GetResult() | Should Be ''
                if($damaged){$process.ExitCode | Should Be 1;$status.status | Should Be 'error';$status.canRollback | Should Be $false}
                else{$process.ExitCode | Should Be 0;$status.status | Should Be 'applied';$status.canRollback | Should Be $true;$status.message | Should Match '源码'}
            }finally{if(-not $process.HasExited){$process.Kill();$process.WaitForExit()};$process.Dispose()}
        }
    }
}

Describe 'bounded PT first-install source acquisition' {
    It 'preserves unknown session files rather than recursively clearing them' {
        $target=Join-Path $TestDrive 'unknown-session-target';$state=New-OwnedProjectInstallTarget $target
        $session=Join-Path $state ('source-'+[Guid]::NewGuid().ToString('N'));New-Item -ItemType Directory -Path $session | Out-Null
        Write-ProjectInstallJson (Join-Path $session 'ownership.json') @{schemaVersion=1;files=@('ownership.json','source.zip')}
        $unknown=Join-Path $session 'user.txt';[IO.File]::WriteAllText($unknown,'preserve this file')
        {Remove-ProjectInstallSession $state $session} | Should Throw '未知'
        [IO.File]::ReadAllText($unknown) | Should Be 'preserve this file'
    }
    It 'recovers persisted publication intent after the installing process was killed' {
        $archive=New-ProjectSourceFixture (Join-Path $TestDrive 'crash-publication.zip')
        $target=Join-Path $TestDrive 'crash-publication-target';$marker=Join-Path $TestDrive 'publication-moved.txt'
        $worker=Join-Path $TestDrive 'crash-worker.ps1'
        [IO.File]::WriteAllText($worker,@'
param($Installer,$Target,$Archive,$Marker)
. $Installer
$script:originalMove=(Get-Command Move-ProjectInstallEntry).ScriptBlock
function Move-ProjectInstallEntry([string]$Source,[string]$Destination){
 & $script:originalMove $Source $Destination
 if($Source -match '[\\/]staging[\\/]'){
  [IO.File]::WriteAllText($Marker,'owned publication move completed')
  Start-Sleep -Seconds 30
 }
}
$code=Invoke-ProjectSourceInstall -ProjectRoot $Target -SourceArchivePath $Archive -NoNetwork
exit $code
'@)
        $start=New-Object Diagnostics.ProcessStartInfo
        $start.FileName=[Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
        $start.Arguments='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$worker+'" "'+[IO.Path]::GetFullPath($script:projectInstallerPath)+'" "'+$target+'" "'+$archive+'" "'+$marker+'"'
        $process=[Diagnostics.Process]::Start($start)
        try {
            $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
            $timer=[Diagnostics.Stopwatch]::StartNew()
            while(-not (Test-Path -LiteralPath $marker) -and -not $process.HasExited -and $timer.Elapsed.TotalSeconds -lt 15){Start-Sleep -Milliseconds 50}
            Test-Path -LiteralPath $marker | Should Be $true
            if(-not $process.HasExited){$process.Kill();$process.WaitForExit()}
        }finally{if(-not $process.HasExited){$process.Kill();$process.WaitForExit()};$process.Dispose()}
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        Assert-ProjectInstallIdentity $target
    }
    It 'reuses a verified pinned archive after a publication failure without its original input' {
        $archiveRoot='DFL-PT-WEBUI-'+('a'*40)
        $archive=New-ProjectSourceFixture (Join-Path $TestDrive 'cached-publication.zip') -ArchiveRoot $archiveRoot
        $pin=Join-Path $TestDrive 'cached-publication-pin.json'
        @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=$archiveRoot;archiveSha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $pin -Encoding UTF8
        $target=Join-Path $TestDrive 'cached-publication-target';$script:publicationFailed=$false
        Mock Move-ProjectInstallEntry {
            if(-not $script:publicationFailed){$script:publicationFailed=$true;throw 'synthetic publication interruption'}
            if([IO.Directory]::Exists($Source)){[IO.Directory]::Move($Source,$Destination)}else{[IO.File]::Move($Source,$Destination)}
        } -ParameterFilter {$Source -match '[\\/]staging[\\/]'}
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -SourcePinPath $pin -NoNetwork | Should Be 1
        Remove-Item -LiteralPath $archive -Force
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -SourcePinPath $pin -NoNetwork | Should Be 0
    }
    It 'installs the native PT archive without Git or runtime dependencies' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'source.zip')
        $target = Join-Path $TestDrive 'clean-target'
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        [IO.File]::ReadAllText((Join-Path $target 'README.md')) | Should Match 'Fixture source'
        Test-Path -LiteralPath (Join-Path $target '_internal/DeepFaceLab/me_backend/engine.py') | Should Be $true
        Test-Path -LiteralPath (Join-Path $target 'DFL-PT-WEBUI-main') | Should Be $false
        Test-Path -LiteralPath (Join-Path $target '.venv') | Should Be $false
        Test-Path -LiteralPath (Join-Path $target '.git') | Should Be $false
        @(Get-ChildItem -LiteralPath (Join-Path $target '.launcher-install') -Filter 'source-*').Count | Should Be 0
        Test-Path -LiteralPath $archive | Should Be $true
    }

    It 'preserves owned installer logs and caches while publishing source' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'resume.zip')
        $target = Join-Path $TestDrive 'owned-target'
        $state = New-OwnedProjectInstallTarget $target
        [IO.File]::WriteAllText((Join-Path $state 'install.log'), 'preserve this log')
        New-Item -ItemType Directory -Path (Join-Path $state 'runtime') | Out-Null
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        [IO.File]::ReadAllText((Join-Path $state 'install.log')) | Should Be 'preserve this log'
        Test-Path -LiteralPath (Join-Path $state 'runtime') | Should Be $true
    }

    It 'validates an existing complete PT project without replacing its files' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'existing.zip')
        $target = Join-Path $TestDrive 'existing-target'
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 0
        [IO.File]::WriteAllText((Join-Path $target 'README.md'), 'user edited README')
        [IO.File]::WriteAllText((Join-Path $target 'personal.txt'), 'user original')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath (Join-Path $TestDrive 'missing.zip') -NoNetwork | Should Be 0
        [IO.File]::ReadAllText((Join-Path $target 'README.md')) | Should Be 'user edited README'
        [IO.File]::ReadAllText((Join-Path $target 'personal.txt')) | Should Be 'user original'
    }

    It 'refuses old DFL product identity and missing PT source before publication' {
        foreach ($case in @('legacy','missing')) {
            $archive = Join-Path $TestDrive ($case + '.zip')
            if ($case -eq 'legacy') { New-ProjectSourceFixture $archive -Product 'DFL-WEBUI' | Out-Null }
            else { New-ProjectSourceFixture $archive -Missing '_internal/DeepFaceLab/me_backend/network.py' | Out-Null }
            $target = Join-Path $TestDrive ($case + '-target')
            Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 1
            Test-Path -LiteralPath (Join-Path $target 'release') | Should Be $false
            Test-Path -LiteralPath (Join-Path $target '_internal') | Should Be $false
        }
    }

    It 'refuses nonempty and unowned directories without changing user content' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'occupied.zip')
        $target = Join-Path $TestDrive 'occupied-target'
        New-Item -ItemType Directory -Path $target | Out-Null
        [IO.File]::WriteAllText((Join-Path $target 'personal.txt'), 'keep me')
        Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 1
        [IO.File]::ReadAllText((Join-Path $target 'personal.txt')) | Should Be 'keep me'
        Test-Path -LiteralPath (Join-Path $target '.launcher-install') | Should Be $false
        $unowned = Join-Path $TestDrive 'unowned-target'
        $state = New-OwnedProjectInstallTarget $unowned
        [IO.File]::WriteAllText((Join-Path $state 'owner.txt'), 'different owner')
        Invoke-ProjectSourceInstall -ProjectRoot $unowned -SourceArchivePath $archive -NoNetwork | Should Be 1
        [IO.File]::ReadAllText((Join-Path $state 'owner.txt')) | Should Be 'different owner'
    }

    It 'rejects traversal, alternate streams, Windows devices and case aliases' {
        $names = @('DFL-PT-WEBUI-main/../../outside.txt', 'DFL-PT-WEBUI-main/webui/file:stream',
            'DFL-PT-WEBUI-main/NUL.txt', 'DFL-PT-WEBUI-main/README.MD', '/absolute.txt',
            'DFL-PT-WEBUI-main/folder\\escape.txt', 'DFL-PT-WEBUI-main/.launcher-install/owner.txt')
        for ($i=0; $i -lt $names.Count; $i++) {
            $archive = New-ProjectSourceFixture (Join-Path $TestDrive ('bad-' + $i + '.zip')) -Extra @(@{name=$names[$i];content='bad'})
            $target = Join-Path $TestDrive ('bad-target-' + $i)
            Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 1
            Test-Path -LiteralPath (Join-Path $target 'webui') | Should Be $false
        }
        Test-Path -LiteralPath (Join-Path $TestDrive 'outside.txt') | Should Be $false
    }

    It 'rejects ZIP symlinks and Windows reparse attributes' {
        $attributeCases = @(-1610612736, 1024) # Unix symlink (0xA000 << 16), FILE_ATTRIBUTE_REPARSE_POINT.
        for ($i=0; $i -lt $attributeCases.Count; $i++) {
            $archive = New-ProjectSourceFixture (Join-Path $TestDrive ('link-' + $i + '.zip')) -Extra @(@{
                name='DFL-PT-WEBUI-main/link';content='../outside';attributes=$attributeCases[$i]})
            $target = Join-Path $TestDrive ('link-target-' + $i)
            Invoke-ProjectSourceInstall -ProjectRoot $target -SourceArchivePath $archive -NoNetwork | Should Be 1
            Test-Path -LiteralPath (Join-Path $target '_internal') | Should Be $false
        }
    }

    It 'bounds archive bytes, expanded bytes, individual entries and file count' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'limits.zip')
        foreach ($case in @('archive','expanded','entry','count')) {
            $arguments = @{ ProjectRoot=(Join-Path $TestDrive ('limit-' + $case)); SourceArchivePath=$archive; NoNetwork=$true }
            switch ($case) {
                'archive' { $arguments.MaxArchiveBytes = 16 }
                'expanded' { $arguments.MaxExpandedBytes = 16 }
                'entry' { $arguments.MaxEntryBytes = 16 }
                'count' { $arguments.MaxEntries = 2 }
            }
            Invoke-ProjectSourceInstall @arguments | Should Be 1
            Test-Path -LiteralPath (Join-Path $arguments.ProjectRoot 'webui') | Should Be $false
        }
    }

    It 'does not attempt a network download in explicit offline mode' {
        Mock Get-ProjectInstallArchive { throw 'unexpected network helper' } -ParameterFilter { -not $Offline }
        Invoke-ProjectSourceInstall -ProjectRoot (Join-Path $TestDrive 'offline') -NoNetwork | Should Be 1
        Assert-MockCalled Get-ProjectInstallArchive -Times 0 -Exactly -ParameterFilter { -not $Offline }
    }

    It 'returns explicit process exit codes and parseable host JSON events' {
        $archiveRoot = 'DFL-PT-WEBUI-' + ('a' * 40)
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'host-exit.zip') -ArchiveRoot $archiveRoot
        $pinPath = Join-Path $TestDrive 'source-pin.json'
        @{schemaVersion=1;product='DFL-PT-WEBUI';applicationVersion='fixture';sourceCommit=('a'*40);archiveRoot=$archiveRoot;archiveSha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $pinPath -Encoding UTF8
        foreach ($case in @('success','failure')) {
            $target = Join-Path $TestDrive ('host-' + $case)
            $start = New-Object Diagnostics.ProcessStartInfo
            $start.FileName = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
            $start.UseShellExecute = $false
            $start.CreateNoWindow = $true
            $start.RedirectStandardOutput = $true
            $start.RedirectStandardError = $true
            $start.StandardOutputEncoding = [Text.Encoding]::UTF8
            $start.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
                [IO.Path]::GetFullPath($script:projectInstallerPath) + '" -ProjectRoot "' + $target + '" -SourcePinPath "' + $pinPath + '" -NoNetwork'
            if ($case -eq 'success') { $start.Arguments += ' -SourceArchivePath "' + $archive + '"' }
            $process = [Diagnostics.Process]::Start($start)
            try {
                $stdoutTask = $process.StandardOutput.ReadToEndAsync()
                $stderrTask = $process.StandardError.ReadToEndAsync()
                $process.WaitForExit(30000) | Should Be $true
                $output = $stdoutTask.GetAwaiter().GetResult()
                $errors = $stderrTask.GetAwaiter().GetResult()
                $errors | Should Be ''
                if ($case -eq 'success') { $process.ExitCode | Should Be 0 }
                else { $process.ExitCode | Should Be 1 }
                $events = @($output -split '\r?\n' | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
                $events.Count | Should BeGreaterThan 0
                foreach ($event in $events) { $event.stage | Should Be 'source' }
                if ($case -eq 'success') { $events[-1].status | Should Be 'complete' }
                else { $events[-1].status | Should Be 'failed' }
            } finally {
                if (-not $process.HasExited) { $process.Kill() }
                $process.Dispose()
            }
        }
    }

    It 'rolls back previously moved source after a later publication failure' {
        $source = Join-Path $TestDrive 'rollback-source'
        $target = Join-Path $TestDrive 'rollback-target'
        New-Item -ItemType Directory -Path $source | Out-Null
        $state = New-OwnedProjectInstallTarget $target
        [IO.File]::WriteAllText((Join-Path $state 'install.log'), 'existing state')
        [IO.File]::WriteAllText((Join-Path $source 'a.txt'), 'a fixture')
        [IO.File]::WriteAllText((Join-Path $source 'b.txt'), 'b fixture')
        $failedMoveSource = Join-Path $source 'b.txt'
        Mock Move-ProjectInstallEntry { throw 'fixture move failure' } -ParameterFilter { $Source -eq $failedMoveSource }
        { Publish-ProjectInstallSource $source $target } | Should Throw
        [IO.File]::ReadAllText((Join-Path $source 'a.txt')) | Should Be 'a fixture'
        [IO.File]::ReadAllText((Join-Path $source 'b.txt')) | Should Be 'b fixture'
        Test-Path -LiteralPath (Join-Path $target 'a.txt') | Should Be $false
        Test-Path -LiteralPath (Join-Path $target 'b.txt') | Should Be $false
        [IO.File]::ReadAllText((Join-Path $state 'install.log')) | Should Be 'existing state'
        Assert-MockCalled Move-ProjectInstallEntry -Times 1 -Exactly -ParameterFilter { $Source -eq $failedMoveSource }
    }

    It 'preflights every top-level collision before moving any source file' {
        $source = Join-Path $TestDrive 'collision-source'
        $target = Join-Path $TestDrive 'collision-target'
        New-Item -ItemType Directory -Path $source | Out-Null
        $state = New-OwnedProjectInstallTarget $target
        [IO.File]::WriteAllText((Join-Path $source 'a.txt'), 'new file')
        New-Item -ItemType Directory -Path (Join-Path $source '.launcher-install') | Out-Null
        { Publish-ProjectInstallSource $source $target } | Should Throw
        Test-Path -LiteralPath (Join-Path $source 'a.txt') | Should Be $true
        Test-Path -LiteralPath (Join-Path $target 'a.txt') | Should Be $false
        [IO.File]::ReadAllText((Join-Path $state 'owner.txt')) | Should Be 'DFL-PT-WEBUI install workspace v1'
    }

    It 'uses atomic destination collision rejection without nesting or overwriting' {
        $source = Join-Path $TestDrive 'atomic-source'
        $target = Join-Path $TestDrive 'atomic-target'
        New-Item -ItemType Directory -Path $source, $target | Out-Null
        [IO.File]::WriteAllText((Join-Path $source 'new.txt'), 'new fixture')
        [IO.File]::WriteAllText((Join-Path $target 'user.txt'), 'user original')
        { Move-ProjectInstallEntry $source $target } | Should Throw
        Test-Path -LiteralPath (Join-Path $source 'new.txt') | Should Be $true
        Test-Path -LiteralPath (Join-Path $target 'atomic-source') | Should Be $false
        [IO.File]::ReadAllText((Join-Path $target 'user.txt')) | Should Be 'user original'
    }

    It 'rejects a linked install target without touching the junction destination' {
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'junction.zip')
        $outside = Join-Path $TestDrive 'junction-outside'
        $link = Join-Path $TestDrive 'junction-target'
        New-Item -ItemType Directory -Path $outside | Out-Null
        [IO.File]::WriteAllText((Join-Path $outside 'user.txt'), 'keep outside')
        New-Item -ItemType Junction -Path $link -Target $outside | Out-Null
        try {
            Invoke-ProjectSourceInstall -ProjectRoot $link -SourceArchivePath $archive -NoNetwork | Should Be 1
            [IO.File]::ReadAllText((Join-Path $outside 'user.txt')) | Should Be 'keep outside'
            Test-Path -LiteralPath (Join-Path $outside '.launcher-install') | Should Be $false
        } finally { [IO.Directory]::Delete($link) }
    }

    It 'refuses to recurse through a linked cache during cleanup' {
        $target = Join-Path $TestDrive 'linked-cache-target'
        $state = New-OwnedProjectInstallTarget $target
        $session = Join-Path $state ('source-' + [Guid]::NewGuid().ToString('N'))
        $outside = Join-Path $TestDrive 'linked-cache-outside'
        New-Item -ItemType Directory -Path $session, $outside | Out-Null
        [IO.File]::WriteAllText((Join-Path $outside 'user.txt'), 'keep outside')
        $link = Join-Path $session 'link'
        New-Item -ItemType Junction -Path $link -Target $outside | Out-Null
        try {
            { Remove-ProjectInstallSession $state $session } | Should Throw
            [IO.File]::ReadAllText((Join-Path $outside 'user.txt')) | Should Be 'keep outside'
            Test-Path -LiteralPath $session | Should Be $true
        } finally { [IO.Directory]::Delete($link) }
    }

    It 'refuses cleanup outside its own UUID session and preserves other logs' {
        $target = Join-Path $TestDrive 'cleanup-target'
        $state = New-OwnedProjectInstallTarget $target
        $other = Join-Path $state 'logs'
        New-Item -ItemType Directory -Path $other | Out-Null
        [IO.File]::WriteAllText((Join-Path $other 'keep.txt'), 'user log')
        { Remove-ProjectInstallSession $state $other } | Should Throw
        [IO.File]::ReadAllText((Join-Path $other 'keep.txt')) | Should Be 'user log'
        { Remove-ProjectInstallSession $state (Join-Path $TestDrive ('source-' + [Guid]::NewGuid().ToString('N'))) } | Should Throw
    }
}
