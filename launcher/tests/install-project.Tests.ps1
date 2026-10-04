$script:projectInstallerPath = Join-Path $PSScriptRoot '../install-project.ps1'
. $script:projectInstallerPath
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function New-ProjectSourceFixture([string]$Archive, [string]$Missing='', [object[]]$Extra=@(), [string]$Product='DFL-PT-WEBUI') {
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
            $entry = $zip.CreateEntry('DFL-PT-WEBUI-main/' + $name)
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

Describe 'bounded PT first-install source acquisition' {
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
        $archive = New-ProjectSourceFixture (Join-Path $TestDrive 'host-exit.zip')
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
                [IO.Path]::GetFullPath($script:projectInstallerPath) + '" -ProjectRoot "' + $target + '" -NoNetwork'
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
