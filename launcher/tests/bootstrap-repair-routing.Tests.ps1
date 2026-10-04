$script:repairBootstrapSource = Join-Path $PSScriptRoot '../bootstrap.ps1'
$script:repairInstallerSource = Join-Path $PSScriptRoot '../install-source.ps1'

function New-RepairBootstrapFixture([string]$Base, [bool]$Ready=$false, [bool]$SeparateHost=$false, [object[]]$RuntimeProcesses=@()) {
    $project = Join-Path $Base 'project'
    $sourceLauncher = Join-Path $project 'launcher'
    $hostLauncher = $sourceLauncher
    if ($SeparateHost) { $hostLauncher = Join-Path $Base 'embedded' }
    foreach ($path in @($sourceLauncher, $hostLauncher, (Join-Path $project 'webui'),
        (Join-Path $project '_internal/DeepFaceLab'))) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    [IO.File]::WriteAllText((Join-Path $project '_internal/DeepFaceLab/me.py'), '# isolated source fixture')
    Copy-Item -LiteralPath $script:repairBootstrapSource -Destination (Join-Path $hostLauncher 'bootstrap.ps1')
    # Inject the read-only process provider into this private script copy. The
    # actual bootstrap idle check runs unchanged, with no host process access.
    $fixtureBootstrap = Join-Path $hostLauncher 'bootstrap.ps1'
    $fixtureText = [IO.File]::ReadAllText($fixtureBootstrap)
    $tokens = $null
    $errors = $null
    $fixtureAst = [Management.Automation.Language.Parser]::ParseInput($fixtureText, [ref]$tokens, [ref]$errors)
    $processJson = ConvertTo-Json -InputObject @($RuntimeProcesses) -Depth 4 -Compress
    $provider = "`nfunction Get-CimInstance { param([string]`$ClassName,[string]`$Filter) '" +
        $processJson.Replace("'", "''") + "' | ConvertFrom-Json }`n"
    $fixtureText = $fixtureText.Insert($fixtureAst.ParamBlock.Extent.EndOffset, $provider)
    [IO.File]::WriteAllText($fixtureBootstrap, $fixtureText, (New-Object Text.UTF8Encoding($true)))
    $paths = @('_internal/python_base/python.exe', '.venv/Scripts/python.exe',
        '_internal/node/bin/node.exe', '_internal/ffmpeg/ffmpeg.exe')
    if ($Ready) {
        foreach ($relative in $paths) {
            $file = Join-Path $project $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
            [IO.File]::WriteAllText($file, 'fixture only; never executed')
        }
    }
    $components = @(
        @{ id='python'; displayName='Python'; required=$true; install=@{relativePath='.venv'};
            validation=@{files=@(@{path='Scripts/python.exe';kind='file';minBytes=1});command=$null} },
        @{ id='node'; displayName='Node'; required=$true; install=@{relativePath='_internal/node/bin'};
            validation=@{files=@(@{path='node.exe';kind='file';minBytes=1});command=$null} },
        @{ id='ffmpeg'; displayName='FFmpeg'; required=$true; install=@{relativePath='_internal/ffmpeg'};
            validation=@{files=@(@{path='ffmpeg.exe';kind='file';minBytes=1});command=$null} }
    )
    [IO.File]::WriteAllText((Join-Path $hostLauncher 'runtime-manifest.json'),
        (@{schemaVersion=3;components=$components} | ConvertTo-Json -Depth 8))
    $setup = @'
param([string]$ProjectRoot, [switch]$NoNetwork)
$entry = @{operation='rebase';noNetwork=[bool]$NoNetwork} | ConvertTo-Json -Compress
[IO.File]::AppendAllText((Join-Path $ProjectRoot 'calls.jsonl'), $entry + [Environment]::NewLine)
$global:LASTEXITCODE = 0
'@
    [IO.File]::WriteAllText((Join-Path $hostLauncher 'setup-runtime.ps1'), $setup)
    $installer = @'
param([string]$ProjectRoot, [switch]$SkipWebuiPreparation, [switch]$NoNetwork, [switch]$SkipVisionAssets)
$entry = @{operation='install';skipWebuiPreparation=[bool]$SkipWebuiPreparation;
    noNetwork=[bool]$NoNetwork;skipVisionAssets=[bool]$SkipVisionAssets;installer=$PSCommandPath} | ConvertTo-Json -Compress
[IO.File]::AppendAllText((Join-Path $ProjectRoot 'calls.jsonl'), $entry + [Environment]::NewLine)
foreach ($relative in @('_internal/python_base/python.exe', '.venv/Scripts/python.exe',
    '_internal/node/bin/node.exe', '_internal/ffmpeg/ffmpeg.exe')) {
    $file = Join-Path $ProjectRoot $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
    [IO.File]::WriteAllText($file, 'fixture only; never executed')
}
if (-not $SkipVisionAssets) { [IO.File]::WriteAllText((Join-Path $ProjectRoot 'weights-checked.txt'), 'fixture validation') }
$global:LASTEXITCODE = 0
'@
    [IO.File]::WriteAllText((Join-Path $hostLauncher 'install-source.ps1'), $installer)
    if ($SeparateHost) {
        [IO.File]::WriteAllText((Join-Path $sourceLauncher 'install-source.ps1'),
            "param([string]`$ProjectRoot)`nthrow 'old source installer must not be selected when the embedded version is present'`n")
    }
    return @{ root=$project; bootstrap=(Join-Path $hostLauncher 'bootstrap.ps1'); host=$hostLauncher }
}

function Invoke-RepairBootstrapFixture([hashtable]$Fixture, [switch]$Repair, [switch]$NoNetwork, [switch]$DryRun) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = [Text.Encoding]::UTF8
    $start.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $Fixture.bootstrap +
        '" -ProjectRoot "' + $Fixture.root + '"'
    if ($Repair) { $start.Arguments += ' -Repair' }
    if ($NoNetwork) { $start.Arguments += ' -NoNetwork' }
    if ($DryRun) { $start.Arguments += ' -DryRun' }
    $process = [Diagnostics.Process]::Start($start)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(30000)) { throw 'Fixture bootstrap exceeded its bound.' }
        $events = @($stdout.GetAwaiter().GetResult() -split '\r?\n' | Where-Object { $_ } |
            ForEach-Object { $_ | ConvertFrom-Json })
        $calls = @()
        $log = Join-Path $Fixture.root 'calls.jsonl'
        if (Test-Path -LiteralPath $log) {
            $calls = @(Get-Content -LiteralPath $log -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
        }
        return @{exitCode=$process.ExitCode;events=$events;calls=$calls;errors=$stderr.GetAwaiter().GetResult()}
    } finally {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
}

function Invoke-IsolatedWebuiPreparation([string]$Root, [bool]$SkipPreparation, [bool]$SkipBuild, [bool]$DamagedModules=$false) {
    # Execute the actual guarded installer block with a local function standing
    # in for Node, so no dependencies, registry, training or runtime are touched.
    $nodeCalls = New-Object 'System.Collections.Generic.List[object]'
    $environmentCalls = New-Object 'System.Collections.Generic.List[string]'
    function Invoke-WebuiNodeFixture {
        $nodeCalls.Add(@($args))
        $global:LASTEXITCODE = 0
    }
    function Set-InstallEnvironment([string]$Name, [string]$Value) { $environmentCalls.Add($Name) }
    function Assert-Exit([string]$Operation) { if ($LASTEXITCODE -ne 0) { throw $Operation } }
    $SkipWebuiPreparation = $SkipPreparation
    $SkipWebuiBuild = $SkipBuild
    $NoNetwork = $false
    $root = $Root
    $webui = Join-Path $Root 'webui'
    $nodeBin = Join-Path $Root '_internal/node/bin'
    $node = 'Invoke-WebuiNodeFixture'
    $basePython = Join-Path $Root '_internal/python_base/python.exe'
    $cache = Join-Path $Root '.launcher-install/source'
    $modules = Join-Path $webui 'node_modules'
    $dist = Join-Path $webui 'dist'
    $createdModules = $false
    $createdDist = $false
    New-Item -ItemType Directory -Path $webui -Force | Out-Null
    if ($DamagedModules) { [IO.File]::WriteAllText($modules, 'damaged module directory fixture') }
    & ([scriptblock]::Create($script:webuiPreparationAst.Extent.Text))
    return @{nodeCalls=@($nodeCalls.ToArray());environmentCalls=@($environmentCalls.ToArray());modules=$modules;dist=$dist}
}

Describe 'launcher runtime repair routing' {
    BeforeAll {
        $tokens = $null
        $errors = $null
        $script:installerAst = [Management.Automation.Language.Parser]::ParseFile(
            [IO.Path]::GetFullPath($script:repairInstallerSource), [ref]$tokens, [ref]$errors)
        if ($errors.Count -gt 0) { throw ($errors.Message -join '; ') }
        $script:webuiPreparationAst = $script:installerAst.Find({param($node)
            $node -is [Management.Automation.Language.IfStatementAst] -and
            $node.Extent.Text.StartsWith('if (-not $SkipWebuiPreparation) {') -and
            $node.Extent.Text.Contains('WebUI dependency / native module validation')}, $true)
        if (-not $script:webuiPreparationAst) { throw 'Complete WebUI preparation guard is missing.' }
        $script:visionPreparationAst = $script:installerAst.Find({param($node)
            $node -is [Management.Automation.Language.IfStatementAst] -and
            $node.Extent.Text.StartsWith('if (-not $SkipVisionAssets) {')}, $true)
    }

    It 'uses the complete installer for a normal first installation' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'first')
        $result = Invoke-RepairBootstrapFixture $fixture
        $result.exitCode | Should Be 0
        $result.errors | Should Be ''
        $result.calls.Count | Should Be 1
        $result.calls[0].operation | Should Be 'install'
        $result.calls[0].skipWebuiPreparation | Should Be $false
        $result.calls[0].skipVisionAssets | Should Be $false
        Test-Path -LiteralPath (Join-Path $fixture.root 'weights-checked.txt') | Should Be $true
    }

    It 'repairs a ready runtime and verifies weights before host WebUI repair' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'repair') -Ready $true
        [IO.File]::WriteAllText((Join-Path $fixture.root 'webui/node_modules'), 'broken fixture')
        $result = Invoke-RepairBootstrapFixture $fixture -Repair
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 2
        $result.calls[0].operation | Should Be 'rebase'
        $result.calls[0].noNetwork | Should Be $true
        $result.calls[1].operation | Should Be 'install'
        $result.calls[1].skipWebuiPreparation | Should Be $true
        $result.calls[1].skipVisionAssets | Should Be $false
        Test-Path -LiteralPath (Join-Path $fixture.root 'weights-checked.txt') | Should Be $true
        [IO.File]::ReadAllText((Join-Path $fixture.root 'webui/node_modules')) | Should Be 'broken fixture'
    }

    It 'uses the embedded matching installer when an old project installer exists' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'embedded') -Ready $true -SeparateHost $true
        $result = Invoke-RepairBootstrapFixture $fixture -Repair
        $result.exitCode | Should Be 0
        $result.calls[1].installer | Should Be (Join-Path $fixture.host 'install-source.ps1')
        $result.calls[1].skipWebuiPreparation | Should Be $true
    }

    It 'does not reinstall a ready runtime on a regular start' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'ready') -Ready $true
        $result = Invoke-RepairBootstrapFixture $fixture
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 1
        $result.calls[0].operation | Should Be 'rebase'
        Test-Path -LiteralPath (Join-Path $fixture.root 'weights-checked.txt') | Should Be $false
    }

    It 'performs offline checking without invoking a downloading installer even for Repair' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'offline') -Ready $true
        $result = Invoke-RepairBootstrapFixture $fixture -Repair -NoNetwork
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 1
        $result.calls[0].operation | Should Be 'rebase'
        $result.calls[0].noNetwork | Should Be $true
        Test-Path -LiteralPath (Join-Path $fixture.root 'weights-checked.txt') | Should Be $false
    }

    It 'rebases the portable venv before installing a missing unrelated component' {
        $fixture = New-RepairBootstrapFixture (Join-Path $TestDrive 'relocate') -Ready $true
        Remove-Item -LiteralPath (Join-Path $fixture.root '_internal/ffmpeg/ffmpeg.exe')
        $result = Invoke-RepairBootstrapFixture $fixture
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 2
        $result.calls[0].operation | Should Be 'rebase'
        $result.calls[1].operation | Should Be 'install'
        $result.calls[1].skipWebuiPreparation | Should Be $false
    }

    It 'skips all WebUI install, validation, environment preparation and build when delegated' {
        $result = Invoke-IsolatedWebuiPreparation (Join-Path $TestDrive 'delegate') $true $false $true
        $result.nodeCalls.Count | Should Be 0
        $result.environmentCalls.Count | Should Be 0
        [IO.File]::ReadAllText($result.modules) | Should Be 'damaged module directory fixture'
        Test-Path -LiteralPath $result.dist | Should Be $false
    }

    It 'keeps default dependency installation and validation while SkipWebuiBuild only omits the build' {
        $result = Invoke-IsolatedWebuiPreparation (Join-Path $TestDrive 'skip-build') $false $true
        $result.nodeCalls.Count | Should Be 2
        ($result.nodeCalls[0] -join ' ') | Should Match 'pnpm install --frozen-lockfile'
        ($result.nodeCalls[1] -join ' ') | Should Match 'require.resolve'
        $result.environmentCalls.Count | Should BeGreaterThan 0
        Test-Path -LiteralPath $result.dist | Should Be $false
    }

    It 'prepares dependencies and the initial production build by default' {
        $result = Invoke-IsolatedWebuiPreparation (Join-Path $TestDrive 'full-build') $false $false
        $result.nodeCalls.Count | Should Be 5
        ($result.nodeCalls[2] -join ' ') | Should Match 'build --configLoader runner'
        ($result.nodeCalls[4] -join ' ') | Should Match 'dist-provenance.mjs write'
        Test-Path -LiteralPath $result.dist | Should Be $true
    }

    It 'keeps helper-weight preparation outside the WebUI guard' {
        $script:visionPreparationAst | Should Not BeNullOrEmpty
        $script:visionPreparationAst.Extent.EndOffset | Should BeLessThan $script:webuiPreparationAst.Extent.StartOffset
        $root = Join-Path $TestDrive 'vision'
        New-Item -ItemType Directory -Path (Join-Path $root 'tools') -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $root 'tools/prepare-vision-runtime.ps1'), @'
param([string]$ProjectRoot, [switch]$NoNetwork, [switch]$SkipFFmpeg, [switch]$PreserveExisting)
[IO.File]::WriteAllText((Join-Path $ProjectRoot 'vision-call.json'),
    (@{noNetwork=[bool]$NoNetwork;skipFFmpeg=[bool]$SkipFFmpeg;preserveExisting=[bool]$PreserveExisting} | ConvertTo-Json))
'@)
        $SkipVisionAssets = $false
        $SkipWebuiPreparation = $true
        $NoNetwork = $false
        & ([scriptblock]::Create($script:visionPreparationAst.Extent.Text))
        $call = Get-Content -LiteralPath (Join-Path $root 'vision-call.json') -Raw | ConvertFrom-Json
        $call.skipFFmpeg | Should Be $true
        $call.preserveExisting | Should Be $true
        $call.noNetwork | Should Be $false
    }

    It 'retains top-level WebUI junction safety checks when dependency preparation is delegated' {
        $assertion = $script:installerAst.Find({param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Assert-ProjectPath'}, $true)
        $pathAssignment = $script:installerAst.Find({param($node)
            $node -is [Management.Automation.Language.AssignmentStatementAst] -and
            $node.Left.Extent.Text -eq '$runtimePaths'}, $true)
        $pathLoop = $script:installerAst.Find({param($node)
            $node -is [Management.Automation.Language.ForEachStatementAst] -and
            $node.Condition.Extent.Text -eq '$runtimePaths'}, $true)
        $assertion | Should Not BeNullOrEmpty
        $pathAssignment | Should Not BeNullOrEmpty
        $pathLoop | Should Not BeNullOrEmpty
        . ([scriptblock]::Create($assertion.Extent.Text))
        $root = Join-Path $TestDrive 'safe-delegate'
        $rootPrefix = $root + [IO.Path]::DirectorySeparatorChar
        $outside = Join-Path $TestDrive 'safe-delegate-outside'
        $modules = Join-Path $root 'webui/node_modules'
        $dist = Join-Path $root 'webui/dist'
        New-Item -ItemType Directory -Path (Split-Path -Parent $modules), $outside -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $outside 'keep.txt'), 'preserve outside')
        New-Item -ItemType Junction -Path $modules -Target $outside | Out-Null
        $cache = $session = $base = $venv = $nodeRoot = $ffmpegRoot = Join-Path $root 'other-runtime'
        $SkipWebuiPreparation = $true
        try {
            . ([scriptblock]::Create($pathAssignment.Extent.Text))
            { & ([scriptblock]::Create($pathLoop.Extent.Text)) } | Should Throw
            [IO.File]::ReadAllText((Join-Path $outside 'keep.txt')) | Should Be 'preserve outside'
        } finally { [IO.Directory]::Delete($modules) }
    }

    It 'blocks a surviving project process before either rebase or installation' {
        $base = Join-Path $TestDrive 'busy'
        $project = Join-Path $base 'project'
        $process = @{ExecutablePath=(Join-Path $project '.venv/Scripts/python.exe');CommandLine='isolated fixture process'}
        $fixture = New-RepairBootstrapFixture $base -Ready $true -RuntimeProcesses @($process)
        $result = Invoke-RepairBootstrapFixture $fixture -Repair
        $result.exitCode | Should Be 1
        $result.calls.Count | Should Be 0
        $result.events[-1].message | Should Match 'runtime processes are active'
        Test-Path -LiteralPath (Join-Path $fixture.root 'weights-checked.txt') | Should Be $false
    }

    It 'does not block processes in another repository with a similar directory prefix' {
        $base = Join-Path $TestDrive 'foreign-process'
        $foreign = (Join-Path $base 'project') + '-other'
        $process = @{ExecutablePath=(Join-Path $foreign '.venv/Scripts/python.exe');
            CommandLine=('"' + (Join-Path $foreign '.venv/Scripts/python.exe') + '" "' + (Join-Path $foreign 'train.py') + '"')}
        $fixture = New-RepairBootstrapFixture $base -Ready $true -RuntimeProcesses @($process)
        $result = Invoke-RepairBootstrapFixture $fixture
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 1
        $result.calls[0].operation | Should Be 'rebase'
    }

    It 'keeps DryRun read-only even if a project runtime process is active' {
        $base = Join-Path $TestDrive 'busy-dry-run'
        $project = Join-Path $base 'project'
        $process = @{ExecutablePath=(Join-Path $project '.venv/Scripts/python.exe');CommandLine='isolated fixture process'}
        $fixture = New-RepairBootstrapFixture $base -Ready $true -RuntimeProcesses @($process)
        $result = Invoke-RepairBootstrapFixture $fixture -Repair -DryRun
        $result.exitCode | Should Be 0
        $result.calls.Count | Should Be 0
    }
}
