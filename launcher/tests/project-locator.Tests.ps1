$sourcePath = Join-Path $PSScriptRoot "..\host\ProjectLocator.cs"

Describe "first-install destination selection" {
    BeforeAll {
        # Keep fixture types separate from other launcher test assemblies.
        $source = (Get-Content -LiteralPath $sourcePath -Raw -Encoding UTF8).Replace(
            "namespace DflPtWebUi.Launcher", "namespace InstallPathTests")
        $source = $source.Replace("internal static class ProjectLocator", "public static class ProjectLocator")
        $source += 'namespace InstallPathTests { public class LauncherSettings { public string ProjectRoot { get; set; } } }'
        Add-Type -TypeDefinition $source -ReferencedAssemblies System.Web.Extensions
    }

    function New-TestProject([string]$Path) {
        foreach ($part in @('.git', 'webui/scripts', '_internal/DeepFaceLab/me_backend', 'release')) {
            New-Item -ItemType Directory -Path (Join-Path $Path $part) -Force | Out-Null
        }
        [IO.File]::WriteAllText((Join-Path $Path '_internal/DeepFaceLab/me.py'), '# ME entry')
        [IO.File]::WriteAllText((Join-Path $Path '_internal/DeepFaceLab/me_backend/engine.py'), '# ME engine')
        [IO.File]::WriteAllText((Join-Path $Path '_internal/DeepFaceLab/me_backend/network.py'), '# ME network')
        [IO.File]::WriteAllText((Join-Path $Path 'webui/package.json'), '{}')
        [IO.File]::WriteAllText((Join-Path $Path 'webui/scripts/local-manager.mjs'), '// manager')
        [IO.File]::WriteAllText((Join-Path $Path 'release/version.json'), '{"schemaVersion":1,"product":"DFL-PT-WEBUI","version":"0.1.1-preview"}')
    }

    It "resumes an owned install workspace without adding another folder" {
        $project = Join-Path $TestDrive 'resume'
        $state = [InstallPathTests.ProjectLocator]::PrepareInstallWorkspace($project)
        New-Item -ItemType Directory -Path (Join-Path $state 'runtime'), (Join-Path $state 'logs') -Force | Out-Null
        [InstallPathTests.ProjectLocator]::SelectInstallPath($project) | Should Be $project
        [InstallPathTests.ProjectLocator]::AssertInstallTarget($project)
        [IO.File]::WriteAllText((Join-Path $project 'personal.txt'), 'keep')
        { [InstallPathTests.ProjectLocator]::AssertInstallTarget($project) } | Should Throw
    }

    It "publishes a clone alongside runtime and logs, preserving their contents" {
        $project = Join-Path $TestDrive 'publish'
        $state = [InstallPathTests.ProjectLocator]::PrepareInstallWorkspace($project)
        $staging = Join-Path $state 'cloning-fixture'
        New-TestProject $staging
        [IO.File]::WriteAllText((Join-Path $state 'install.log'), 'keep log')
        [IO.File]::WriteAllText((Join-Path $staging 'README.md'), 'source')
        [InstallPathTests.ProjectLocator]::PublishClone($staging, $project)
        [InstallPathTests.ProjectLocator]::IsProject($project) | Should Be $true
        [IO.File]::ReadAllText((Join-Path $state 'install.log')) | Should Be 'keep log'
        [IO.File]::ReadAllText((Join-Path $project 'README.md')) | Should Be 'source'
    }

    It "rejects a cloned tree that collides with installer state before moving files" {
        $project = Join-Path $TestDrive 'publish-conflict'
        $state = [InstallPathTests.ProjectLocator]::PrepareInstallWorkspace($project)
        $staging = Join-Path $state 'cloning-fixture'
        New-TestProject $staging
        New-Item -ItemType Directory -Path (Join-Path $staging '.launcher-install') | Out-Null
        { [InstallPathTests.ProjectLocator]::PublishClone($staging, $project) } | Should Throw
        Test-Path (Join-Path $project '.git') | Should Be $false
        Test-Path (Join-Path $staging '.git') | Should Be $true
    }

    It "uses an empty selected folder directly and does not create anything" {
        $selected = Join-Path $TestDrive 'empty folder'
        New-Item -ItemType Directory -Path $selected | Out-Null
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be $selected
        @(Get-ChildItem -LiteralPath $selected -Force).Count | Should Be 0
    }

    It "uses a new explicit folder directly" {
        $selected = Join-Path $TestDrive 'new-folder'
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be $selected
        Test-Path -LiteralPath $selected | Should Be $false
    }

    It "puts drive-root installs in DFL-PT-WEBUI without writing to the drive" {
        $drive = [IO.Path]::GetPathRoot($TestDrive)
        $destination = Join-Path $drive 'DFL-PT-WEBUI'
        # Use a read-only call on the host drive; unrelated content must be refused.
        if ((Test-Path -LiteralPath $destination) -and
            -not [InstallPathTests.ProjectLocator]::IsEmptyDirectory($destination) -and
            -not [InstallPathTests.ProjectLocator]::IsProject($destination)) {
            { [InstallPathTests.ProjectLocator]::SelectInstallPath($drive) } | Should Throw
        } else {
            [InstallPathTests.ProjectLocator]::SelectInstallPath($drive) | Should Be $destination
        }
    }

    It "counts a single hidden file as non-empty and preserves it" {
        $selected = Join-Path $TestDrive 'occupied'
        New-Item -ItemType Directory -Path $selected | Out-Null
        $sentinel = Join-Path $selected 'sentinel.txt'
        [IO.File]::WriteAllText($sentinel, 'keep')
        [IO.File]::SetAttributes($sentinel, [IO.FileAttributes]::Hidden)
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be (Join-Path $selected 'DFL-PT-WEBUI')
        [IO.File]::ReadAllText($sentinel) | Should Be 'keep'
        Test-Path -LiteralPath (Join-Path $selected 'DFL-PT-WEBUI') | Should Be $false
    }

    It "counts subdirectories and reuses an existing empty target without nesting" {
        $selected = Join-Path $TestDrive 'parent'
        $destination = Join-Path $selected 'DFL-PT-WEBUI'
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be $destination
        [InstallPathTests.ProjectLocator]::SelectInstallPath($destination) | Should Be $destination
    }

    It "reuses projects directly and under the selected parent" {
        $selected = Join-Path $TestDrive 'existing'
        $destination = Join-Path $selected 'DFL-PT-WEBUI'
        New-TestProject $destination
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be $destination
        [InstallPathTests.ProjectLocator]::SelectInstallPath($destination) | Should Be $destination
        $settings = New-Object InstallPathTests.LauncherSettings
        $settings.ProjectRoot = $destination
        [InstallPathTests.ProjectLocator]::Resolve($settings) | Should Be $destination
    }

    It "refuses an occupied target, files, and changes made after selection" {
        $selected = Join-Path $TestDrive 'conflict'
        $destination = Join-Path $selected 'DFL-PT-WEBUI'
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) | Should Be $destination
        $file = Join-Path $destination 'keep.txt'
        [IO.File]::WriteAllText($file, 'keep')
        { [InstallPathTests.ProjectLocator]::SelectInstallPath($selected) } | Should Throw
        { [InstallPathTests.ProjectLocator]::SelectInstallPath($destination) } | Should Throw
        { [InstallPathTests.ProjectLocator]::SelectInstallPath($file) } | Should Throw
        { [InstallPathTests.ProjectLocator]::AssertInstallTarget($destination) } | Should Throw
        [IO.File]::ReadAllText($file) | Should Be 'keep'
    }

    It "bounds writable paths to the selected project" {
        $project = Join-Path $TestDrive 'write-paths'
        New-TestProject $project
        [InstallPathTests.ProjectLocator]::AssertWritableChildPath($project, 'webui/dist') | Should Be (Join-Path $project 'webui/dist')
        { [InstallPathTests.ProjectLocator]::AssertWritableChildPath($project, '../outside') } | Should Throw
        { [InstallPathTests.ProjectLocator]::AssertWritableChildPath($project, $TestDrive) } | Should Throw
    }

    It "rejects linked output directories while allowing internal pnpm links" {
        $project = Join-Path $TestDrive 'linked-webui'
        New-TestProject $project
        $outside = Join-Path $TestDrive 'outside-output'
        New-Item -ItemType Directory -Path $outside | Out-Null
        [IO.File]::WriteAllText((Join-Path $outside 'keep.txt'), 'keep')
        foreach ($relative in @('webui/node_modules', 'webui/dist')) {
            $link = Join-Path $project $relative
            New-Item -ItemType Junction -Path $link -Target $outside | Out-Null
            try {
                { [InstallPathTests.ProjectLocator]::AssertWritableChildPath($project, $relative) } | Should Throw
            } finally { [IO.Directory]::Delete($link) }
        }
        $modules = Join-Path $project 'webui/node_modules'
        New-Item -ItemType Directory -Path $modules | Out-Null
        $internalLink = Join-Path $modules 'fixture-package'
        New-Item -ItemType Junction -Path $internalLink -Target $outside | Out-Null
        try {
            [InstallPathTests.ProjectLocator]::AssertWritableChildPath($project, 'webui/node_modules') | Should Be $modules
        } finally { [IO.Directory]::Delete($internalLink) }
        [IO.File]::ReadAllText((Join-Path $outside 'keep.txt')) | Should Be 'keep'
    }

    It "rejects linked projects and ancestor directories before writes" {
        $project = Join-Path $TestDrive 'project-link-target'
        New-TestProject $project
        $link = Join-Path $TestDrive 'project-link'
        New-Item -ItemType Junction -Path $link -Target $project | Out-Null
        try {
            { [InstallPathTests.ProjectLocator]::AssertWritableChildPath($link, 'webui/dist') } | Should Throw
        } finally { [IO.Directory]::Delete($link) }
        $ancestor = Join-Path $TestDrive 'ancestor-link'
        New-Item -ItemType Junction -Path $ancestor -Target $TestDrive | Out-Null
        try {
            $nested = Join-Path $ancestor 'project-link-target'
            { [InstallPathTests.ProjectLocator]::AssertWritableChildPath($nested, 'webui/dist') } | Should Throw
        } finally { [IO.Directory]::Delete($ancestor) }
    }
}
