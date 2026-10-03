Describe "renamed repository identity" {
    BeforeAll {
        $source = (Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\host\LauncherConstants.cs') -Raw -Encoding UTF8)
        $source = $source.Replace('namespace DflPtWebUi.Launcher', 'namespace RepositoryIdentityTests')
        $source = $source.Replace('internal static class LauncherConstants', 'public static class LauncherConstants')
        Add-Type -TypeDefinition $source
    }

    It "rejects all source remotes while the new local repository is unpublished" {
        [RepositoryIdentityTests.LauncherConstants]::IsOfficialGitRemote('https://github.com/LeoSasion/DFL-PT-WEBUI.git') | Should Be $false
        [RepositoryIdentityTests.LauncherConstants]::IsOfficialGitRemote('https://github.com/LeoSasion/DFL-PT-WEBUI.git') | Should Be $false
        [RepositoryIdentityTests.LauncherConstants]::IsOfficialGitRemote('https://github.com/LeoSasion/DFL-PT-WEBUI/') | Should Be $false
    }

    It "rejects different owners, lookalike hosts, credentials, and query suffixes" {
        foreach ($remote in @(
            'https://github.com/other/DFL-PT-WEBUI.git',
            'https://github.com.evil.example/LeoSasion/DFL-PT-WEBUI.git',
            'https://user@github.com/LeoSasion/DFL-PT-WEBUI.git',
            'https://github.com/LeoSasion/DFL-PT-WEBUI.git?other',
            ''
        )) {
            [RepositoryIdentityTests.LauncherConstants]::IsOfficialGitRemote($remote) | Should Be $false
        }
    }
}
