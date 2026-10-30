Describe 'deterministic download route ordering' {
    BeforeAll {
        $script:previousMirror=$env:DFL_MIRROR
        . (Join-Path $PSScriptRoot '../download-sources.ps1')
    }
    BeforeEach {$env:DFL_MIRROR='auto';$script:DflDownloadFailures=@{}}
    AfterAll {$env:DFL_MIRROR=$script:previousMirror}
    It 'keeps official first when both hosts have equal priority' {
        $url='https://codeload.github.com/LeoSasion/DFL-PT-WEBUI/zip/fixed'
        foreach($index in 1..10){@(Get-DflDownloadUrls $url)[0]|Should Be $url}
    }
    It 'temporarily moves a failed host after the healthy mirror' {
        $script:DflDownloadFailures['nodejs.org']=[DateTime]::UtcNow.AddSeconds(60)
        @(Get-DflDownloadUrls 'https://nodejs.org/dist/fixed.zip')[0]|Should Match '^https://registry.npmmirror.com/'
        $script:DflDownloadFailures['nodejs.org']=[DateTime]::UtcNow.AddSeconds(-1)
        @(Get-DflDownloadUrls 'https://nodejs.org/dist/fixed.zip')[0]|Should Be 'https://nodejs.org/dist/fixed.zip'
    }
    It 'retains explicit domestic and official preferences' {
        $env:DFL_MIRROR='china'
        @(Get-DflDownloadUrls 'https://nodejs.org/dist/fixed.zip')[0]|Should Match '^https://registry.npmmirror.com/'
        $env:DFL_MIRROR='official'
        @(Get-DflDownloadUrls 'https://nodejs.org/dist/fixed.zip').Count|Should Be 1
    }
    It 'does not forward signed URLs to public relays' {
        $url='https://github.com/example/asset?token=private'
        $urls=@(Get-DflDownloadUrls $url)
        $urls.Count|Should Be 1
        $urls[0]|Should Be $url
    }
}
