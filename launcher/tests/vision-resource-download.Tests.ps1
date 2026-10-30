$script:visionSource = Join-Path $PSScriptRoot '../../tools/prepare-vision-runtime.ps1'

function Invoke-VisionDownloadFixture([string]$Cache, [string]$Failure, [switch]$Repeat) {
    . (Join-Path $PSScriptRoot '../download-sources.ps1')
    $tokens=$null; $errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile([IO.Path]::GetFullPath($script:visionSource),[ref]$tokens,[ref]$errors)
    foreach($name in @('Test-Hash','Get-Verified')) {
        $function=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true)
        . ([scriptblock]::Create($function.Extent.Text))
    }
    $cacheRoot=$Cache; $NoNetwork=$false; $calls=New-Object 'System.Collections.Generic.List[string]'
    function Invoke-WebRequest {
        param([switch]$UseBasicParsing,[string]$Uri,[string]$OutFile,[int]$TimeoutSec)
        $calls.Add($Uri)
        [IO.File]::WriteAllText($OutFile,$(if($Failure -eq 'checksum'){'bad'}else{'valid'}))
        if($Failure -eq 'network'){throw 'synthetic interrupted download'}
    }
    New-Item -ItemType Directory -Path $Cache -Force | Out-Null
    $hash='ec654fac9599f62e79e2706abef23dfb7c07c08185aa86db4d8695f0b718d1b3'
    $failed=$false
    try {
        $value=Get-Verified 'helper.bin' 'https://example.org/helper' $hash
        if($Repeat){$value=Get-Verified 'helper.bin' 'https://example.org/helper' $hash}
    } catch {$failed=$true}
    return @{failed=$failed;calls=$calls.Count;partial=@(Get-ChildItem -LiteralPath $Cache -Filter '*.partial').Count;cached=(Test-Path -LiteralPath (Join-Path $Cache 'helper.bin'))}
}

Describe 'auxiliary legacy download retry and cache cleanup' {
    It 'removes a failed network partial without publishing it' {
        $result=Invoke-VisionDownloadFixture (Join-Path $TestDrive 'network') 'network'
        $result.failed | Should Be $true
        $result.partial | Should Be 0
        $result.cached | Should Be $false
    }
    It 'removes a checksum failure without publishing it' {
        $result=Invoke-VisionDownloadFixture (Join-Path $TestDrive 'checksum') 'checksum'
        $result.failed | Should Be $true
        $result.partial | Should Be 0
        $result.cached | Should Be $false
    }
    It 'reuses a verified helper on repeat instead of downloading again' {
        $result=Invoke-VisionDownloadFixture (Join-Path $TestDrive 'repeat') '' -Repeat
        $result.failed | Should Be $false
        $result.calls | Should Be 1
        $result.partial | Should Be 0
        $result.cached | Should Be $true
    }
}
