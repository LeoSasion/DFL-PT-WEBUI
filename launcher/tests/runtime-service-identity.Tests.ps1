Describe 'PT runtime ownership' {
    BeforeAll {
        $source = (Get-Content (Join-Path $PSScriptRoot '../host/RuntimeServiceIdentity.cs') -Raw -Encoding UTF8)
        $source = $source.Replace('namespace DflPtWebUi.Launcher', 'namespace RuntimeOwnershipTests')
        $source = $source.Replace('internal static class', 'public static class')
        Add-Type -TypeDefinition $source -ReferencedAssemblies System.Web.Extensions
    }

    function New-Health([string]$Product, [string]$DflRoot) {
        return @{ ok = $true; data = @{ service = $Product; runtime = @{ current = @{ dflRoot = $DflRoot } } } } | ConvertTo-Json -Depth 6 -Compress
    }

    It 'accepts only the selected PT repository, including Windows path spelling' {
        $health = New-Health 'DFL-PT-WEBUI Local Runtime' 'E:/DFL-PT-WEBUI/_internal/DeepFaceLab/'
        [RuntimeOwnershipTests.RuntimeServiceIdentity]::Matches($health, 'e:\dfl-pt-webui') | Should Be $true
    }

    It 'refuses original DFL and another PT installation on the same ports' {
        foreach ($health in @(
            (New-Health 'DFL-WEBUI Local Runtime' 'E:/DFL-PT-WEBUI/_internal/DeepFaceLab'),
            (New-Health 'DFL-PT-WEBUI Local Runtime' 'E:/Another-PT/_internal/DeepFaceLab')
        )) {
            [RuntimeOwnershipTests.RuntimeServiceIdentity]::Matches($health, 'E:/DFL-PT-WEBUI') | Should Be $false
        }
    }

    It 'rejects malformed, incomplete, failed and oversized health responses' {
        foreach ($health in @('{', '{}', '{"ok":false,"data":{}}', '{"service":"DFL-PT-WEBUI Local Runtime","runtime":null}', (' ' * 65537))) {
            [RuntimeOwnershipTests.RuntimeServiceIdentity]::Matches($health, 'E:/DFL-PT-WEBUI') | Should Be $false
        }
    }

    It 'does not resolve relative or drive-relative service paths' {
        foreach ($candidate in @('_internal/DeepFaceLab', 'E:DFL-PT-WEBUI/_internal/DeepFaceLab', '\DFL-PT-WEBUI\_internal\DeepFaceLab')) {
            [RuntimeOwnershipTests.RuntimeServiceIdentity]::Matches(
                (New-Health 'DFL-PT-WEBUI Local Runtime' $candidate), 'E:/DFL-PT-WEBUI') | Should Be $false
        }
    }
}
