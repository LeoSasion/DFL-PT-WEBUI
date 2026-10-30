# Shared by embedded bootstrap and source installers; no global configuration.
$script:DflDownloadFailures=@{}
function Get-DflDownloadPolicy {
    $path=Join-Path $PSScriptRoot 'download-sources.json'
    if(-not (Test-Path -LiteralPath $path)){$path=Join-Path (Split-Path -Parent $PSScriptRoot) 'release/download-sources.json'}
    $value=Get-Content -LiteralPath $path -Raw -Encoding UTF8|ConvertFrom-Json
    if($value.schemaVersion -ne 1){throw 'Unsupported download source policy'}
    return $value
}
function Get-DflDownloadUrls([string]$Url) {
    $uri=[Uri]$Url
    if($uri.Scheme -ne 'https' -or $uri.UserInfo -or $uri.Query -or $uri.Fragment){return $Url}
    $urls=@($Url)
    foreach($rule in (Get-DflDownloadPolicy).rules){
        if($Url.StartsWith($rule.prefix,[StringComparison]::Ordinal)){$urls+=([string]$rule.mirror+$Url.Substring($rule.prefix.Length))}
    }
    if($env:DFL_MIRROR -eq 'official'){return $Url}
    if($env:DFL_MIRROR -eq 'china' -and $urls.Count -gt 1){return @($urls|Select-Object -Skip 1)+@($Url)}
    # Windows PowerShell's Sort-Object is not stable for equal priorities.
    $ready=@();$cooling=@();$now=[DateTime]::UtcNow
    foreach($source in $urls){
        $hostName=([Uri]$source).Host
        if($script:DflDownloadFailures.ContainsKey($hostName) -and $script:DflDownloadFailures[$hostName] -gt $now){$cooling+=$source}
        else{$ready+=$source}
    }
    return $ready+$cooling
}
function Invoke-DflPinnedDownload([string]$Url,[string]$Destination,[string]$Sha256) {
    if(Test-Path -LiteralPath $Destination){throw 'Download target already exists'}
    $errors=@()
    foreach($source in Get-DflDownloadUrls $Url){
        try{
            Write-Host ('[download] '+([Uri]$source).Host)
            Invoke-WebRequest -UseBasicParsing -Uri $source -OutFile $Destination -TimeoutSec 20
            if((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ine $Sha256){throw 'Pinned download SHA-256 mismatch'}
            return
        }catch{
            $script:DflDownloadFailures[([Uri]$source).Host]=[DateTime]::UtcNow.AddSeconds(60)
            $errors+=(([Uri]$source).Host+': '+$_.Exception.Message)
            if(Test-Path -LiteralPath $Destination -PathType Leaf){Remove-Item -LiteralPath $Destination -Force}
        }
    }
    throw ('All download sources failed: '+($errors -join '; '))
}
function Get-DflNpmRegistries {
    $urls=@((Get-DflDownloadPolicy).npmRegistries)
    if($env:DFL_MIRROR -eq 'official'){return $urls[0]}
    if($env:DFL_MIRROR -eq 'china'){return @($urls|Select-Object -Skip 1)+@($urls[0])}
    return $urls
}
