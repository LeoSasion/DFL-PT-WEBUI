[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('EbSynth','XnViewMP','VisiPics')][string]$Tool,
    [string]$ExecutablePath,
    [string]$InputDirectory,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root '.launcher-install/external-tools.json'
$defaults = @{
    EbSynth = '_internal/EbSynth/EbSynth.exe'
    XnViewMP = '_internal/XnViewMP/xnviewmp.exe'
    VisiPics = '_internal/VisiPics/VisiPics.exe'
}
$configured = @{}
if (Test-Path -LiteralPath $configPath) {
    $stored = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($property in $stored.PSObject.Properties) { $configured[$property.Name] = [string]$property.Value }
}
if ($ExecutablePath) {
    $executable = [IO.Path]::GetFullPath($ExecutablePath)
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf) -or
        [IO.Path]::GetFileName($executable) -ine [IO.Path]::GetFileName($defaults[$Tool])) {
        throw "Choose the actual $Tool application executable."
    }
    $configured[$Tool] = $executable
    New-Item -ItemType Directory -Path (Split-Path -Parent $configPath) -Force | Out-Null
    [IO.File]::WriteAllText($configPath,($configured | ConvertTo-Json),(New-Object Text.UTF8Encoding($false)))
} elseif ($configured.ContainsKey($Tool)) { $executable = $configured[$Tool] }
else { $executable = Join-Path $root $defaults[$Tool] }
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    Write-Host "[DFL-PT-WEBUI] $Tool is optional and its application has not been provided: $executable" -ForegroundColor Yellow
    Write-Host "Configure it with: powershell.exe -File legacy-cli/external-tools.ps1 -Tool $Tool -ExecutablePath <application.exe> -ConfigureOnly"
    exit 1
}
if ($ConfigureOnly) { Write-Host "[OK] $Tool configured: $executable"; exit 0 }
$parameters = @{ FilePath=$executable; WorkingDirectory=(Split-Path -Parent $executable); WindowStyle='Normal' }
if ($InputDirectory) {
    $inputPath = [IO.Path]::GetFullPath($InputDirectory)
    if (-not (Test-Path -LiteralPath $inputPath -PathType Container)) { throw "Input directory is missing: $inputPath" }
    $parameters.ArgumentList = @('"'+$inputPath+'"')
}
Start-Process @parameters
