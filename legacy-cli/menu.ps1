[CmdletBinding()]
param([switch]$Check,[switch]$Preview)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$root = Split-Path -Parent $PSScriptRoot
$commands = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'commands.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$categories = @('视频处理 / Video','SRC 数据 / Source','DST 数据 / Destination','XSeg 遮罩 / Masks','ME 训练 / Training','ME 应用 / Merge and DFM','视频封装 / Encode','辅助工具 / Utilities')
function Show-Menu {
    Write-Host ''
    Write-Host 'DFL-PT-WEBUI | PyTorch ME 与配套工具' -ForegroundColor Green
    for ($index=0; $index -lt $categories.Count; $index++) { Write-Host ("[{0}] {1}" -f ($index+1), $categories[$index]) }
    Write-Host '[W] WebUI 管理器    [Q] 退出'
}
foreach ($command in $commands) {
    if ([IO.Path]::GetFileName([string]$command.file) -ne [string]$command.file -or
        -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $command.file) -PathType Leaf)) { throw "Invalid registered tool: $($command.file)" }
}
if ($Check) {
    if (-not (Test-Path -LiteralPath (Join-Path $root '_internal\setenv.bat')) -or
        -not (Test-Path -LiteralPath (Join-Path $root '_internal\DeepFaceLab\me.py'))) { throw 'The PyTorch project environment is incomplete.' }
    Write-Host "[OK] DFL-PT-WEBUI: $($commands.Count) fixed tool routes, 8 categories."
    exit 0
}
if ($Preview) { Show-Menu; exit 0 }
while ($true) {
    Show-Menu
    $selection = Read-Host '请选择'
    if ($selection -match '^[Qq]$') { break }
    if ($selection -match '^[Ww]$') {
        & $env:ComSpec '/d' '/c' ('call "' + (Join-Path $root '启动 WebUI.bat') + '"')
        continue
    }
    if ($selection -notmatch '^[1-8]$') { continue }
    $items = @($commands | Where-Object { [int]$_.category -eq [int]$selection })
    for ($index=0; $index -lt $items.Count; $index++) { Write-Host ("[{0}] {1}" -f ($index+1),$items[$index].title) }
    Write-Host '[0] 返回'
    $toolSelection = Read-Host '选择工具'
    $toolIndex = 0
    if (-not [int]::TryParse($toolSelection,[ref]$toolIndex) -or $toolIndex -lt 1 -or $toolIndex -gt $items.Count) { continue }
    $target = Join-Path $PSScriptRoot $items[$toolIndex-1].file
    & $env:ComSpec '/d' '/c' ('call "' + $target + '"')
    if ($LASTEXITCODE -ne 0) { Write-Host "工具退出码：$LASTEXITCODE" -ForegroundColor Yellow }
}
