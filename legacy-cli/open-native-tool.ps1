[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('aligned','similarity')][string]$View,
    [ValidateSet('src','dst')][string]$Side='src',
    [switch]$Check
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$python=Join-Path $root '.venv/Scripts/python.exe'
$selection=& $python -I (Join-Path $root 'launcher/resolve-active-project.py') --root $root
if ($LASTEXITCODE -ne 0) { throw 'Active project could not be resolved; repair the project registry first.' }
$project=$selection | ConvertFrom-Json
$url="http://127.0.0.1:4173/#/$View/$Side"
if ($Check) {
    @{view=$View;side=$Side;url=$url;projectId=$project.id;workspace=$project.workspace} | ConvertTo-Json -Compress
    exit 0
}
$node=Join-Path $root '_internal/node/bin/node.exe'
& $node (Join-Path $root 'webui/scripts/local-manager.mjs') start
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Process $url
