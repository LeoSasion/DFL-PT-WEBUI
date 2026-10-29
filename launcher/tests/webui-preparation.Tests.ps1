[CmdletBinding()]
param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ([String]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $repo '.runtime/launcher-lifecycle-audit-20261030/webdeps-qa' }
$qa = [IO.Path]::GetFullPath($OutputDirectory)
$allowed = [IO.Path]::GetFullPath((Join-Path $repo '.runtime/launcher-lifecycle-audit-20261030/webdeps-qa')).TrimEnd('\')
if ($qa -ne $allowed -and -not $qa.StartsWith($allowed+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'QA output must remain in the isolated lifecycle audit folder.' }
New-Item -ItemType Directory -Path $qa -Force | Out-Null
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
$fake = Join-Path $qa 'fake-node.exe'
$harness = Join-Path $qa 'webui-preparation.exe'
& $compiler /nologo /langversion:5 /target:exe "/out:$fake" (Join-Path $PSScriptRoot 'webui-preparation.FakeNode.cs')
if ($LASTEXITCODE -ne 0) { throw 'Fake Node compilation failed.' }
& $compiler /nologo /langversion:5 /target:exe "/out:$harness" (Join-Path $PSScriptRoot 'webui-preparation.Harness.cs') (Join-Path $repo 'launcher/host/WebUiDependencies.cs') (Join-Path $repo 'launcher/host/PreparedRuntimeStore.cs')
if ($LASTEXITCODE -ne 0) { throw 'Real .NET 4.8 preparation harness compilation failed.' }
$fixtures = Join-Path $qa ('run-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtures | Out-Null
$start = New-Object Diagnostics.ProcessStartInfo
$start.FileName = $harness
$start.Arguments = '"'+$fixtures+'" "'+$fake+'"'
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$process = [Diagnostics.Process]::Start($start)
try {
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(60000)) { $process.Kill(); throw 'Preparation harness exceeded its bound.' }
    $output = @(($stdout.GetAwaiter().GetResult()+$stderr.GetAwaiter().GetResult()) -split '\r?\n' | Where-Object {$_})
    $code = $process.ExitCode
} finally { $process.Dispose() }
$output | Out-File -LiteralPath (Join-Path $qa 'preparation.log') -Encoding UTF8
$passed = @($output | Where-Object { "$_".StartsWith('PASS ') }).Count
@{passed=($code -eq 0);exitCode=$code;cases=$passed;fixtures=$fixtures;syntheticOnly=$true;framework='C#5 / .NET Framework 4.8';finishedAt=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $qa 'preparation-report.json') -Encoding UTF8
$output
if ($code -ne 0) { throw 'WebUI preparation regression failed.' }
