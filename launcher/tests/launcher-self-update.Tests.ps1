$ErrorActionPreference='Stop'
Describe 'isolated native launcher replacement and startup rollback' {
    BeforeAll {
        $script:updateTestRoot=Join-Path $TestDrive 'native-update'
        New-Item -ItemType Directory -Path $script:updateTestRoot|Out-Null
        $script:compiler=Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        $wpf=Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/WPF'
        $padding=Join-Path $script:updateTestRoot 'padding.bin';[IO.File]::WriteAllBytes($padding,(New-Object byte[] (1024*1024)))
        $script:updateImage=Join-Path $script:updateTestRoot 'helper.exe'
        $refs=@('/r:System.Web.Extensions.dll','/r:System.Xaml.dll',('/r:'+(Join-Path $wpf 'PresentationFramework.dll')),('/r:'+(Join-Path $wpf 'PresentationCore.dll')),('/r:'+(Join-Path $wpf 'WindowsBase.dll')))
        $sources=@('launcher-self-update.Harness.cs','../host/LauncherSelfUpdater.cs','../host/LauncherUpdateMetadata.cs','../host/LauncherUpdateFileSystem.cs')|ForEach-Object{[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $_))}
        $compilerOutput=& $script:compiler /nologo /target:exe ("/out:"+$script:updateImage) ("/resource:"+$padding) @refs @sources 2>&1
        if($LASTEXITCODE -ne 0){throw ('Native updater acceptance compilation failed: '+($compilerOutput -join "`n"))}
        function Invoke-UpdateCase([string]$Name,[bool]$FailStart) {
            $case=Join-Path $script:updateTestRoot $Name;$settings=Join-Path $case 'settings';$updates=Join-Path $settings 'updates'
            New-Item -ItemType Directory -Path $updates -Force|Out-Null
            $target=Join-Path $case 'launcher.exe';Copy-Item -LiteralPath $script:updateImage -Destination $target
            [IO.File]::AppendAllText($target,'synthetic old build trailer')
            $helper=Join-Path $updates 'helper.exe';Copy-Item -LiteralPath $script:updateImage -Destination $helper
            $oldHash=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant();$newHash=(Get-FileHash -LiteralPath $helper).Hash.ToLowerInvariant()
            $old=Start-Process -FilePath $target -ArgumentList '--old' -PassThru -WindowStyle Hidden
            $request=Join-Path $updates 'apply-case.json'
            [IO.File]::WriteAllText($request,(@{SchemaVersion=1;TargetPath=$target;HelperPath=$helper;OldProcessId=$old.Id;ExpectedCurrentSha256=$oldHash;ExpectedNewSha256=$newHash;ExpectedNewSize=(Get-Item -LiteralPath $helper).Length}|ConvertTo-Json))
            $start=New-Object Diagnostics.ProcessStartInfo;$start.FileName=$helper;$start.Arguments='--apply-update "'+$request+'"'
            $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardError=$true;$start.RedirectStandardOutput=$true
            $start.EnvironmentVariables['DFL_UPDATE_TEST_SETTINGS']=$settings
            $start.EnvironmentVariables['DFL_UPDATE_TEST_FAIL_START']=$(if($FailStart){'1'}else{'0'})
            $process=[Diagnostics.Process]::Start($start)
            try {
                $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
                if(-not $process.WaitForExit(15000)){$process.Kill();throw 'Isolated updater exceeded acceptance bound'}
                $errorText=$stderr.GetAwaiter().GetResult()
                if(-not $FailStart -and $process.ExitCode -ne 0){throw $errorText}
                # Wait only for the private restarted process and its cleanup task.
                $timer=[Diagnostics.Stopwatch]::StartNew()
                while((Test-Path -LiteralPath $request) -and -not $FailStart -and $timer.ElapsedMilliseconds -lt 5000){Start-Sleep -Milliseconds 100}
                return @{exitCode=$process.ExitCode;oldHash=$oldHash;newHash=$newHash;actual=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant();requestExists=(Test-Path -LiteralPath $request);partialCount=@(Get-ChildItem -LiteralPath $case -Filter '*.tmp').Count;backupCount=@(Get-ChildItem -LiteralPath $case -Filter '*.previous-*.exe').Count}
            }finally{$process.Dispose();$old.Dispose()}
        }
    }
    It 'replaces once and cleans backup only after the new process reports readiness' {
        $result=Invoke-UpdateCase 'success' $false
        $result.exitCode|Should Be 0
        $result.actual|Should Be $result.newHash
        $result.requestExists|Should Be $false
        $result.partialCount|Should Be 0
        $result.backupCount|Should Be 0
    }
    It 'restores the exact old executable if the restarted version exits before readiness' {
        $result=Invoke-UpdateCase 'failed-start' $true
        $result.exitCode|Should Be 7
        $result.actual|Should Be $result.oldHash
        $result.requestExists|Should Be $true
        $result.partialCount|Should Be 0
    }
}
