@echo off
setlocal EnableExtensions
chcp 65001 >nul
call "%~dp0..\_internal\setenv.bat"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0external-tools.ps1" -Tool XnViewMP -InputDirectory "%WORKSPACE%\data_dst\aligned"
set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%
