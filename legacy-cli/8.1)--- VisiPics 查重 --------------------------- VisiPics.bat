@echo off
setlocal EnableExtensions
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0open-native-tool.ps1" -View similarity -Side src
set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%
