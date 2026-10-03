@echo off
setlocal EnableExtensions
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0external-tools.ps1" -Tool VisiPics
set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%
