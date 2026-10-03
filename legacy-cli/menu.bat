@echo off
setlocal EnableExtensions
chcp 65001 >nul
if /i "%~1"=="--check" goto check
if /i "%~1"=="--preview" goto preview
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0menu.ps1"
exit /b %errorlevel%
:check
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0menu.ps1" -Check
exit /b %errorlevel%
:preview
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0menu.ps1" -Preview
exit /b %errorlevel%
