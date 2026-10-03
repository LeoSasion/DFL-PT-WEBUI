@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\install-source.ps1" -ProjectRoot "%~dp0." %*
set "INSTALL_EXIT_CODE=%ERRORLEVEL%"
if not "%INSTALL_EXIT_CODE%"=="0" echo Source installation failed. See the message above and launcher\runtime-README.md.
if "%~1"=="" pause
exit /b %INSTALL_EXIT_CODE%
