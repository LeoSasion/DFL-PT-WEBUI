@echo off
setlocal EnableExtensions
chcp 65001 >nul
title DFL-PT-WEBUI ME DFM Export
call "%~dp0..\_internal\setenv.bat"
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME --force-model-name ME
set "ME_EXIT=%errorlevel%"
pause
exit /b %ME_EXIT%
