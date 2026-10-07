@echo off
setlocal EnableExtensions
chcp 65001 >nul
title DFL-PT-WEBUI ME Training
call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%
echo PyTorch ME - Ctrl+C saves and stops training.
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME --force-model-name ME --no-preview
set "ME_EXIT=%errorlevel%"
pause
exit /b %ME_EXIT%
