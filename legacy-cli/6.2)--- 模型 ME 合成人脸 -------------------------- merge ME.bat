@echo off
setlocal EnableExtensions
chcp 65001 >nul
title DFL-PT-WEBUI ME Merge
call "%~dp0..\_internal\setenv.bat"
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" merge ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\merged" ^
    --output-mask-dir "%WORKSPACE%\data_dst\merged_mask" ^
    --aligned-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME --force-model-name ME ^
    --xseg-dir "%WORKSPACE%\xseg_model"
set "ME_EXIT=%errorlevel%"
pause
exit /b %ME_EXIT%
