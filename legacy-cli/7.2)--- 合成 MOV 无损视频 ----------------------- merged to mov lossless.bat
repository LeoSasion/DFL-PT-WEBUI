@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"


"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed video-from-sequence ^
    --input-dir "%WORKSPACE%\data_dst\merged" ^
    --output-file "%WORKSPACE%\result.mov" ^
    --reference-file "%WORKSPACE%\data_dst.*" ^
    --include-audio ^
    --lossless

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed video-from-sequence ^
    --input-dir "%WORKSPACE%\data_dst\merged_mask" ^
    --output-file "%WORKSPACE%\result_mask.mov" ^
    --reference-file "%WORKSPACE%\data_dst.*" ^
    --lossless

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%