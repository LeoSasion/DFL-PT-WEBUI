@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

echo [最近使用] 已写入，如果需要清空历史请手动删除！

call "%~dp0..\_internal\setenv.bat"

mkdir "%WORKSPACE%\data_dst" 2>nul

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed extract-video ^
    --input-file "%WORKSPACE%\data_dst.*" ^
    --output-dir "%WORKSPACE%\data_dst" ^
    --fps 0

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%