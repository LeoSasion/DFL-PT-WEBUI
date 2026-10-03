@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"

mkdir "%WORKSPACE%\data_src" 2>nul

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed extract-video ^
    --input-file "%WORKSPACE%\data_src.*" ^
    --output-dir "%WORKSPACE%\data_src"

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%