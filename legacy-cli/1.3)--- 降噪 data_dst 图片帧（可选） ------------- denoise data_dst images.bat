@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed denoise-image-sequence ^
    --input-dir "%WORKSPACE%\data_dst"

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%