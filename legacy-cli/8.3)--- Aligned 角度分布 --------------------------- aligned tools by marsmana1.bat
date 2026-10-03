@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"

cd /d "%~dp0..\_internal\facesets"
"%PYTHON_EXECUTABLE%" facesets.py

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%