@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\ErrFaceFilter\ErrFaceFilter.py" "1"
set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%