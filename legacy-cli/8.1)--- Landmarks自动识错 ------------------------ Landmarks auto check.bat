@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\ErrFaceFilter\ErrFaceFilter.py" "1"
set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%