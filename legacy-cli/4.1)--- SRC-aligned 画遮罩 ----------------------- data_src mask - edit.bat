@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg editor ^
    --input-dir "%WORKSPACE%\data_src\aligned"

if %errorlevel% NEQ 0 (
  pause
)