@echo off
setlocal
chcp 65001 > nul

set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [FAIL] Project Python not found: %PYTHON_EXE%
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\launcher\setup-runtime.ps1" -NoNetwork
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" "%~dp0smoke_test.py"
set "SMOKE_EXIT=%ERRORLEVEL%"

endlocal & exit /b %SMOKE_EXIT%
