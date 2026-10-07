@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" extract ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\aligned" ^
    --output-debug ^
    --detector s3fd ^
    --max-faces-from-image 0 ^
    --manual-fix

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%