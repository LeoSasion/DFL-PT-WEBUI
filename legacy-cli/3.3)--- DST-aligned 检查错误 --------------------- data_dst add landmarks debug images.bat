@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

echo 本程序是向data_dst\aligned文件夹添加landmarks-debug图片文件供手动检查

echo.

echo 如果需要自动检查，请使用 8.1)--- Landmarks自动识错 ------------------------ Landmarks auto check

call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --add-landmarks-debug-images

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%