@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

echo 注意！将清除手画标记的Xseg信息！请备份好这部分素材！

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%
echo.

call "%~dp0..\_internal\setenv.bat"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg remove_labels ^
    --input-dir "%WORKSPACE%\data_dst\aligned"

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%