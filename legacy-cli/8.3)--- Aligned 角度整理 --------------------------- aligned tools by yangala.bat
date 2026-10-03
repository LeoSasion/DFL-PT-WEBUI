@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

echo 本工具的三种打开方式：
echo.

echo 1：拖动aligned文件夹到bat文件图标上

echo 2：拖动aligned文件夹到cmd窗口内

echo 3：复制aligned文件夹路径粘贴到cmd窗口内
echo.
echo.

cd /d "%~dp0.."
call "%~dp0..\_internal\setenv.bat"

set "ALIGNED_DIR=%~1"
if not defined ALIGNED_DIR (
    set /p "ALIGNED_DIR=请输入 aligned 文件夹完整路径："
)

if not exist "%ALIGNED_DIR%\" (
    echo 未找到 aligned 文件夹：%ALIGNED_DIR%
    pause
    exit /b 1
)

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\yaw_image_filter.py" "%ALIGNED_DIR%"

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%