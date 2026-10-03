@echo off

chcp 65001 > nul

title DFL-PT-WEBUI Tools


echo.

call "%~dp0..\_internal\setenv.bat"
echo 请将遮罩模型放入workspace\xseg_model\

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg apply ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --model-dir "%WORKSPACE%\xseg_model"

set "TOOL_EXIT=%errorlevel%"
pause
exit /b %TOOL_EXIT%