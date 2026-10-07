@echo off
chcp 65001 >nul
rem DFL-PT-WEBUI project-local environment. Preserve Windows user directories.
for %%D in ("%~dp0..") do set "DFL_PT_ROOT=%%~fD"
set "INTERNAL=%DFL_PT_ROOT%\_internal"
set "PYTHON_PATH=%DFL_PT_ROOT%\.venv\Scripts"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe"
set "PYTHONEXECUTABLE=%PYTHON_EXECUTABLE%"
set "PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe"
set "PYTHONWEXECUTABLE=%PYTHONW_EXECUTABLE%"
set "PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%"
set "PYTHON_LIB_PATH=%DFL_PT_ROOT%\.venv\Lib\site-packages"
set "NODE_BIN_PATH=%INTERNAL%\node\bin"
set "FFMPEG_PATH=%INTERNAL%\ffmpeg"
set "XNVIEWMP_PATH=%INTERNAL%\XnViewMP"
set "PATH=%PYTHON_PATH%;%NODE_BIN_PATH%;%FFMPEG_PATH%;%XNVIEWMP_PATH%;%PATH%"
rem Re-read the active selection on every call. Fail closed rather than target
rem the wrong project's images after WebUI has switched projects.
set "WORKSPACE="
set "DFL_ACTIVE_PROJECT_ID="
for /f "usebackq tokens=1,* delims=|" %%I in (`""%PYTHON_EXECUTABLE%" -I "%DFL_PT_ROOT%\launcher\resolve-active-project.py" --root "%DFL_PT_ROOT%" --field bat"`) do (
  set "DFL_ACTIVE_PROJECT_ID=%%I"
  set "WORKSPACE=%%J"
)
if not defined WORKSPACE (
  echo [ERROR] Active project could not be resolved. Reopen WebUI project selection.
  exit /b 2
)
set "DFL_WORKSPACE=%WORKSPACE%"
echo [DFL-PT-WEBUI] Project: %DFL_ACTIVE_PROJECT_ID%  Workspace: "%WORKSPACE%"
set "DFL_ROOT=%INTERNAL%\DeepFaceLab"
