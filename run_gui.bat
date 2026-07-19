@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%PYTHON%" -m radar.gui %*
if errorlevel 1 pause
endlocal
