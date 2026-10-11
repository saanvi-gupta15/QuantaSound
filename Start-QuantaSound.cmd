@echo off
setlocal
cd /d "%~dp0"
set "QUANTA_PYTHON=python"
if exist "..\..\work\.venv\Scripts\python.exe" set "QUANTA_PYTHON=..\..\work\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "QUANTA_PYTHON=.venv\Scripts\python.exe"
echo Open http://127.0.0.1:7860 after startup. Keep this window open.
"%QUANTA_PYTHON%" -m demo4.app %*
pause
