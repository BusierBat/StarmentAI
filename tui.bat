@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u tui.py
) else (
    python -u tui.py
)

exit /b %errorlevel%