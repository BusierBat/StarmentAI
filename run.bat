@echo off
setlocal
cd /d "%~dp0"

where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama is not installed or not in PATH.
    exit /b 1
)

rem Ensure the Ollama daemon is up (ollama list auto-starts it).
ollama list >nul 2>nul
if errorlevel 1 (
    echo Starting Ollama...
    start "" /B ollama serve
    timeout /t 3 /nobreak >nul
)

rem Create the model if it does not exist yet.
ollama list | findstr /C:"starmentai" >nul
if errorlevel 1 (
    echo Creating StarmentAI model...
    ollama create starmentai -f Modelfile
    if errorlevel 1 exit /b 1
)

rem Install Python dependencies if needed.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
    ".venv\Scripts\python.exe" -u tui.py
) else (
    python -m pip install -q -r requirements.txt
    python -u tui.py
)

exit /b %errorlevel%=pause
