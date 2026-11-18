@echo off
setlocal

where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama is not installed or not in PATH.
    exit /b 1
)

ollama list | findstr /C:"starmentai" >nul
if errorlevel 1 (
    echo Creating StarmentAI model...
    ollama create starmentai -f Modelfile
    if errorlevel 1 exit /b 1
)

python main.py