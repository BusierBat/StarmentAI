@echo off
setlocal

where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama is not installed or not in PATH.
    exit /b 1
)

ollama create starmentai -f Modelfile