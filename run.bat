@echo off
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found. Install uv first: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

set "UV_CACHE_DIR=%CD%\.uv-cache"
set "UV_HTTP_TIMEOUT=300"

echo Using uv-managed environment...
uv sync --no-dev --inexact
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)

echo Starting AI live-translate service [LOCAL Kotoba Whisper]... browser opens http://127.0.0.1:5231
echo First run downloads the Kotoba Whisper model to the uv/HuggingFace cache - please wait.
echo Press Ctrl+C to stop.
uv run --no-sync python app.py

echo.
echo Service exited. Press any key to close.
pause >nul
