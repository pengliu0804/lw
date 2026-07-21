@echo off
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=%CD%\.uv-cache"

uv sync --frozen
if errorlevel 1 exit /b %errorlevel%

uv run --frozen python train_demo.py --steps 2 --batch-size 1

