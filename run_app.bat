@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Streamlit Online Launch (Single Tab)

REM ===== CONFIGURATION =====
set "HOST=127.0.0.1"
set "PORT=8501"
set "APP_FILE=app.py"
set "VENV_DIR=.venv"

REM ===== Change to script directory (supports spaces in path) =====
cd /d "%~dp0"

echo [1/6] Checking Python...
set "PY_CMD="
py --version >nul 2>&1 && set "PY_CMD=py"
if not defined PY_CMD python --version >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo Python 3.10+ not detected, cannot auto-install. Please install manually: https://www.python.org/
  pause & exit /b 1
)

echo [2/6] Creating/reusing virtual environment...
if not exist "%VENV_DIR%\Scripts\python.exe" (
  %PY_CMD% -m venv "%VENV_DIR%" || (echo Failed to create virtual environment & pause & exit /b 1)
)
set "PY=%CD%\%VENV_DIR%\Scripts\python.exe"

echo [3/6] Installing dependencies from PyPI...
"%PY%" -m pip install --upgrade pip >nul
if exist requirements.txt (
  "%PY%" -m pip install -r requirements.txt || (
    echo Dependency installation failed. & pause & exit /b 1
  )
)
"%PY%" -m pip install streamlit-autorefresh >nul

echo [4/6] Preparing secrets (generate template if missing)
if not exist ".streamlit" mkdir ".streamlit" >nul
if not exist ".streamlit\secrets.toml" (
  > ".streamlit\secrets.toml" (
    echo # After connecting to the internet, fill in Gate.io API here
    echo GATEIO_API_KEY=""
    echo GATEIO_SECRET=""
  )
)

echo [5/6] Opening browser (local, only open once)
start "" "http://%HOST%:%PORT%"
timeout /t 2 /nobreak >nul

echo [6/6] Starting Streamlit (headless, avoid opening another tab)
call "%PY%" -m streamlit run "%APP_FILE%" --server.address=%HOST% --server.port=%PORT% --server.headless=true --browser.gatherUsageStats=false
set "ERR=%ERRORLEVEL%"

echo Finished, exit code %ERR%
pause
exit /b %ERR%
