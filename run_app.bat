@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Streamlit 离线启动 (单标签页)

REM ===== 配置 =====
set "HOST=127.0.0.1"
set "PORT=8501"
set "APP_FILE=app.py"
set "VENV_DIR=.venv"
set "WHEEL_DIR=wheels"

REM ===== 切到脚本所在目录（支持空格路径） =====
cd /d "%~dp0"

echo [1/6] 检查 Python...
set "PY_CMD="
py --version >nul 2>&1 && set "PY_CMD=py"
if not defined PY_CMD python --version >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo 未检测到 Python 3.10+，离线无法自动安装。请先手动安装：https://www.python.org/
  pause & exit /b 1
)

echo [2/6] 创建/复用虚拟环境...
if not exist "%VENV_DIR%\Scripts\python.exe" (
  %PY_CMD% -m venv "%VENV_DIR%" || (echo 创建虚拟环境失败 & pause & exit /b 1)
)
set "PY=%CD%\%VENV_DIR%\Scripts\python.exe"

echo [3/6] 从本地 wheels 安装依赖...
if not exist "%WHEEL_DIR%" (
  echo 未找到 %WHEEL_DIR% 目录。请先在联网机器运行 prepare_offline.bat 生成依赖缓存并拷贝过来。
  pause & exit /b 1
)
"%PY%" -m pip install --upgrade pip >nul
if exist requirements.txt (
  "%PY%" -m pip install --no-index --find-links="%WHEEL_DIR%" -r requirements.txt || (
    echo 依赖安装失败（可能缺 wheel 或 Python 版本不匹配）。 & pause & exit /b 1
  )
)
"%PY%" -m pip install --no-index --find-links="%WHEEL_DIR%" streamlit-autorefresh >nul

echo [4/6] 准备 secrets（如无则生成模板）
if not exist ".streamlit" mkdir ".streamlit" >nul
if not exist ".streamlit\secrets.toml" (
  > ".streamlit\secrets.toml" (
    echo # 连网后在此填入 Gate.io API
    echo GATEIO_API_KEY=""
    echo GATEIO_SECRET=""
  )
)

echo [5/6] 打开浏览器（本地，仅打开一次）
start "" "http://%HOST%:%PORT%"
timeout /t 2 /nobreak >nul

echo [6/6] 启动 Streamlit（headless，避免自动再开一页）
call "%PY%" -m streamlit run "%APP_FILE%" --server.address=%HOST% --server.port=%PORT% --server.headless=true --browser.gatherUsageStats=false
set "ERR=%ERRORLEVEL%"

echo 结束，退出码 %ERR%
pause
exit /b %ERR%
