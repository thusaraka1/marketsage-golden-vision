@echo off
setlocal EnableDelayedExpansion
title MarketSage Setup

REM ================================================================================
REM Get the directory where this script is located (handles running from any path)
REM ================================================================================
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ===============================================================================
echo MarketSage One-Click Setup
echo ===============================================================================
echo This script will install all dependencies for Frontend, Backend, and AI Model.
echo Working Directory: %CD%
echo.

REM ================================================================================
REM 1. Check for Python and verify version
REM ================================================================================
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9, 3.10, or 3.11 and try again.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] Python found: %PYVER%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

REM Check if Python version is compatible with TensorFlow (3.9-3.11)
if "%PYMAJOR%"=="3" (
    if %PYMINOR% LSS 9 (
        echo [ERROR] Python %PYVER% is too old. TensorFlow requires Python 3.9-3.11.
        echo Please install Python 3.9, 3.10, or 3.11.
        pause
        exit /b 1
    )
    if %PYMINOR% GTR 11 (
        echo [ERROR] Python %PYVER% is too new. TensorFlow does not yet support Python 3.12+.
        echo Please install Python 3.9, 3.10, or 3.11.
        echo Download from: https://www.python.org/downloads/release/python-3119/
        pause
        exit /b 1
    )
) else (
    echo [ERROR] Python 2.x is not supported. Please install Python 3.9-3.11.
    pause
    exit /b 1
)
echo [OK] Python version is compatible with TensorFlow.

REM ================================================================================
REM 2. Check for Node.js
REM ================================================================================
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js LTS version and try again.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found.
echo.

REM ================================================================================
REM Step 1: Python Virtual Environment
REM ================================================================================
echo [1/4] Setting up Python Virtual Environment...
if not exist "venv_ml" (
    echo Creating virtual environment 'venv_ml'...
    python -m venv venv_ml
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment 'venv_ml' already exists.
)

echo Activating virtual environment and installing AI dependencies...
call venv_ml\Scripts\activate.bat

REM Check if requirements.txt exists before trying to install
if not exist "backend\prediction_api\requirements.txt" (
    echo [ERROR] requirements.txt not found at: backend\prediction_api\requirements.txt
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Installing Python packages (this may take a few minutes)...
pip install --upgrade pip >nul 2>nul
pip install -r backend\prediction_api\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    echo.
    echo Common fixes:
    echo   1. Make sure you have Python 3.9, 3.10, or 3.11 (not 3.12+)
    echo   2. Make sure you have 64-bit Python (not 32-bit)
    echo   3. Try: pip install tensorflow-cpu instead of tensorflow
    pause
    exit /b 1
)
call deactivate
echo [OK] Python dependencies installed.
echo.

REM ================================================================================
REM Step 2: Node Backend Dependencies
REM ================================================================================
echo [2/4] Installing Node Backend Dependencies...
if not exist "backend\node_server\package.json" (
    echo [ERROR] package.json not found at: backend\node_server\
    pause
    exit /b 1
)
pushd backend\node_server
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install backend dependencies.
    popd
    pause
    exit /b 1
)
popd
echo [OK] Backend dependencies installed.
echo.

REM ================================================================================
REM Step 3: Frontend Dependencies
REM ================================================================================
echo [3/4] Installing Frontend Dependencies...
if not exist "frontend\package.json" (
    echo [ERROR] package.json not found at: frontend\
    pause
    exit /b 1
)
pushd frontend
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    popd
    pause
    exit /b 1
)
popd
echo [OK] Frontend dependencies installed.
echo.

REM ================================================================================
REM Step 4: Verify Setup
REM ================================================================================
echo [4/4] Verifying Setup...
if exist "models\ensemble\bi_lstm_model.keras" (
    echo [OK] AI Models found.
) else (
    echo [WARNING] AI Models missing in 'models\ensemble'.
)

if exist "data\historical" (
    echo [OK] Historical data found.
) else (
    echo [WARNING] Historical data missing in 'data\historical'.
)

echo.
echo ===============================================================================
echo Setup Complete!
echo ===============================================================================
echo.
echo [IMPORTANT] Admin Credentials:
echo If this is a new installation, a default Super Admin account has been created:
echo - Username: admin
echo - Password: admin
echo.
echo You can now run the system using 'run.bat'.
echo.
pause
