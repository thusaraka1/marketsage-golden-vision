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
REM 1. Check for Python Launcher and find compatible version
REM ================================================================================
where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher 'py' not found.
    echo Please install Python 3.9, 3.10, or 3.11 from python.org
    pause
    exit /b 1
)

REM Try to find Python 3.11, 3.10, or 3.9 (in order of preference)
set "PYTHON_CMD="
py -3.11 --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto :found_python
)
py -3.10 --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto :found_python
)
py -3.9 --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.9"
    goto :found_python
)

echo [ERROR] No compatible Python version found.
echo TensorFlow requires Python 3.9, 3.10, or 3.11.
echo You have Python installed but not a compatible version.
echo.
echo Available Python versions:
py -0
echo.
echo Please install Python 3.11 from: https://www.python.org/downloads/release/python-31114/
pause
exit /b 1

:found_python
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%i
echo [OK] Using: %PYTHON_CMD% (Python %PYVER%)

REM ================================================================================
REM 2. Check for Node.js
REM ================================================================================
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js LTS version from: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found.
echo.

REM ================================================================================
REM Step 1: Python Virtual Environment
REM ================================================================================
echo [1/4] Setting up Python Virtual Environment...
if exist "venv_ml" (
    echo Deleting old virtual environment (may have wrong Python version)...
    rmdir /s /q venv_ml
)

echo Creating virtual environment with %PYTHON_CMD%...
%PYTHON_CMD% -m venv venv_ml
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment and installing AI dependencies...
call venv_ml\Scripts\activate.bat

REM Check if requirements.txt exists
if not exist "backend\prediction_api\requirements.txt" (
    echo [ERROR] requirements.txt not found at: backend\prediction_api\requirements.txt
    pause
    exit /b 1
)

echo Installing Python packages (this may take a few minutes)...
python -m pip install --upgrade pip >nul 2>nul
pip install -r backend\prediction_api\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
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
