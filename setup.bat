@echo off
setlocal EnableDelayedExpansion
title MarketSage Setup

echo ===============================================================================
echo MarketSage One-Click Setup
echo ===============================================================================
echo This script will install all dependencies for Frontend, Backend, and AI Model.
echo.

REM 1. Check for Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)
echo [OK] Python found.

REM 2. Check for Node.js
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js LTS version and try again.
    pause
    exit /b 1
)
echo [OK] Node.js found.
echo.

echo [1/4] Setting up Python Virtual Environment...
if not exist "venv_ml" (
    echo Creating virtual environment 'venv_ml'...
    python -m venv venv_ml
) else (
    echo Virtual environment 'venv_ml' already exists.
)

echo Activating virtual environment and installing AI dependencies...
call venv_ml\Scripts\activate.bat
pip install -r backend\prediction_api\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
call deactivate
echo.

echo [2/4] Installing Node Backend Dependencies...
pushd backend\node_server
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install backend dependencies.
    popd
    pause
    exit /b 1
)
popd
echo.

echo [3/4] Installing Frontend Dependencies...
pushd frontend
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    popd
    pause
    exit /b 1
)
popd
echo.

echo [4/4] Verifying Setup...
if exist "models\ensemble\bi_lstm_model.keras" (
    echo [OK] AI Models found.
) else (
    echo [WARNING] AI Models missing in 'models\ensemble'. Please ensure data is copied correctly.
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
