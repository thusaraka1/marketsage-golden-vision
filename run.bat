@echo off
setlocal
title MarketSage System

REM ================================================================================
REM Get the directory where this script is located (handles running from any path)
REM ================================================================================
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ===============================================================================
echo Starting MarketSage System...
echo ===============================================================================
echo Working Directory: %CD%
echo.

REM 1. Start Prediction API (Python)
echo [1/3] Starting Prediction API (Port 5000)...
start "MarketSage - AI Prediction API" cmd /k "cd /d %SCRIPT_DIR% && call venv_ml\Scripts\activate.bat && cd backend\prediction_api && python app.py"

REM 2. Start Node Server (Backend)
echo [2/3] Starting Backend Server (Port 3001)...
start "MarketSage - Node Backend" cmd /k "cd /d %SCRIPT_DIR%backend\node_server && npm run dev"

REM 3. Start Frontend (React)
echo [3/3] Starting Frontend (Port 8080)...
start "MarketSage - Frontend" cmd /k "cd /d %SCRIPT_DIR%frontend && npm run dev"

REM Wait a moment for servers to spin up
timeout /t 5 >nul

REM Open in Browser
echo.
echo Opening MarketSage in your default browser...
start http://localhost:8080

echo.
echo ===============================================================================
echo System is Running!
echo ===============================================================================
echo - AI API: http://localhost:5000
echo - Backend: http://localhost:3001
echo - Frontend: http://localhost:8080
echo.
echo Close the popup command windows to stop the servers.
pause
