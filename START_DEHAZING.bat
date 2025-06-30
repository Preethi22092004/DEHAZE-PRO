@echo off
echo ========================================
echo    DEHAZING SYSTEM STARTUP
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Starting the dehazing system...
echo.

REM Run the startup script
python start_dehazing_system.py

echo.
echo Press any key to exit...
pause >nul
