@echo off
echo Starting Dehazing System in Production Mode...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if the virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install requirements if needed
if not exist venv\Scripts\flask.exe (
    echo Installing requirements...
    pip install -r requirements.txt
    pip install waitress
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        exit /b 1
    )
)

REM Generate model weights if they don't exist
if not exist static\models\weights\aod_net.pth (
    echo Generating model weights...
    python generate_weights.py
    if %ERRORLEVEL% neq 0 (
        echo Failed to generate model weights.
        exit /b 1
    )
)

REM Set environment variables for production
set FLASK_ENV=production
set FLASK_DEBUG=0

REM Run the application with Waitress (production server)
echo Starting the production server...
python -m waitress --port=5000 main:app

REM Deactivate the virtual environment when done
call venv\Scripts\deactivate.bat
