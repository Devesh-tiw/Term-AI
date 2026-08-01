@echo off
cd /d "%~dp0"

if not exist .env (
    echo WARNING: No .env file found! Please copy .env.example to .env and add your API key.
    pause
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the Python app
python ai_app.py

pause
