@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please create it first using Python 3.11:
    echo   py -3.11 -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Dog Breed Recognition...
echo Open http://127.0.0.1:5000/ in your browser.
echo The model may take a few seconds to load. Press Ctrl+C to stop.
".venv\Scripts\python.exe" app.py

pause
