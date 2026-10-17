@echo off
setlocal
cd /d "%~dp0"
title Dog Breed Recognition

echo ==========================================
echo        DOG BREED RECOGNITION
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" goto NO_PYTHON
if not exist "app.py" goto NO_APP

set TF_ENABLE_ONEDNN_OPTS=0

echo [INFO] Starting Dog Breed Recognition...
echo [INFO] Loading TensorFlow and model...
echo [INFO] Please wait. First startup may take longer.
echo.

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:5000/'; while($true){try{$r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2 -ErrorAction Stop;if($r.StatusCode -ge 200){Start-Process $u;break}}catch{};Start-Sleep -Seconds 2}"

".venv\Scripts\python.exe" -u app.py

set "EXITCODE=%ERRORLEVEL%"
echo.
echo [INFO] Application stopped. Exit code: %EXITCODE%
pause
exit /b %EXITCODE%

:NO_PYTHON
echo [ERROR] Virtual environment not found.
echo [ERROR] Missing .venv\Scripts\python.exe
pause
exit /b 1

:NO_APP
echo [ERROR] app.py not found.
pause
exit /b 1