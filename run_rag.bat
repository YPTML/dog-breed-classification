@echo off
setlocal
cd /d "%~dp0"
title He Thong RAG - Dog Breed Classification

echo =======================================================
echo    HE THONG RAG HOI DAP - DOG BREED CLASSIFICATION
echo =======================================================
echo.

if not defined RAG_PYTHON (
    if exist "%LOCALAPPDATA%\DogBreedRAG\venv\Scripts\python.exe" (
        set "RAG_PYTHON=%LOCALAPPDATA%\DogBreedRAG\venv\Scripts\python.exe"
    ) else (
        set "RAG_PYTHON=%~dp0.venv\Scripts\python.exe"
    )
)
if not exist "%RAG_PYTHON%" (
    echo [ERROR] Khong tim thay Python RAG. Xem docs\RAG.md.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
if not defined PYTHONPYCACHEPREFIX set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\DogBreedRAG\pycache"
set TF_ENABLE_ONEDNN_OPTS=0

rem Chay script run_rag.py voi tat ca tham so truyen vao (mac dinh la khoi chay Web Server)
"%RAG_PYTHON%" run_rag.py %*

set "EXITCODE=%ERRORLEVEL%"
echo.
echo [INFO] Ung dung da dung. Ma thoat: %EXITCODE%
if "%1"=="" pause
exit /b %EXITCODE%
