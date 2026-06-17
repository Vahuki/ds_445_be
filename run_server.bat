@echo off
echo === Thiet lap moi truong Python 3.12 (Moi) ===

REM Python 3.12 install root
set PYTHONHOME=F:\Python312
set PATH=F:\Python312;F:\Python312\Scripts;%PATH%

echo PYTHONHOME=%PYTHONHOME%
echo.

echo === Kiem tra Python ===
python --version
if %errorlevel% neq 0 (
    echo [LOI] python.exe khong chay duoc!
    pause
    exit /b 1
)

echo.
echo === Khoi dong FastAPI server tai http://localhost:8000 ===
python app.py
pause
