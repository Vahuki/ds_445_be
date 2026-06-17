@echo off
echo ================================================
echo   SUA CHUA PYTHON 3.13 - Tu dong tai va cai
echo ================================================
echo.

REM Kiem tra neu python.exe hoat dong duoc roi thi thoat
"C:\Users\ADMIN\python.exe" --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python 3.13 dang hoat dong. Khong can sua.
    goto run_server
)

echo [!] Python 3.13 bi hong (thieu python313.dll)
echo [*] Dang tai Python 3.13.5 installer...
echo.

REM Tai Python 3.13 installer bang PowerShell
powershell -Command "& {$url='https://www.python.org/ftp/python/3.13.5/python-3.13.5-amd64.exe'; $out='%TEMP%\python313_installer.exe'; Write-Host 'Dang tai... '; Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing; Write-Host 'Tai xong!'}"

if not exist "%TEMP%\python313_installer.exe" (
    echo [LOI] Khong tai duoc. Vui long tai thu cong:
    echo       https://www.python.org/ftp/python/3.13.5/python-3.13.5-amd64.exe
    echo       Sau do chay installer va chon "Repair"
    pause
    exit /b 1
)

echo.
echo [*] Dang chay installer (che do Repair/Install)...
echo     - Vui long chon "Install Now" hoac "Repair"
echo     - Sau khi xong, dong lai cua so nay va chay lai run_server.bat
echo.

REM Chay installer o che do tu dong (install for current user)
"%TEMP%\python313_installer.exe" /passive InstallAllUsers=0 PrependPath=0 TargetDir="F:\DevProfileData" Include_pip=1

echo.
echo [*] Cai dat xong. Dang kiem tra lai...
"C:\Users\ADMIN\python.exe" --version
if %errorlevel% neq 0 (
    echo [LOI] Van khong chay duoc. Hay thu cai voi TargetDir khac.
    pause
    exit /b 1
)

:run_server
echo.
echo === Khoi dong server ===
set PYTHONHOME=F:\DevProfileData
set PATH=C:\Users\ADMIN;F:\DevProfileData;F:\DevProfileData\DLLs;%PATH%
"C:\Users\ADMIN\python.exe" app.py
pause
