@echo off
REM Kinetra MT5 Bridge Quick Setup for Windows
REM Run this from CMD or double-click

echo ========================================
echo   Kinetra MT5 Bridge Quick Setup
echo ========================================
echo.

REM Find Python
echo Looking for Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH!
    echo Checking common locations...

    if exist "%USERPROFILE%\Anaconda3\python.exe" (
        set PYTHON=%USERPROFILE%\Anaconda3\python.exe
        echo Found Anaconda: %PYTHON%
    ) else if exist "%USERPROFILE%\miniconda3\python.exe" (
        set PYTHON=%USERPROFILE%\miniconda3\python.exe
        echo Found Miniconda: %PYTHON%
    ) else if exist "C:\Anaconda3\python.exe" (
        set PYTHON=C:\Anaconda3\python.exe
        echo Found: %PYTHON%
    ) else (
        echo No Python found! Install Anaconda or Python first.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=*" %%i in ('where python') do set PYTHON=%%i
    echo Found: %PYTHON%
)

echo.
echo Installing MetaTrader5 package...
"%PYTHON%" -m pip install --upgrade MetaTrader5

echo.
echo Testing MT5 connection...
"%PYTHON%" -c "import MetaTrader5 as mt5; print('MT5 OK' if mt5.initialize() else 'MT5 FAIL - is terminal running?'); mt5.shutdown()"

echo.
echo Creating bridge server...
echo Copy the bridge server from: scripts/mt5_bridge_server.py
echo Or download from your WSL2 Kinetra folder

echo.
echo ========================================
echo   Next Steps:
echo ========================================
echo 1. Copy mt5_bridge_server.py to this Windows machine
echo 2. Run: python mt5_bridge_server.py
echo 3. In WSL2: python -c "from kinetra import MT5Bridge; b=MT5Bridge(mode='bridge')"
echo.

pause
