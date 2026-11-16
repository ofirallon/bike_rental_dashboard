@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The local Python environment is missing.
    echo Follow the Installation and run section in README.md first.
    pause
    exit /b 1
)

set "PYTHONDONTWRITEBYTECODE=1"
echo Starting Bike Rental Dashboard...
echo App address: http://127.0.0.1:8501
echo Keep this window open while using the app. Press Ctrl+C to stop it.
echo.

".venv\Scripts\python.exe" -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.serverAddress 127.0.0.1 --browser.gatherUsageStats false
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo The server could not keep running. Read the error above.
    echo If port 8501 is already in use, open http://127.0.0.1:8501 first.
    pause
)

exit /b %APP_EXIT_CODE%
