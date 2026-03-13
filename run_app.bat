@echo off
cd /d "%~dp0"

echo Starting AI Career Guidance...
echo.

REM Clear SSL keylog if set (can cause permission errors)
set SSLKEYLOGFILE=

REM Start backend in background
echo Starting backend on http://127.0.0.1:8000
start "Backend" cmd /k "cd /d %~dp0 && uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

REM Wait for backend to start
timeout /t 3 /nobreak > nul

REM Start Streamlit and open browser
echo Starting Streamlit on http://127.0.0.1:8501
echo.
echo Opening browser...
start http://127.0.0.1:8501

streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless false
