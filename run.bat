@echo off
REM Starts PhishDrift and opens http://localhost:8000 in your browser. No Streamlit needed.
cd /d "%~dp0"
python server.py
pause
