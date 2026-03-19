@echo off
cd /d "%~dp0\DL"
echo Starting FastAPI Backend...
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
