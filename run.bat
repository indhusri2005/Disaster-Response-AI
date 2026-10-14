@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
if not exist "backend\.env" copy "backend\.env.example" "backend\.env"
echo.
echo Backend starting at http://127.0.0.1:8000
echo Open frontend\index.html in your browser.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
