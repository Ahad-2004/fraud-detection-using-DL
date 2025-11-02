@echo off
echo Installing required Python packages...
pip install -r requirements.txt

echo Starting Fraud Detection Server...
python -m uvicorn main:app --reload

pause
