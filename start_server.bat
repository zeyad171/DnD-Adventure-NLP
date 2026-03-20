@echo off
echo ============================================
echo Starting D and D Adventure Backend Server
echo ============================================
echo.
echo Make sure you have:
echo 1. Installed dependencies: pip install -r requirements.txt
echo 2. Set GEMINI_API_KEY in .env file
echo.
echo Starting Flask server on http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.
python app.py
pause
