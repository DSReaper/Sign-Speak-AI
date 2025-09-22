@echo off
echo Stopping Flask and Node servers...

REM Kill Flask (Python) processes
taskkill /F /IM python.exe >nul 2>&1

REM Kill Node.js processes
taskkill /F /IM node.exe >nul 2>&1

echo All servers stopped.
pause
