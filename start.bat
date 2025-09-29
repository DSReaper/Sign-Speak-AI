@echo off
echo Starting Flask and Node (development mode) in separate terminals...

REM Start Flask backend on port 5000
start "Flask Server" cmd /k python server/app.py

REM Start Node backend with nodemon on port 8001
start "Node Server" cmd /k nodemon server/app.js
