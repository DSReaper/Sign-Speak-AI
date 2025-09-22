@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Installing Node.js dependencies...
npm install

echo Installing nodemon for development...
npm install --save-dev nodemon

echo All dependencies installed successfully.
pause
