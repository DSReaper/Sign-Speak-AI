/**
 * Activation file to run the Flask app.py located at Backend_stuff/middlewares/web-app/app.py
 * This script spawns a child process to run the Flask application.
 */

const { spawn } = require('child_process');
const path = require('path');

// Path to the Flask app.py file
const appPath = path.join(__dirname, 'Backend_stuff', 'middlewares', 'web-app', 'app.py');

// Spawn a child process to run the Flask app
const flaskProcess = spawn('python', [appPath], {
  stdio: 'inherit',
  shell: true
});

flaskProcess.on('error', (err) => {
  console.error('Failed to start Flask app:', err);
});

flaskProcess.on('exit', (code, signal) => {
  if (code !== null) {
    console.log(`Flask app exited with code ${code}`);
  } else if (signal !== null) {
    console.log(`Flask app was killed with signal ${signal}`);
  } else {
    console.log('Flask app exited');
  }
});
