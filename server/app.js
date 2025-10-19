const express = require('express');
const path = require('path');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { spawn } = require('child_process');
const crypto = require('crypto');
const connectDB = require('../Backend_stuff/database/db.js');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();

console.log('__dirname:', __dirname);

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors()); // Enable CORS for all routes
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// Authentication middleware
const authenticateToken = (req, res, next) => {
    let token = req.headers['authorization'] && req.headers['authorization'].split(' ')[1];
    if (!token) {
        token = req.cookies && req.cookies.token;
    }

    console.log('Token found:', !!token);
    console.log('Headers:', req.headers);
    console.log('Cookies:', req.cookies);

    if (!token) {
        console.log('No token, redirecting to /');
        return res.redirect('/');
    }

    jwt.verify(token, process.env.JWT_SECRET || 'secret', (err, user) => {
        if (err) {
            console.log('Token verification failed:', err.message);
            return res.redirect('/');
        }
        req.user = user;
        next();
    });
};

// Set EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../client/views'));

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '../public')));

// Proxy Flask AI endpoints to avoid CORS issues
const flaskProxyOptions = {
    target: 'https://flaskssai.belgiumcampus.ac.za',//flaskssai.belgiumcampus.ac.za
    changeOrigin: true,
    onError: (err, req, res) => {
        console.error('Flask proxy error:', err.message);
        res.status(503).json({
            status: 'error',
            message: 'AI service unavailable. Please ensure Flask server is running on port 5000.'
        });
    },
    onProxyReq: (proxyReq, req, res) => {
        console.log(`Proxying ${req.method} ${req.url} to Flask`);

        // Add user-specific headers for personalization
        if (req.user && req.user.userId) {
            proxyReq.setHeader('X-User-ID', req.user.userId);
            proxyReq.setHeader('X-User-Token', req.cookies.token || req.headers.authorization?.split(' ')[1]);
        }
    }
};

app.use('/ai', createProxyMiddleware({
    ...flaskProxyOptions,
    pathRewrite: {
        '^/ai': '', 
    },
}));

// Health check for Flask service
app.get('/ai/health', async (req, res) => {
    try {
        const response = await fetch('https://flaskssai.belgiumcampus.ac.za/status');
        if (response.ok) {
            res.json({ status: 'healthy', message: 'Flask AI service is running' });
        } else {
            throw new Error(`Flask responded with status: ${response.status}`);
        }
    } catch (error) {
        res.status(503).json({ 
            status: 'unhealthy', 
            message: 'Flask AI service is not responding. Please start the Flask server.' 
        });
    }
});

// Routes - Import route modules
const translateRoutes = require('../Backend_stuff/routes/translateRoute.js');
const authRoutes = require('../Backend_stuff/routes/authRoute.js');
app.use('/translate', translateRoutes);
app.use('/auth', authRoutes);

// Main application routes
app.get('/', (req, res) => { // This is the login page 
    res.render('index', { currentPath: '/' });
});

// Application routes
app.get('/camera', authenticateToken, async (req, res) => {
    // Check if Flask AI service is running
    try {
        const healthCheck = await fetch('https://flaskssai.belgiumcampus.ac.za/status');
        if (healthCheck.ok) {
            res.render('camera', {
                aiServiceStatus: 'available',
                flaskUrl: 'https://flaskssai.belgiumcampus.ac.za',
                currentPath: '/camera',
                userId: req.user.userId, // Pass user ID for personalization
                userToken: req.cookies.token // Pass token for WebSocket auth
            });
        } else {
            throw new Error('Flask service not responding');
        }
    } catch (error) {
        console.warn('Flask AI service not available:', error.message);
        res.render('camera', {
            aiServiceStatus: 'unavailable',
            flaskUrl: 'https://flaskssai.belgiumcampus.ac.za',
            errorMessage: 'AI service is not available. Please start the Flask server on port 5000.',
            currentPath: '/camera',
            userId: req.user.userId,
            userToken: req.cookies.token
        });
    }
});

app.get('/reset-password', (req, res) => {
    const token = req.query.token;
    res.render('reset-password', { token });
});

app.get('/storage', authenticateToken, (req, res) => {
    res.render('storage', { currentPath: '/storage' });
});

app.get('/settings', authenticateToken, (req, res) => {
  res.render('settings', { currentPath: '/settings' });
});

app.get('/application_settings', authenticateToken, (req, res) => {
  res.render('application_settings');
});

// Legal / Documents routes
app.get('/privacy-policy', authenticateToken, (req, res) => {
        res.render('privacy_policy', { currentPath: '/privacy-policy' });
});

app.get('/terms-and-conditions', authenticateToken, (req, res) => {
        res.render('terms_conditions', { currentPath: '/terms-and-conditions' });
});

// API endpoint to check Flask service status
app.get('/api/flask-status', async (req, res) => {
    try {
        const response = await fetch('https://flaskssai.belgiumcampus.ac.za/status');
        if (response.ok) {
            const data = await response.json();
            res.json({ 
                available: true, 
                status: 'connected',
                flaskData: data
            });
        } else {
            throw new Error(`Flask responded with status: ${response.status}`);
        }
    } catch (error) {
        res.json({ 
            available: false, 
            status: 'disconnected',
            error: error.message
        });
    }
});

// Health check endpoint for the main server
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        database: 'Connected',
        server: 'Express with EJS',
        aiService: 'Proxied to Flask on port 5000'
    });
});

// ------------------------------------------------------------------
// Text-to-Speech endpoint (uses Python tts-transcript.py)
// POST /tts { text: "Hello" }
// Returns audio/wav binary.
// ------------------------------------------------------------------
app.post('/tts', authenticateToken, async (req, res) => {
    try {
        const text = (req.body && req.body.text || '').trim();
        if (!text) {
            return res.status(400).json({ ok: false, error: 'Missing text' });
        }

        // Create a unique temp filename each request 
        const fs = require('fs');
        const tmpDir = path.join(__dirname, 'tts_tmp');
        fs.mkdirSync(tmpDir, { recursive: true });
        const unique = Date.now().toString(36) + '_' + crypto.randomBytes(4).toString('hex');
        const outPath = path.join(tmpDir, `tts_${unique}.wav`);

        const pythonExe = process.env.PYTHON || 'python';
        const scriptPath = path.join(__dirname, 'tts-transcript.py');
        const args = [scriptPath, '--text', text, '--out', outPath];

        const py = spawn(pythonExe, args, { stdio: ['ignore','pipe','pipe'] });
        let stdout = '';
        let stderr = '';
        py.stdout.on('data', d => { stdout += d.toString(); });
        py.stderr.on('data', d => { stderr += d.toString(); });
        py.on('error', err => console.error('TTS spawn error:', err));
        py.on('close', code => {
            const cleanup = () => {
                // Delete temp file asynchronously 
                fs.unlink(outPath, () => {});
            };
            if (code !== 0) {
                console.error('TTS failed:', code, stderr, stdout);
                let parsed;
                try { parsed = JSON.parse(stdout.trim()); } catch {}
                cleanup();
                return res.status(500).json({ ok: false, error: parsed?.error || 'TTS process failed', code, stderr });
            }
            if (!fs.existsSync(outPath)) {
                cleanup();
                return res.status(500).json({ ok: false, error: 'Output file missing after TTS' });
            }
            res.setHeader('Content-Type', 'audio/wav');
            // Stream and then delete file at end
            const stream = fs.createReadStream(outPath);
            stream.on('close', cleanup);
            stream.on('error', err => {
                console.error('Stream error:', err);
                cleanup();
            });
            stream.pipe(res);
        });
    } catch (e) {
        console.error('TTS route error:', e);
        res.status(500).json({ ok: false, error: e.message });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Application error:', err);
    res.status(500).json({ 
        status: 'error', 
        message: 'Internal server error' 
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).render('404', { 
        message: 'Page not found',
        currentPath: req.path
    });
});

// Start the server
const PORT = 8001; // The Express server port

app.listen(PORT, () => {
    console.log(`Express server running on http://localhost:${PORT}`);
    console.log('Flask AI service proxied to http://flaskssai.belgiumcampus.ac.za');
    console.log('Database: MongoDB connected');
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err.message);
    // Optionally, close server & exit process here
});

module.exports = app;
