const express = require('express');
const path = require('path');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const { createProxyMiddleware } = require('http-proxy-middleware');
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
    target: 'http://localhost:5000',
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
        const response = await fetch('http://localhost:5000/status');
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
    res.render('index');
});

// Application routes
app.get('/camera', authenticateToken, async (req, res) => {
    // Check if Flask AI service is running
    try {
        const healthCheck = await fetch('http://localhost:5000/status');
        if (healthCheck.ok) {
            res.render('camera', { 
                aiServiceStatus: 'available',
                flaskUrl: 'http://localhost:5000'
            });
        } else {
            throw new Error('Flask service not responding');
        }
    } catch (error) {
        console.warn('Flask AI service not available:', error.message);
        res.render('camera', { 
            aiServiceStatus: 'unavailable',
            flaskUrl: 'http://localhost:5000',
            errorMessage: 'AI service is not available. Please start the Flask server on port 5000.'
        });
    }
});

app.get('/reset-password', (req, res) => {
    const token = req.query.token;
    res.render('reset-password', { token });
});

app.get('/storage', (req, res) => {
    res.render('storage');
});

app.get('/settings', (req, res) => {
  res.render('settings');
});

// API endpoint to check Flask service status
app.get('/api/flask-status', async (req, res) => {
    try {
        const response = await fetch('http://localhost:5000/status');
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
        message: 'Page not found' 
    });
});

// Start the server
const PORT = 8001; // The Express server port

app.listen(PORT, () => {
    console.log(`Express server running on http://localhost:${PORT}`);
    console.log('Flask AI service proxied to http://localhost:5000');
    console.log('Database: MongoDB connected');
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err.message);
    // Optionally, close server & exit process here
});

module.exports = app;
