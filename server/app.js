const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

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

// Flask AI service proxy routes
app.use('/ai/start_camera', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/start_camera': '/start_camera' }
}));

app.use('/ai/stop_camera', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/stop_camera': '/stop_camera' }
}));

app.use('/ai/video_feed', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/video_feed': '/video_feed' }
}));

app.use('/ai/status', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/status': '/status' }
}));

app.use('/ai/toggle_hands', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/toggle_hands': '/toggle_hands' }
}));

app.use('/ai/reset_detector', createProxyMiddleware({ 
    ...flaskProxyOptions, 
    pathRewrite: { '^/ai/reset_detector': '/reset_detector' }
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

// Route to render index.ejs
app.get('/', (req, res) => {
    res.render('index');
});

// Login post route
app.post('/login', (req, res) => {
    const { email, password } = req.body;

    if (email === 'test@example.com' && password === 'password123') {
        res.redirect('/camera');
    } else {
        res.status(401).send('Invalid credentials');
    }
});

// Application routes
app.get('/camera', async (req, res) => {
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

app.get('/storage', (req, res) => {
    res.render('storage');
});

app.get('/profile', (req, res) => {
    res.render('profile');
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
const port = process.env.PORT || 8000;

app.listen(port, () => {
    console.log(`Node.js server running on http://localhost:${port}`);
    console.log('Flask AI service proxied to http://localhost:5000');
    console.log('Make sure Flask server is running on port 5000 for AI features');
});

module.exports = app;