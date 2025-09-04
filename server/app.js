const express = require('express');
const connectDB = require('../Backend_stuff/database/db.js');
require('dotenv').config();

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use(express.static('public'));

// Routes
const translateRoutes = require('../Backend_stuff/routes/translateRoute.js');
const authRoutes = require('../Backend_stuff/routes/authRoute.js');
app.use('/translate', translateRoutes);
app.use('/auth', authRoutes);

// Serve index.html at root
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', database: 'Connected' });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(` Server running on port ${PORT}`));
