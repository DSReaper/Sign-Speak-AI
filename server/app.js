const express = require('express');
const connectDB = require('../Backend_stuff/database/db.js');
require('dotenv').config();

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
const translateRoutes = require('../Backend_stuff/routes/translateRoute.js');
app.use('/translate', translateRoutes);

// Basic route for testing
app.get('/', (req, res) => {
  res.json({ message: 'Sign-Speak-AI API is running!' });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', database: 'Connected' });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(` Server running on port ${PORT}`));
