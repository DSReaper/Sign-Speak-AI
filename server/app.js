const express = require('express');
const cors = require('cors');
const connectDB = require('../Backend_stuff/database/db.js');
require('dotenv').config();


const app = express();

console.log('__dirname:', __dirname);


// Connect to MongoDB
connectDB();

// Middleware
app.use(cors()); // Enable CORS for all routes
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
const path = require('path');

app.get('/', (req, res) => {
  res.sendFile('index.html', { root: path.resolve(__dirname, '..') });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', database: 'Connected' });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(` Server running on port ${PORT}`));

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('Unhandled Rejection:', err.message);
  // Optionally, close server & exit process here
});
