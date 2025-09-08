// db.js
const mongoose = require('mongoose');
require('dotenv').config();

const uri = process.env.MONGODB_URI || 'mongodb+srv://Zoe:kaas@mainsign.7aqed10.mongodb.net/?tls=true';

const connectDB = async () => {
  try {
    console.log('Attempting to connect to MongoDB...');
    console.log(`Using MongoDB URI: ${process.env.MONGODB_URI ? 'from environment variable' : 'hardcoded URI'}`);
    await mongoose.connect(uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log(' MongoDB connected successfully with Mongoose');
  } catch (err) {
    console.error(' MongoDB connection error:', err.message);
    console.error('Please check your MongoDB URI and network connectivity.');
    process.exit(1);
  }
};

module.exports = connectDB;
