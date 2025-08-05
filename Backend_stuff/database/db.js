// db.js
const mongodb = require('mongodb');
require('dotenv').config();

const uri = process.env.MONGODB_URI || 'mongodb+srv://Zoe:kaas@mainsign.7aqed10.mongodb.net/?tls=true';

const connectDB = async () => {
  try {
    await mongodb.connect(uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log(' MongoDB connected successfully');
  } catch (err) {
    console.error(' MongoDB connection error:', err);
    process.exit(1);
  }
};

module.exports = connectDB;
