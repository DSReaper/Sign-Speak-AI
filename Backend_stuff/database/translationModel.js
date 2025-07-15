const mongoose = require('mongoose');
const schema = new mongoose.Schema(
{
  userId: String,
  originalVideo: String,
  text: String,
  speechUrl: String,
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Translation', schema);
