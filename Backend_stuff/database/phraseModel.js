const mongoose = require('mongoose');

const phraseSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, unique: true },
  phrases: [
    {
      text: { type: String, required: true },
      timestamp: { type: Date, default: Date.now }
    }
  ]
});

const Phrase = mongoose.model('Phrase', phraseSchema);

module.exports = Phrase;
