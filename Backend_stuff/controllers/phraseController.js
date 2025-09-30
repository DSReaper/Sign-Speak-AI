const Phrase = require('../database/phraseModel');
const User = require('./authController').User; // Import User model from authController

// Save a phrase for the logged-in user
exports.savePhrase = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { text } = req.body;

    if (!text || text.trim() === '') {
      return res.status(400).json({ message: 'Phrase text is required' });
    }

    // Check if user exists
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Find phrase document for user or create new
    let phraseDoc = await Phrase.findOne({ userId });
    if (!phraseDoc) {
      phraseDoc = new Phrase({ userId, phrases: [] });
    }

    phraseDoc.phrases.push({ text: text.trim(), timestamp: new Date() });
    await phraseDoc.save();

    res.status(200).json({ message: 'Phrase saved successfully' });
  } catch (error) {
    console.error('Error in savePhrase:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get stored phrases for the logged-in user
exports.getPhrases = async (req, res) => {
  try {
    const userId = req.user.userId;

    // Check if user exists
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const phraseDoc = await Phrase.findOne({ userId });
    if (!phraseDoc) {
      return res.status(200).json({ phrases: [] });
    }

    res.status(200).json({ phrases: phraseDoc.phrases });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};
