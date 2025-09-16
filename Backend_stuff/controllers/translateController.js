const signToText = require('../services/signToTextService');
const textToSpeech = require('../services/textToSpeechService');
const Translation = require('../database/translationModel');

exports.handleTranslation = async (req, res) =>
{
  try
  {
    // Assuming text is provided in request body or some other way
    const text = req.body.text || 'Sample translated text';
    const speechUrl = await textToSpeech(text);

    await Translation.create(
    {
      userId: req.body.userId,
      originalVideo: 'no video', // Since no file upload
      text,
      speechUrl,
    });

    res.status(200).json({ text, speechUrl });
  }
  catch (err)
  {
    res.status(500).json({ error: err.message });
  }
};

exports.getHistory = async (req, res) => {
  try {
    const { userId } = req.params;
    const history = await Translation.find({ userId });
    res.status(200).json({ history });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
