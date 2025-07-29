const signToText = require('../services/signToTextService');
const textToSpeech = require('../services/textToSpeechService');
const Translation = require('../database/translationModel');

exports.handleTranslation = async (req, res) => 
{
  try 
  {
    const text = await signToText('Backend_stuff\services\signToTextService.js');
    const speechUrl = await textToSpeech(text);

    await Translation.create(
    {
      userId: req.body.userId,
      originalVideo: req.file.filename,
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
