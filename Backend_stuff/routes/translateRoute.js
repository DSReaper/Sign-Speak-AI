const express = require('express');
const router = express.Router();
const controller = require('../controllers/translateController.js');
const phraseController = require('../controllers/phraseController.js');
const authenticateToken = require('../middlewares/auth');

// Removed upload middleware as no file upload is needed
router.post('/upload', controller.handleTranslation);
router.get('/history/:userId', controller.getHistory);

// Phrase routes
router.post('/phrase', authenticateToken, phraseController.savePhrase);
router.get('/phrases', authenticateToken, phraseController.getPhrases);

module.exports = router;
