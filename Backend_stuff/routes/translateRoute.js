const express = require('express');
const router = express.Router();
const controller = require('../controllers/translateController.js');

// Removed upload middleware as no file upload is needed
router.post('/upload', controller.handleTranslation);
router.get('/history/:userId', controller.getHistory);

module.exports = router;
