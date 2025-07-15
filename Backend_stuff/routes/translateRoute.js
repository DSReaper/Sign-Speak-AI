const express = require('express');
const router = express.Router();
const controller = require('../controllers/translateController.js');
const upload = require('../middlewares/upload.js');

router.post('/upload', upload.single('video'), controller.handleTranslation);
router.get('/history/:userId', controller.getHistory);

module.exports = router;