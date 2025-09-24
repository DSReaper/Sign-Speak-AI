const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');

// Sign up route
router.post('/signup', authController.signup);

// Sign in route
router.post('/signin', authController.signin);

// Forgot password route
router.post('/forgot-password', authController.forgotPassword);

// Reset password route
router.post('/reset-password', authController.resetPassword);

module.exports = router;
