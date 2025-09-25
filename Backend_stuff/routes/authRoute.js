const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const authenticateToken = require('../middlewares/auth');

// Sign up route
router.post('/signup', authController.signup);

// Sign in route
router.post('/signin', authController.signin);

// Forgot password route
router.post('/forgot-password', authController.forgotPassword);

// Reset password route
router.post('/reset-password', authController.resetPassword);

// Change password route
router.post('/change-password', authenticateToken, authController.changePassword);

// Change email route
router.post('/change-email', authenticateToken, authController.changeEmail);

module.exports = router;
