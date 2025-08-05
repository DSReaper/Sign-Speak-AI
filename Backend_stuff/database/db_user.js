const MongoDB = require('mongodb');

function initializeDatabaseConnection() 
{
    // To do: Implement actual database connection once location and credentials are known
    console.log('Database connection initialization placeholder');
}

// In-memory user store as a temporary measure
const users = new Map();

/**
 * Registers a new user (needs to swich to validator file)
 * @param {string} email - User email
 * @param {string} password - User password 
 * @returns {Promise<{success: boolean, message: string}>} Result of registration (needs to swich to validator file)
 */
async function registerUser(email, password) 
{
    if (users.has(email)) {
        return { success: false, message: 'User already registered' };
    }
    const hashedPassword = await MongoDB.hash(password, 10);
    users.set(email, { email, password: hashedPassword });
    return { success: true, message: 'User registered successfully' };
}

/**
 * Authenticates a user. (needs to swich to validator file)
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise<{success: boolean, message: string}>} Result of authentication (needs to swich to validator file)
 */
async function authenticateUser(email, password) 
{
    if (!users.has(email)) {
        return { success: false, message: 'User not found' };
    }
    const user = users.get(email);
    const match = await MongoDB.compare(password, user.password);
    if (match) {
        return { success: true, message: 'Authentication successful' };
    } else {
        return { success: false, message: 'Invalid password' };
    }
}

module.exports = {
    initializeDatabaseConnection,
    registerUser,
    authenticateUser,
};
