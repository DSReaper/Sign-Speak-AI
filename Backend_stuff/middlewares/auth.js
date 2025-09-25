const jwt = require('jsonwebtoken');

const authenticateToken = (req, res, next) => {
    let token = req.headers['authorization'] && req.headers['authorization'].split(' ')[1];
    if (!token) {
        token = req.cookies && req.cookies.token;
    }

    console.log('Token found:', !!token);
    console.log('Headers:', req.headers);
    console.log('Cookies:', req.cookies);

    if (!token) {
        console.log('No token, redirecting to /');
        return res.redirect('/');
    }

    jwt.verify(token, process.env.JWT_SECRET || 'secret', (err, user) => {
        if (err) {
            console.log('Token verification failed:', err.message);
            return res.redirect('/');
        }
        req.user = user;
        next();
    });
};

module.exports = authenticateToken;
