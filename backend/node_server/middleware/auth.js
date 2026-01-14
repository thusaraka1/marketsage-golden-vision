const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'marketsage-secret-key-change-in-production';

// Verify JWT token middleware
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) {
            return res.status(403).json({ error: 'Invalid or expired token' });
        }
        req.user = decoded;
        next();
    });
}

// Generate JWT token
function generateToken(user) {
    return jwt.sign(
        { id: user.id, email: user.email, name: user.name },
        JWT_SECRET,
        { expiresIn: '7d' }
    );
}

module.exports = { authenticateToken, generateToken, JWT_SECRET };
