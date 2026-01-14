/**
 * ================================================================================
 * MarketSage Node.js Server - Main Entry Point
 * ================================================================================
 * 
 * This Express.js server handles authentication, user management, admin operations,
 * settings, reports, and user reviews. It runs alongside the Python Prediction API.
 * 
 * Architecture:
 * - Frontend (React)     → http://localhost:8080
 * - Node.js Server       → http://localhost:3001 (this file)
 * - Python Prediction API → http://localhost:5000
 * 
 * Available API Routes:
 * - /api/auth     - User authentication (login, register, profile)
 * - /api/admin    - Admin operations (user management)
 * - /api/settings - User settings / preferences
 * - /api/reports  - Financial reports management
 * - /api/reviews  - User reviews and ratings
 * - /api/health   - Server health check
 * 
 * Author: MarketSage Team
 * ================================================================================
 */

// =============================================================================
// IMPORTS
// =============================================================================

// Express.js - Fast, minimalist web framework for Node.js
const express = require('express');

// CORS (Cross-Origin Resource Sharing) - Allows frontend to access this API
const cors = require('cors');

// Route modules - Each handles a specific domain of functionality
const authRoutes = require('./routes/auth');       // Authentication endpoints
const adminRoutes = require('./routes/admin');     // Admin management endpoints
const settingsRoutes = require('./routes/settings'); // User settings endpoints
const reviewRoutes = require('./routes/reviews');   // Reviews endpoints

// =============================================================================
// APPLICATION SETUP
// =============================================================================

// Create Express application instance
const app = express();

// Port configuration - Uses environment variable or defaults to 3001
const PORT = process.env.PORT || 3001;

// =============================================================================
// MIDDLEWARE CONFIGURATION
// =============================================================================

/**
 * CORS Middleware
 * Allows the frontend application to make requests to this API.
 * Without CORS, browser security would block cross-origin requests.
 */
app.use(cors({
    // Allowed origins (frontend URLs)
    origin: [
        'http://localhost:5173',  // Vite dev server default
        'http://localhost:8080',  // Alternative dev server port
        'http://127.0.0.1:5173'   // localhost alias
    ],
    credentials: true  // Allow cookies to be sent with requests
}));

/**
 * JSON Body Parser
 * Automatically parses incoming JSON request bodies
 * Makes req.body available in route handlers
 */
app.use(express.json());

/**
 * Request Logging Middleware
 * Logs every incoming request with timestamp, method, and path
 * Useful for debugging and monitoring
 */
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
    next();  // Pass control to next middleware
});

// =============================================================================
// ROUTE REGISTRATION
// =============================================================================

// Authentication routes: /api/auth/login, /api/auth/register, /api/auth/profile
app.use('/api/auth', authRoutes);

// Admin routes: /api/admin/users, /api/admin/analytics
app.use('/api/admin', adminRoutes);

// Settings routes: /api/settings (GET, POST)
app.use('/api/settings', settingsRoutes);

// Reports routes: /api/reports (list, download, upload)
app.use('/api/reports', require('./routes/reports'));

// Reviews routes: /api/reviews (GET all, POST new)
app.use('/api/reviews', reviewRoutes);

/**
 * Health Check Endpoint
 * Used for monitoring and verifying server is running
 * Returns simple JSON with status and timestamp
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString()
    });
});

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * Global Error Handler
 * Catches any unhandled errors from route handlers
 * Returns 500 Internal Server Error with safe error message
 * 
 * Note: This is a 4-parameter middleware (err, req, res, next),
 * which Express recognizes as an error handler
 */
app.use((err, req, res, next) => {
    console.error('Error:', err.message);
    res.status(500).json({ error: 'Internal server error' });
});

// =============================================================================
// SERVER STARTUP
// =============================================================================

/**
 * Start the Express server
 * Listens on configured PORT for incoming HTTP requests
 */
app.listen(PORT, () => {
    console.log(`\n🚀 MarketSage Server running on http://localhost:${PORT}`);
    console.log(`   Health check: http://localhost:${PORT}/api/health\n`);
});
