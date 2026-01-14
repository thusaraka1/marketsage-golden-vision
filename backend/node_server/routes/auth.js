/**
 * ================================================================================
 * Authentication Router
 * ================================================================================
 * 
 * Handles user authentication operations including registration, login, and
 * account upgrades. Uses bcrypt for password hashing and JWT for session tokens.
 * 
 * Endpoints:
 * - POST /api/auth/register - Create new user account
 * - POST /api/auth/login    - Authenticate and get JWT token
 * - POST /api/auth/upgrade  - Upgrade user to Pro plan
 * 
 * Security:
 * - Passwords are hashed using bcrypt (10 salt rounds)
 * - JWT tokens are issued upon successful authentication
 * - SQL injection prevented via parameterized queries
 * 
 * Database: SQLite (users table)
 * 
 * Author: MarketSage Team
 * ================================================================================
 */

// =============================================================================
// IMPORTS
// =============================================================================

// Express Router for modular route handling
const express = require('express');

// bcryptjs - Password hashing library (pure JS implementation)
const bcrypt = require('bcryptjs');

// Database connection
const db = require('../db/database');

// JWT token generation utility
const { generateToken } = require('../middleware/auth');

// Create router instance
const router = express.Router();

// =============================================================================
// POST /api/auth/register - User Registration
// =============================================================================

/**
 * Register a new user account
 * 
 * Request Body:
 * {
 *   name: string (required) - User's display name
 *   email: string (required) - User's email address
 *   password: string (required) - Password (min 6 chars)
 * }
 * 
 * Response:
 * - 201 Created: { message, user, token }
 * - 400 Bad Request: Validation errors
 * - 409 Conflict: Email already registered
 * - 500 Error: Server error
 */
router.post('/register', async (req, res) => {
    try {
        // Extract registration data from request body
        const { name, email, password } = req.body;

        // -------------------------------------------------------------------------
        // Input Validation
        // -------------------------------------------------------------------------

        // Check all required fields are present
        if (!name || !email || !password) {
            return res.status(400).json({ error: 'Name, email, and password are required' });
        }

        // Enforce minimum password length for security
        if (password.length < 6) {
            return res.status(400).json({ error: 'Password must be at least 6 characters' });
        }

        // -------------------------------------------------------------------------
        // Check for Existing User
        // -------------------------------------------------------------------------

        // Query database to check if email is already registered
        const existingUser = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
        if (existingUser) {
            return res.status(409).json({ error: 'Email already registered' });
        }

        // -------------------------------------------------------------------------
        // Password Hashing
        // -------------------------------------------------------------------------

        // Hash password using bcrypt with 10 salt rounds
        // Salt rounds determine computational cost (higher = more secure but slower)
        const saltRounds = 10;
        const passwordHash = await bcrypt.hash(password, saltRounds);

        // -------------------------------------------------------------------------
        // Create User in Database
        // -------------------------------------------------------------------------

        // Insert new user with default values:
        // - plan: 'Free' (can upgrade to Premium)
        // - status: 'Active' (can be Suspended by admin)
        // - is_pro: 0 (false, not a pro user)
        const stmt = db.prepare(`
            INSERT INTO users (name, email, password_hash, plan, status, is_pro)
            VALUES (?, ?, ?, 'Free', 'Active', 0)
        `);
        const result = stmt.run(name, email, passwordHash);

        // Retrieve the newly created user (to get auto-generated fields)
        const user = db.prepare(
            'SELECT id, name, email, plan, status, is_pro, created_at FROM users WHERE id = ?'
        ).get(result.lastInsertRowid);

        // -------------------------------------------------------------------------
        // Generate JWT Token for Automatic Login
        // -------------------------------------------------------------------------

        // User is automatically logged in after registration
        const token = generateToken(user);

        // Log successful registration
        console.log(`✓ New user registered: ${email}`);

        // Return success response with user data and token
        res.status(201).json({
            message: 'User created successfully',
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                isPro: Boolean(user.is_pro),  // Convert 0/1 to boolean
                plan: user.plan
            },
            token  // JWT token for future authenticated requests
        });

    } catch (error) {
        console.error('Registration error:', error);
        res.status(500).json({ error: 'Failed to create user' });
    }
});

// =============================================================================
// POST /api/auth/login - User Login
// =============================================================================

/**
 * Authenticate user and issue JWT token
 * 
 * Request Body:
 * {
 *   email: string (required) - User's email address OR Username
 *   password: string (required) - User's password
 * }
 * 
 * Response:
 * - 200 OK: { message, user, token }
 * - 400 Bad Request: Missing email or password
 * - 401 Unauthorized: Invalid credentials
 * - 403 Forbidden: Account suspended
 * - 500 Error: Server error
 */
router.post('/login', async (req, res) => {
    try {
        // Extract login credentials (email field can be username for admin)
        const { email, password } = req.body;

        // -------------------------------------------------------------------------
        // Input Validation
        // -------------------------------------------------------------------------
        if (!email || !password) {
            return res.status(400).json({ error: 'Username/Email and password are required' });
        }

        // -------------------------------------------------------------------------
        // Check for Super Admin (admins table)
        // -------------------------------------------------------------------------
        const adminUser = db.prepare('SELECT * FROM admins WHERE username = ?').get(email);

        if (adminUser) {
            const validPassword = await bcrypt.compare(password, adminUser.password_hash);
            if (validPassword) {
                const token = generateToken({ ...adminUser, role: 'admin' });
                console.log(`✓ Super Admin logged in: ${email}`);
                return res.json({
                    message: 'Admin Login successful',
                    user: {
                        id: adminUser.id,
                        name: 'Super Admin',
                        email: 'admin@marketsage.com', // Placeholder
                        username: adminUser.username,
                        isAdmin: true,
                        isPro: true,
                        plan: 'Enterprise'
                    },
                    token
                });
            }
        }

        // -------------------------------------------------------------------------
        // Find User by Email (Standard Users)
        // -------------------------------------------------------------------------
        const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email);

        // Return generic error (don't reveal if email exists for security)
        if (!user) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        // -------------------------------------------------------------------------
        // Verify Password
        // -------------------------------------------------------------------------

        // Compare provided password with stored hash
        const validPassword = await bcrypt.compare(password, user.password_hash);
        if (!validPassword) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        // -------------------------------------------------------------------------
        // Check Account Status
        // -------------------------------------------------------------------------

        // Prevent suspended users from logging in
        if (user.status === 'Suspended') {
            return res.status(403).json({ error: 'Account is suspended' });
        }

        // -------------------------------------------------------------------------
        // Generate JWT Token
        // -------------------------------------------------------------------------
        const token = generateToken(user);

        // Log successful login
        console.log(`✓ User logged in: ${email}`);

        // Return success response
        res.json({
            message: 'Login successful',
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                isPro: Boolean(user.is_pro),
                plan: user.plan
            },
            token
        });

    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ error: 'Login failed' });
    }
});

// =============================================================================
// POST /api/auth/upgrade - Upgrade to Pro
// =============================================================================

/**
 * Upgrade user account to Pro/Premium plan
 * 
 * Note: In a production app, this would integrate with a payment provider
 * (e.g., Stripe) to process payment before upgrading.
 * 
 * Request Body:
 * {
 *   userId: number (required) - ID of user to upgrade
 * }
 * 
 * Response:
 * - 200 OK: { message, user }
 * - 400 Bad Request: Missing userId
 * - 500 Error: Server error
 */
router.post('/upgrade', async (req, res) => {
    try {
        const { userId } = req.body;

        // Validate userId is provided
        if (!userId) {
            return res.status(400).json({ error: 'User ID required' });
        }

        // -------------------------------------------------------------------------
        // Update User to Pro
        // -------------------------------------------------------------------------

        // Set is_pro flag to 1 (true) and plan to 'Premium'
        const stmt = db.prepare('UPDATE users SET is_pro = 1, plan = ? WHERE id = ?');
        stmt.run('Premium', userId);

        // Retrieve updated user data
        const user = db.prepare(
            'SELECT id, name, email, plan, is_pro FROM users WHERE id = ?'
        ).get(userId);

        // Log upgrade
        console.log(`✓ User upgraded to Pro: ${user.email}`);

        // Return success response
        res.json({
            message: 'Upgraded to Pro successfully',
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                isPro: Boolean(user.is_pro),
                plan: user.plan
            }
        });

    } catch (error) {
        console.error('Upgrade error:', error);
        res.status(500).json({ error: 'Upgrade failed' });
    }
});

// =============================================================================
// EXPORT ROUTER
// =============================================================================

module.exports = router;
