/**
 * ================================================================================
 * Reviews Router
 * ================================================================================
 * 
 * Handles CRUD operations for user reviews of the MarketSage platform.
 * Users can view all reviews and submit new ones.
 * 
 * Endpoints:
 * - GET  /api/reviews     - Get all reviews (newest first)
 * - POST /api/reviews     - Submit a new review
 * 
 * Database: SQLite (using better-sqlite3)
 * Table: reviews (id, name, comment, rating, date, avatar_initial)
 * 
 * Author: MarketSage Team
 * ================================================================================
 */

// =============================================================================
// IMPORTS
// =============================================================================

// Express Router - Creates modular route handlers
const express = require('express');
const router = express.Router();

// Database connection - SQLite database for persistent storage
const db = require('../db/database');

// =============================================================================
// GET /api/reviews - Fetch All Reviews
// =============================================================================

/**
 * GET /api/reviews
 * 
 * Retrieves all reviews from the database, ordered by date (newest first).
 * 
 * Response:
 * - 200 OK: Array of review objects
 *   [{
 *     id: number,
 *     name: string,
 *     comment: string,
 *     rating: number (1-5),
 *     date: ISO string,
 *     avatar_initial: string (first letter of name)
 *   }, ...]
 * - 500 Error: { error: "Failed to fetch reviews" }
 */
router.get('/', (req, res) => {
    try {
        // Prepare SQL query - ORDER BY date DESC sorts newest first
        const stmt = db.prepare('SELECT * FROM reviews ORDER BY date DESC');

        // Execute query and get all results
        const reviews = stmt.all();

        // Return reviews as JSON array
        res.json(reviews);

    } catch (error) {
        // Log error for debugging
        console.error('Error fetching reviews:', error);

        // Return error response
        res.status(500).json({ error: 'Failed to fetch reviews' });
    }
});

// =============================================================================
// POST /api/reviews - Create New Review
// =============================================================================

/**
 * POST /api/reviews
 * 
 * Creates a new review in the database.
 * 
 * Request Body (JSON):
 * {
 *   name: string (required) - Reviewer's name
 *   comment: string (required) - Review text
 *   rating: number (required) - Rating 1-5
 * }
 * 
 * Response:
 * - 201 Created: The newly created review object
 * - 400 Bad Request: { error: "Name, comment, and rating are required" }
 * - 500 Error: { error: "Failed to add review" }
 */
router.post('/', (req, res) => {
    // Extract fields from request body (JSON)
    const { name, comment, rating } = req.body;

    // -------------------------------------------------------------------------
    // Validation: Ensure all required fields are present
    // -------------------------------------------------------------------------
    if (!name || !comment || !rating) {
        return res.status(400).json({
            error: 'Name, comment, and rating are required'
        });
    }

    try {
        // Generate avatar initial (first letter of name, uppercase)
        // Used for displaying user avatar placeholders in the UI
        const avatar_initial = name.trim().charAt(0).toUpperCase();

        // -------------------------------------------------------------------------
        // Insert new review into database
        // -------------------------------------------------------------------------
        // Prepare parameterized query (prevents SQL injection)
        const stmt = db.prepare(
            'INSERT INTO reviews (name, comment, rating, avatar_initial) VALUES (?, ?, ?, ?)'
        );

        // Execute insert with values
        const result = stmt.run(name, comment, rating, avatar_initial);

        // -------------------------------------------------------------------------
        // Build response object with new review data
        // -------------------------------------------------------------------------
        const newReview = {
            id: result.lastInsertRowid,  // Auto-generated ID from SQLite
            name,
            comment,
            rating,
            date: new Date().toISOString(),  // Current timestamp
            avatar_initial
        };

        // Return 201 Created with new review object
        res.status(201).json(newReview);

    } catch (error) {
        // Log error for debugging
        console.error('Error adding review:', error);

        // Return error response
        res.status(500).json({ error: 'Failed to add review' });
    }
});

// =============================================================================
// EXPORT ROUTER
// =============================================================================

// Export router for use in main index.js
module.exports = router;
