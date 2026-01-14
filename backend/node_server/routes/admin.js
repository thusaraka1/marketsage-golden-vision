const express = require('express');
const db = require('../db/database');

const router = express.Router();

// GET /api/admin/users - Get all users for admin panel
router.get('/users', (req, res) => {
    try {
        const users = db.prepare(`
            SELECT 
                id,
                name,
                email,
                plan,
                status,
                is_pro,
                created_at as joined
            FROM users 
            ORDER BY created_at DESC
        `).all();

        // Format for frontend
        const formattedUsers = users.map(user => ({
            id: user.id,
            name: user.name,
            email: user.email,
            plan: user.plan,
            status: user.status,
            joined: user.joined ? user.joined.split('T')[0] : new Date().toISOString().split('T')[0]
        }));

        res.json({ users: formattedUsers });
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).json({ error: 'Failed to fetch users' });
    }
});

// PUT /api/admin/users/:id/status - Toggle user status
router.put('/users/:id/status', (req, res) => {
    try {
        const { id } = req.params;

        // Get current status
        const user = db.prepare('SELECT status FROM users WHERE id = ?').get(id);
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }

        // Toggle status
        const newStatus = user.status === 'Active' ? 'Suspended' : 'Active';
        db.prepare('UPDATE users SET status = ? WHERE id = ?').run(newStatus, id);

        console.log(`✓ User ${id} status changed to ${newStatus}`);

        res.json({ message: 'Status updated', status: newStatus });
    } catch (error) {
        console.error('Error updating user status:', error);
        res.status(500).json({ error: 'Failed to update status' });
    }
});

// DELETE /api/admin/users/:id - Delete user
router.delete('/users/:id', (req, res) => {
    try {
        const { id } = req.params;

        const result = db.prepare('DELETE FROM users WHERE id = ?').run(id);

        if (result.changes === 0) {
            return res.status(404).json({ error: 'User not found' });
        }

        console.log(`✓ User ${id} deleted`);

        res.json({ message: 'User deleted successfully' });
    } catch (error) {
        console.error('Error deleting user:', error);
        res.status(500).json({ error: 'Failed to delete user' });
    }
});

module.exports = router;
