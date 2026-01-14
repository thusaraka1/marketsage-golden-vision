const express = require('express');
const db = require('../db/database');

const router = express.Router();

// Currencies list (shared between admin and frontend)
const currencies = [
    { code: "USD", symbol: "$", name: "US Dollar" },
    { code: "EUR", symbol: "€", name: "Euro" },
    { code: "GBP", symbol: "£", name: "British Pound" },
    { code: "LKR", symbol: "Rs.", name: "Sri Lankan Rupee" },
    { code: "INR", symbol: "₹", name: "Indian Rupee" },
    { code: "AUD", symbol: "A$", name: "Australian Dollar" },
    { code: "JPY", symbol: "¥", name: "Japanese Yen" },
    // Add more as needed
];

// GET /api/settings/subscription - Get subscription settings (public)
router.get('/subscription', (req, res) => {
    try {
        const price = db.prepare('SELECT value FROM settings WHERE key = ?').get('subscription_price');
        const currency = db.prepare('SELECT value FROM settings WHERE key = ?').get('subscription_currency');

        const currencyInfo = currencies.find(c => c.code === (currency?.value || 'USD')) || currencies[0];

        res.json({
            price: price?.value || '3.00',
            currency: currency?.value || 'USD',
            symbol: currencyInfo.symbol,
            currencyName: currencyInfo.name
        });
    } catch (error) {
        console.error('Error fetching subscription settings:', error);
        res.status(500).json({ error: 'Failed to fetch settings' });
    }
});

// PUT /api/settings/subscription - Update subscription settings (admin)
router.put('/subscription', (req, res) => {
    try {
        const { price, currency } = req.body;

        if (!price || !currency) {
            return res.status(400).json({ error: 'Price and currency are required' });
        }

        // Update or insert settings
        const stmt = db.prepare('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)');
        stmt.run('subscription_price', price.toString());
        stmt.run('subscription_currency', currency);

        const currencyInfo = currencies.find(c => c.code === currency) || currencies[0];

        console.log(`✓ Subscription settings updated: ${currencyInfo.symbol}${price} ${currency}/month`);

        res.json({
            message: 'Subscription settings updated',
            price,
            currency,
            symbol: currencyInfo.symbol
        });
    } catch (error) {
        console.error('Error updating subscription settings:', error);
        res.status(500).json({ error: 'Failed to update settings' });
    }
});

module.exports = router;
