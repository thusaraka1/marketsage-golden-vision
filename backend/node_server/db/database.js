const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

// Ensure data directory exists
const dataDir = path.join(__dirname, '..', 'data');
if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
}

// Initialize database
const dbPath = path.join(dataDir, 'marketsage.db');
const db = new Database(dbPath);

// Enable foreign keys
db.pragma('journal_mode = WAL');

const bcrypt = require('bcryptjs');

// Initialize schema
function initializeDatabase() {
    const schemaPath = path.join(__dirname, 'schema.sql');
    const schema = fs.readFileSync(schemaPath, 'utf-8');
    db.exec(schema);

    // Seed default admin user
    const admin = db.prepare('SELECT * FROM admins WHERE username = ?').get('admin');
    if (!admin) {
        const salt = bcrypt.genSaltSync(10);
        const hash = bcrypt.hashSync('admin', salt);
        db.prepare('INSERT INTO admins (username, password_hash) VALUES (?, ?)').run('admin', hash);
        console.log('✓ Default admin user seeded');
    }

    console.log('✓ Database initialized');
}

// Initialize on module load
initializeDatabase();

module.exports = db;
