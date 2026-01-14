-- Users table for storing registered users
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    plan TEXT DEFAULT 'Free',
    status TEXT DEFAULT 'Active',
    is_pro INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Settings table for storing app configuration
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Insert default subscription settings if they don't exist
INSERT OR IGNORE INTO settings (key, value) VALUES ('subscription_price', '3.00');
INSERT OR IGNORE INTO settings (key, value) VALUES ('subscription_currency', 'USD');

-- Reports table for storing uploaded reports
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    size TEXT NOT NULL,
    file_path TEXT,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    downloads INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    comment TEXT NOT NULL,
    rating INTEGER NOT NULL,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    avatar_initial TEXT
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Admins table for super admin access
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
