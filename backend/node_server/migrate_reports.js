const db = require('./db/database');

try {
    const stmt = db.prepare("ALTER TABLE reports ADD COLUMN file_path TEXT");
    stmt.run();
    console.log("Successfully added file_path column to reports table.");
} catch (error) {
    if (error.message.includes("duplicate column name")) {
        console.log("Column file_path already exists.");
    } else {
        console.error("Migration failed:", error);
    }
}
