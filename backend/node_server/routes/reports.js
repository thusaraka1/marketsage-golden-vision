const express = require('express');
const router = express.Router();
const db = require('../db/database');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Configure Multer Storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '../uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir);
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        // Sanitize filename and append timestamp to prevent collisions
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// GET all reports
router.get('/', (req, res) => {
    try {
        const stmt = db.prepare('SELECT * FROM reports ORDER BY upload_date DESC');
        const reports = stmt.all();
        res.json(reports);
    } catch (error) {
        console.error('Error fetching reports:', error);
        res.status(500).json({ error: 'Failed to fetch reports' });
    }
});

// POST new report with file
router.post('/', upload.single('file'), (req, res) => {
    try {
        const { name, category, description } = req.body;
        const file = req.file;

        // If no file uploaded, handle as error or optional? For reports, file is crucial.
        // Assuming file is mandatory for a new report upload
        if (!file) {
            return res.status(400).json({ error: 'Report file is required' });
        }

        const size = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
        const file_path = file.filename; // Store only the filename

        const stmt = db.prepare('INSERT INTO reports (name, category, description, size, file_path) VALUES (?, ?, ?, ?, ?)');
        const result = stmt.run(name, category, description, size, file_path);

        const newReport = {
            id: result.lastInsertRowid,
            name,
            category,
            description,
            size,
            file_path,
            upload_date: new Date().toISOString(),
            downloads: 0
        };

        res.status(201).json(newReport);
    } catch (error) {
        console.error('Error adding report:', error);
        res.status(500).json({ error: 'Failed to add report' });
    }
});

// GET download report
router.get('/:id/download', (req, res) => {
    try {
        const { id } = req.params;
        const stmt = db.prepare('SELECT * FROM reports WHERE id = ?');
        const report = stmt.get(id);

        if (!report || !report.file_path) {
            return res.status(404).json({ error: 'File not found' });
        }

        const filePath = path.join(__dirname, '../uploads', report.file_path);

        if (fs.existsSync(filePath)) {
            // Increment download count
            const updateStmt = db.prepare('UPDATE reports SET downloads = downloads + 1 WHERE id = ?');
            updateStmt.run(id);

            res.download(filePath, report.name + '.pdf'); // Download with original report name
        } else {
            res.status(404).json({ error: 'File on disk not found' });
        }
    } catch (error) {
        console.error('Error downloading report:', error);
        res.status(500).json({ error: 'Download failed' });
    }
});

// DELETE report
router.delete('/:id', (req, res) => {
    try {
        const { id } = req.params;

        // Get file path to delete file from disk
        const getStmt = db.prepare('SELECT file_path FROM reports WHERE id = ?');
        const report = getStmt.get(id);

        if (report && report.file_path) {
            const filePath = path.join(__dirname, '../uploads', report.file_path);
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
            }
        }

        const stmt = db.prepare('DELETE FROM reports WHERE id = ?');
        stmt.run(id);
        res.json({ message: 'Report deleted successfully' });
    } catch (error) {
        console.error('Error deleting report:', error);
        res.status(500).json({ error: 'Failed to delete report' });
    }
});

module.exports = router;
