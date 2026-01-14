const fs = require('fs');
const path = require('path');
// Native fetch is available in Node 22

// Create a dummy PDF file
const dummyPdfPath = path.join(__dirname, 'test_report.pdf');
fs.writeFileSync(dummyPdfPath, 'Dummy PDF content');

async function testUpload() {
    try {
        const formData = new FormData();
        const fileBlob = new Blob([fs.readFileSync(dummyPdfPath)], { type: 'application/pdf' });
        formData.append('file', fileBlob, 'test_report.pdf');
        formData.append('name', 'Test Report API');
        formData.append('category', 'Risk Analysis');
        formData.append('description', 'Test Description');

        // Check if global fetch works or fail gracefully
        if (typeof fetch === 'undefined') {
            console.log('Fetch API not available globally (Node < 18). Please update node or install undici.');
            return;
        }

        const res = await fetch('http://localhost:3001/api/reports', {
            method: 'POST',
            body: formData
        });

        if (res.ok) {
            const data = await res.json();
            console.log('Upload Successful:', data);

            // Now try download
            const downloadRes = await fetch(`http://localhost:3001/api/reports/${data.id}/download`);
            if (downloadRes.ok) {
                console.log('Download check successful');
            } else {
                console.error('Download check failed');
            }
        } else {
            console.error('Upload failed:', await res.text());
        }
    } catch (e) {
        console.error('Error:', e);
    } finally {
        fs.unlinkSync(dummyPdfPath);
    }
}

testUpload();
