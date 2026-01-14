const db = require('./db/database');

const reviews = [
    {
        name: "Kasun Perera",
        comment: "This platform helped me make informed decisions on my JKH investments. The AI predictions are surprisingly accurate!",
        rating: 5,
        date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
        avatar_initial: "K"
    },
    {
        name: "Dilini Fernando",
        comment: "Clean interface and easy to understand charts. Love the Golden Ratio design - very professional.",
        rating: 4,
        date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
        avatar_initial: "D"
    },
    {
        name: "Ashan Silva",
        comment: "Finally, a trading tool focused on CSE stocks. The time filters make analysis so much easier.",
        rating: 5,
        date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 1 week ago
        avatar_initial: "A"
    }
];

const stmt = db.prepare(`
  INSERT INTO reviews (name, comment, rating, date, avatar_initial)
  VALUES (?, ?, ?, ?, ?)
`);

const insertMany = db.transaction((reviews) => {
    for (const review of reviews) {
        stmt.run(review.name, review.comment, review.rating, review.date, review.avatar_initial);
        console.log(`Inserted review from: ${review.name}`);
    }
});

try {
    insertMany(reviews);
    console.log('Successfully seeded reviews!');
} catch (error) {
    console.error('Error seeding reviews:', error);
}
