import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Send, User, Star } from "lucide-react";

const PHI = 1.618;

const API_URL = 'http://localhost:3001/api';

interface Review {
    id: number;
    name: string;
    comment: string;
    rating: number;
    date: string;
    avatar_initial?: string;
}

const timeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (seconds < 60) return "Just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    const weeks = Math.floor(days / 7);
    if (weeks < 4) return `${weeks}w ago`;
    return date.toLocaleDateString();
};

const ReviewsSection = () => {
    const [reviews, setReviews] = useState<Review[]>([]);
    const [name, setName] = useState("");
    const [comment, setComment] = useState("");
    const [rating, setRating] = useState(5);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        fetchReviews();
    }, []);

    const fetchReviews = async () => {
        try {
            const response = await fetch(`${API_URL}/reviews`);
            const data = await response.json();
            setReviews(data); // Data is already sorted by date desc from API
        } catch (error) {
            console.error('Failed to fetch reviews:', error);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!name.trim() || !comment.trim()) return;

        setIsSubmitting(true);

        try {
            const response = await fetch(`${API_URL}/reviews`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name.trim(), comment: comment.trim(), rating })
            });

            if (response.ok) {
                const newReview = await response.json();
                setReviews([newReview, ...reviews]);
                setName("");
                setComment("");
                setRating(5);
            }
        } catch (error) {
            console.error('Failed to submit review:', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <section
            id="reviews"
            className="min-h-screen py-32 relative"
        >
            {/* Background gradient */}
            <div className="absolute top-0 left-0 right-0 h-[400px] bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />

            <div className="container mx-auto px-6">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                    className="text-center"
                    style={{ marginBottom: `${PHI * 3}rem` }}
                >
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
                        <MessageSquare className="w-4 h-4" />
                        User Testimonials
                    </div>
                    <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
                        <span className="text-foreground">What Traders</span>{" "}
                        <span className="text-primary text-glow">Say</span>
                    </h2>
                    <p className="text-lg text-foreground/50 max-w-2xl mx-auto">
                        Join thousands of traders who trust MarketSage for their investment decisions
                    </p>
                </motion.div>

                {/* Two Column Layout - Golden Ratio Split */}
                <div className="flex flex-col lg:flex-row" style={{ gap: `${PHI * 2}rem` }}>
                    {/* Left: Comment Form (38.2%) */}
                    <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                        className="lg:w-[38.2%]"
                    >
                        <div className="card-premium sticky top-32" style={{ padding: `${PHI * 1.5}rem` }}>
                            <h3 className="font-display text-2xl font-bold text-foreground mb-2">
                                Share Your Experience
                            </h3>
                            <p className="text-foreground/50 text-sm mb-6">
                                Your feedback helps us improve and helps others make informed decisions.
                            </p>

                            <form onSubmit={handleSubmit} className="space-y-4">
                                {/* Name Input */}
                                <div>
                                    <label className="block text-sm font-medium text-foreground/70 mb-2">
                                        Your Name
                                    </label>
                                    <div className="relative">
                                        <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-primary/50" />
                                        <input
                                            type="text"
                                            value={name}
                                            onChange={(e) => setName(e.target.value)}
                                            placeholder="Enter your name"
                                            className="w-full input-premium pl-12"
                                            style={{ height: `${PHI * 2.5}rem` }}
                                        />
                                    </div>
                                </div>

                                {/* Rating */}
                                <div>
                                    <label className="block text-sm font-medium text-foreground/70 mb-2">
                                        Rating
                                    </label>
                                    <div className="flex items-center" style={{ gap: `${PHI * 0.3}rem` }}>
                                        {[1, 2, 3, 4, 5].map((star) => (
                                            <button
                                                key={star}
                                                type="button"
                                                onClick={() => setRating(star)}
                                                className="transition-transform hover:scale-110"
                                            >
                                                <Star
                                                    className={`w-6 h-6 ${star <= rating
                                                        ? "text-primary fill-primary"
                                                        : "text-foreground/30"
                                                        }`}
                                                />
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Comment Textarea */}
                                <div>
                                    <label className="block text-sm font-medium text-foreground/70 mb-2">
                                        Your Review
                                    </label>
                                    <textarea
                                        value={comment}
                                        onChange={(e) => setComment(e.target.value)}
                                        placeholder="Share your experience with MarketSage..."
                                        rows={4}
                                        className="w-full input-premium resize-none"
                                        style={{ padding: `${PHI * 0.8}rem` }}
                                    />
                                </div>

                                {/* Submit Button */}
                                <button
                                    type="submit"
                                    disabled={isSubmitting || !name.trim() || !comment.trim()}
                                    className="w-full flex items-center justify-center gap-2 rounded-xl bg-primary text-primary-foreground font-bold transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
                                    style={{ height: `${PHI * 2.5}rem` }}
                                >
                                    {isSubmitting ? (
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <>
                                            <Send className="w-4 h-4" />
                                            Submit Review
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>
                    </motion.div>

                    {/* Right: Reviews List (61.8%) */}
                    <motion.div
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="lg:w-[61.8%]"
                    >
                        <div className="space-y-4" style={{ gap: `${PHI}rem` }}>
                            <AnimatePresence mode="popLayout">
                                {reviews.map((review, index) => (
                                    <motion.div
                                        key={review.id}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -20 }}
                                        transition={{ delay: index * 0.1 }}
                                        layout
                                        className="card-premium"
                                        style={{ padding: `${PHI * 1.2}rem` }}
                                    >
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex items-center gap-3">
                                                {/* Avatar */}
                                                <div
                                                    className="rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold"
                                                    style={{ width: `${PHI * 2.5}rem`, height: `${PHI * 2.5}rem` }}
                                                >
                                                    {review.name.charAt(0).toUpperCase()}
                                                </div>
                                                <div>
                                                    <h4 className="font-semibold text-foreground">
                                                        {review.name}
                                                    </h4>
                                                    <p className="text-xs text-foreground/40">{timeAgo(review.date)}</p>
                                                </div>
                                            </div>
                                            {/* Stars */}
                                            <div className="flex items-center gap-0.5">
                                                {Array.from({ length: 5 }).map((_, i) => (
                                                    <Star
                                                        key={i}
                                                        className={`w-4 h-4 ${i < review.rating
                                                            ? "text-primary fill-primary"
                                                            : "text-foreground/20"
                                                            }`}
                                                    />
                                                ))}
                                            </div>
                                        </div>
                                        <p className="text-foreground/70 leading-relaxed">
                                            {review.comment}
                                        </p>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default ReviewsSection;
