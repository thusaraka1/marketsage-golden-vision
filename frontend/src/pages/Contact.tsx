import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, Mail, MessageSquare, MapPin, Send, Check } from "lucide-react";

const PHI = 1.618;

const Contact = () => {
    const [formData, setFormData] = useState({ name: "", email: "", subject: "", message: "" });
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        // Simulate form submission
        setTimeout(() => {
            setSubmitted(true);
        }, 500);
    };

    return (
        <div className="min-h-screen bg-gradient-void">
            {/* Header */}
            <header className="fixed top-0 left-0 right-0 z-50 frosted-header">
                <div className="container mx-auto px-6 py-4 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 text-foreground/70 hover:text-primary transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Home</span>
                    </Link>
                </div>
            </header>

            {/* Content */}
            <main className="pt-32 pb-24">
                <div className="container mx-auto px-6">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="max-w-5xl mx-auto"
                    >
                        {/* Header */}
                        <div className="text-center mb-12">
                            <div
                                className="rounded-2xl bg-primary/20 flex items-center justify-center mx-auto mb-6"
                                style={{ width: `${PHI * 4}rem`, height: `${PHI * 4}rem` }}
                            >
                                <MessageSquare className="w-10 h-10 text-primary" />
                            </div>
                            <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-4">
                                Get in Touch
                            </h1>
                            <p className="text-foreground/50 max-w-xl mx-auto">
                                Have questions about MarketSage? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
                            </p>
                        </div>

                        {/* Two Column Layout - Golden Ratio */}
                        <div className="flex flex-col lg:flex-row" style={{ gap: `${PHI * 2}rem` }}>
                            {/* Left - Contact Form (61.8%) */}
                            <motion.div
                                initial={{ opacity: 0, x: -30 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.2 }}
                                className="lg:w-[61.8%]"
                            >
                                <div className="card-premium" style={{ padding: `${PHI * 2}rem` }}>
                                    {submitted ? (
                                        <motion.div
                                            initial={{ opacity: 0, scale: 0.9 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            className="text-center py-12"
                                        >
                                            <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-6">
                                                <Check className="w-10 h-10 text-primary" />
                                            </div>
                                            <h3 className="text-2xl font-display font-bold text-foreground mb-2">
                                                Message Sent!
                                            </h3>
                                            <p className="text-foreground/60">
                                                Thank you for reaching out. We'll get back to you within 24 hours.
                                            </p>
                                        </motion.div>
                                    ) : (
                                        <form onSubmit={handleSubmit} className="space-y-6">
                                            <div className="grid md:grid-cols-2 gap-6">
                                                <div>
                                                    <label className="block text-sm font-medium text-foreground/70 mb-2">
                                                        Your Name
                                                    </label>
                                                    <input
                                                        type="text"
                                                        required
                                                        value={formData.name}
                                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                                        className="w-full input-premium"
                                                        placeholder="John Doe"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-foreground/70 mb-2">
                                                        Email Address
                                                    </label>
                                                    <input
                                                        type="email"
                                                        required
                                                        value={formData.email}
                                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                                        className="w-full input-premium"
                                                        placeholder="john@example.com"
                                                    />
                                                </div>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-foreground/70 mb-2">
                                                    Subject
                                                </label>
                                                <input
                                                    type="text"
                                                    required
                                                    value={formData.subject}
                                                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                                    className="w-full input-premium"
                                                    placeholder="How can we help?"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-foreground/70 mb-2">
                                                    Message
                                                </label>
                                                <textarea
                                                    required
                                                    rows={5}
                                                    value={formData.message}
                                                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                                    className="w-full input-premium resize-none"
                                                    placeholder="Tell us more about your inquiry..."
                                                />
                                            </div>
                                            <button
                                                type="submit"
                                                className="w-full flex items-center justify-center gap-2 rounded-xl bg-primary text-primary-foreground font-bold transition-all duration-300 hover:scale-[1.02]"
                                                style={{ height: `${PHI * 2.5}rem` }}
                                            >
                                                <Send className="w-4 h-4" />
                                                Send Message
                                            </button>
                                        </form>
                                    )}
                                </div>
                            </motion.div>

                            {/* Right - Contact Info (38.2%) */}
                            <motion.div
                                initial={{ opacity: 0, x: 30 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.3 }}
                                className="lg:w-[38.2%] space-y-6"
                            >
                                {/* Email */}
                                <div className="card-premium flex items-start gap-4" style={{ padding: `${PHI * 1.2}rem` }}>
                                    <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center flex-shrink-0">
                                        <Mail className="w-6 h-6 text-primary" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-foreground mb-1">Email</h3>
                                        <a href="mailto:support@marketsage.com" className="text-foreground/60 hover:text-primary transition-colors">
                                            support@marketsage.com
                                        </a>
                                    </div>
                                </div>

                                {/* Location */}
                                <div className="card-premium flex items-start gap-4" style={{ padding: `${PHI * 1.2}rem` }}>
                                    <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center flex-shrink-0">
                                        <MapPin className="w-6 h-6 text-primary" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-foreground mb-1">Location</h3>
                                        <p className="text-foreground/60">
                                            Plymouth University<br />
                                            Devon, United Kingdom
                                        </p>
                                    </div>
                                </div>

                                {/* Research Project Info */}
                                <div className="card-premium" style={{ padding: `${PHI * 1.2}rem` }}>
                                    <div className="p-4 rounded-xl bg-primary/5 border border-primary/10">
                                        <p className="text-sm text-foreground/70 leading-relaxed">
                                            <strong className="text-primary">Note:</strong> This is an academic research project. Response times may vary during examination periods.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                </div>
            </main>
        </div>
    );
};

export default Contact;
