import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, Shield } from "lucide-react";

const PHI = 1.618;

const PrivacyPolicy = () => {
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
                        className="max-w-4xl mx-auto"
                    >
                        {/* Header */}
                        <div className="flex items-center gap-4 mb-8">
                            <div
                                className="rounded-2xl bg-primary/20 flex items-center justify-center"
                                style={{ width: `${PHI * 3}rem`, height: `${PHI * 3}rem` }}
                            >
                                <Shield className="w-8 h-8 text-primary" />
                            </div>
                            <div>
                                <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground">
                                    Privacy Policy
                                </h1>
                                <p className="text-foreground/50">Last updated: January 2025</p>
                            </div>
                        </div>

                        {/* Content Card */}
                        <div className="card-premium" style={{ padding: `${PHI * 2}rem` }}>
                            <div className="prose prose-invert max-w-none space-y-8">
                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">1. Information We Collect</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        MarketSage collects minimal personal information necessary to provide our stock prediction services. This includes:
                                    </p>
                                    <ul className="list-disc list-inside text-foreground/70 mt-4 space-y-2">
                                        <li>Account information (email, username) when you register</li>
                                        <li>Usage data to improve our AI models and user experience</li>
                                        <li>Technical data such as browser type and IP address</li>
                                    </ul>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">2. How We Use Your Data</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        Your data is used exclusively to:
                                    </p>
                                    <ul className="list-disc list-inside text-foreground/70 mt-4 space-y-2">
                                        <li>Provide personalized stock predictions and market analysis</li>
                                        <li>Improve our machine learning algorithms</li>
                                        <li>Send important service notifications</li>
                                        <li>Maintain platform security and prevent fraud</li>
                                    </ul>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">3. Data Protection</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        We implement industry-standard security measures including:
                                    </p>
                                    <ul className="list-disc list-inside text-foreground/70 mt-4 space-y-2">
                                        <li>256-bit SSL/TLS encryption for all data transmission</li>
                                        <li>Secure data storage with regular security audits</li>
                                        <li>Strict access controls for our development team</li>
                                    </ul>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">4. Third-Party Services</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        We do not sell your personal data to third parties. Limited data may be shared with trusted service providers who assist in operating our platform, subject to strict confidentiality agreements.
                                    </p>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">5. Your Rights</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        Under GDPR and similar regulations, you have the right to:
                                    </p>
                                    <ul className="list-disc list-inside text-foreground/70 mt-4 space-y-2">
                                        <li>Access your personal data</li>
                                        <li>Request data correction or deletion</li>
                                        <li>Withdraw consent at any time</li>
                                        <li>Export your data in a portable format</li>
                                    </ul>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">6. Contact Us</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        For privacy-related inquiries, please contact us at{" "}
                                        <a href="mailto:privacy@marketsage.com" className="text-primary hover:underline">
                                            privacy@marketsage.com
                                        </a>
                                    </p>
                                </section>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </main>
        </div>
    );
};

export default PrivacyPolicy;
