import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, FileText } from "lucide-react";

const PHI = 1.618;

const TermsOfService = () => {
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
                                <FileText className="w-8 h-8 text-primary" />
                            </div>
                            <div>
                                <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground">
                                    Terms of Service
                                </h1>
                                <p className="text-foreground/50">Effective: January 2025</p>
                            </div>
                        </div>

                        {/* Content Card */}
                        <div className="card-premium" style={{ padding: `${PHI * 2}rem` }}>
                            <div className="prose prose-invert max-w-none space-y-8">
                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">1. Acceptance of Terms</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        By accessing and using MarketSage, you agree to be bound by these Terms of Service. If you disagree with any part of these terms, you may not access the service.
                                    </p>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">2. Description of Service</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        MarketSage provides AI-powered stock market predictions and analytics for the Colombo Stock Exchange (CSE). Our platform uses machine learning algorithms including BiLSTM and XGBoost to analyze historical price data.
                                    </p>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">3. Disclaimer</h2>
                                    <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                                        <p className="text-red-300/90 leading-relaxed">
                                            <strong>IMPORTANT:</strong> MarketSage predictions are for informational and educational purposes only. They do not constitute financial advice. Our model achieves approximately 62% directional accuracy and should not be the sole basis for any investment decision. Past performance does not guarantee future results.
                                        </p>
                                    </div>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">4. User Responsibilities</h2>
                                    <ul className="list-disc list-inside text-foreground/70 space-y-2">
                                        <li>You are responsible for maintaining the confidentiality of your account</li>
                                        <li>You agree not to use the service for any unlawful purpose</li>
                                        <li>You will not attempt to reverse engineer or exploit the system</li>
                                        <li>You acknowledge that all investment decisions are your own responsibility</li>
                                    </ul>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">5. Intellectual Property</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        All content, algorithms, and designs on MarketSage are the intellectual property of the platform and its developers. This project was developed as an undergraduate research project at Plymouth University, UK.
                                    </p>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">6. Limitation of Liability</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        MarketSage and its developers shall not be liable for any direct, indirect, incidental, or consequential damages arising from the use of this service, including but not limited to financial losses from trading decisions.
                                    </p>
                                </section>

                                <section>
                                    <h2 className="text-2xl font-display font-bold text-primary mb-4">7. Modifications</h2>
                                    <p className="text-foreground/70 leading-relaxed">
                                        We reserve the right to modify these terms at any time. Continued use of the service after changes constitutes acceptance of the new terms.
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

export default TermsOfService;
