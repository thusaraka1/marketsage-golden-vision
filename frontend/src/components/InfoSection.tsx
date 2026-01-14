import { motion } from "framer-motion";
import { Info, Cpu, BarChart3, GraduationCap, AlertCircle, ShieldAlert } from "lucide-react";

const PHI = 1.618;

const dataFeatures = [
    "Trade Date", "Open (Rs.)", "High (Rs.)", "Low (Rs.)",
    "Close (Rs.)", "Trade Volume", "Share Volume", "Turnover (Rs.)"
];

const InfoSection = () => {
    return (
        <section id="info" className="min-h-screen flex items-center py-32 relative overflow-hidden">
            {/* Ambient Background Effects */}
            <div className="absolute top-1/2 left-0 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />
            <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-blue-500/5 blur-[100px] rounded-full pointer-events-none" />

            <div className="container mx-auto px-6 relative z-10">
                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                    className="text-center"
                    style={{ marginBottom: `${PHI * 3}rem` }}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-primary text-sm font-medium mb-6 backdrop-blur-sm"
                    >
                        <Info className="w-4 h-4" />
                        <span>System Architecture</span>
                    </motion.div>
                    <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-white/90 to-white/50">
                        Behind the <span className="text-primary text-glow">Intelligence</span>
                    </h2>
                    <p className="text-lg text-foreground/60 max-w-2xl mx-auto leading-relaxed">
                        A transparent look at the multi-model AI architecture and data sources powering the next generation of market predictions.
                    </p>
                </motion.div>

                {/* Main Content - Golden Ratio Split */}
                <div className="flex flex-col lg:flex-row" style={{ gap: `${PHI * 2}rem` }}>

                    {/* Left Side - 61.8% - Main Info Cards */}
                    <motion.div
                        className="lg:w-[61.8%] space-y-6"
                    >
                        {/* Multi-Model AI Card */}
                        <motion.div
                            initial={{ opacity: 0, x: -30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            whileHover={{ y: -5, borderColor: "rgba(16, 185, 129, 0.3)" }}
                            className="group relative overflow-hidden rounded-3xl border border-white/5 bg-white/5 backdrop-blur-sm p-8 transition-all duration-500"
                        >
                            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                            <div className="relative z-10 flex flex-col md:flex-row gap-6">
                                <div className="flex-shrink-0">
                                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border border-primary/20 group-hover:scale-110 transition-transform duration-500 shadow-lg shadow-primary/10">
                                        <Cpu className="w-8 h-8 text-primary" />
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <h3 className="font-display text-2xl font-bold text-white mb-3">
                                        Multi-Model AI Architecture
                                    </h3>
                                    <p className="text-white/60 leading-relaxed text-base">
                                        Our system employs an advanced ensemble approach, orchestrating <span className="text-primary font-semibold">BiLSTM</span> (Bidirectional Long Short-Term Memory) neural networks for temporal sequence learning with <span className="text-primary font-semibold">XGBoost</span> gradient boosting for decision tree optimization.
                                    </p>
                                </div>
                            </div>
                        </motion.div>

                        {/* Accuracy Card */}
                        <motion.div
                            initial={{ opacity: 0, x: -30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 }}
                            whileHover={{ y: -5, borderColor: "rgba(234, 179, 8, 0.3)" }}
                            className="group relative overflow-hidden rounded-3xl border border-white/5 bg-white/5 backdrop-blur-sm p-8 transition-all duration-500"
                        >
                            <div className="absolute inset-0 bg-gradient-to-br from-yellow-500/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                            <div className="relative z-10 flex flex-col md:flex-row gap-6">
                                <div className="flex-shrink-0">
                                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-yellow-500/20 to-yellow-500/5 flex items-center justify-center border border-yellow-500/20 group-hover:scale-110 transition-transform duration-500 shadow-lg shadow-yellow-500/10">
                                        <BarChart3 className="w-8 h-8 text-yellow-400" />
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-center gap-4 mb-3">
                                        <h3 className="font-display text-2xl font-bold text-white">
                                            Model Accuracy
                                        </h3>
                                        <span className="px-3 py-1 rounded-full bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 text-sm font-bold shadow-[0_0_15px_rgba(234,179,8,0.2)]">
                                            ~62.1% Directional
                                        </span>
                                    </div>
                                    <p className="text-white/60 leading-relaxed text-base">
                                        Achieves consistent accuracy on historical test data. This metric represents the probability of correctly predicting market direction (Up/Down) for the next trading day.
                                    </p>
                                </div>
                            </div>
                        </motion.div>

                        {/* Research Project Card */}
                        <motion.div
                            initial={{ opacity: 0, x: -30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.2 }}
                            whileHover={{ y: -5, borderColor: "rgba(59, 130, 246, 0.3)" }}
                            className="group relative overflow-hidden rounded-3xl border border-white/5 bg-white/5 backdrop-blur-sm p-8 transition-all duration-500"
                        >
                            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                            <div className="relative z-10 flex flex-col md:flex-row gap-6">
                                <div className="flex-shrink-0">
                                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-blue-500/5 flex items-center justify-center border border-blue-500/20 group-hover:scale-110 transition-transform duration-500 shadow-lg shadow-blue-500/10">
                                        <GraduationCap className="w-8 h-8 text-blue-400" />
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <h3 className="font-display text-2xl font-bold text-white mb-3">
                                        Academic Research
                                    </h3>
                                    <p className="text-white/60 leading-relaxed text-base">
                                        Developed by <span className="text-blue-400 font-semibold group-hover:text-blue-300 transition-colors">Nethmi</span> as a final year undergraduate research project at <span className="text-white font-medium">Plymouth University, UK</span>. This project explores the efficacy of hybrid deep learning models in emerging markets.
                                    </p>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Right Side - 38.2% - Data Features */}
                    <motion.div
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="lg:w-[38.2%]"
                    >
                        <div className="sticky top-32 rounded-3xl border border-white/10 bg-[#0B1221]/80 backdrop-blur-xl p-8 shadow-2xl">
                            <div className="flex items-center gap-4 mb-8">
                                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center border border-primary/20">
                                    <svg className="w-6 h-6 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M3 3v18h18" />
                                        <path d="M7 16l4-4 4 4 6-6" />
                                    </svg>
                                </div>
                                <div>
                                    <h3 className="font-display text-xl font-bold text-white">
                                        Data Features
                                    </h3>
                                    <p className="text-white/40 text-xs uppercase tracking-wider font-medium">Model Inputs</p>
                                </div>
                            </div>

                            <div className="mb-8 p-4 rounded-xl bg-white/5 border border-white/5">
                                <p className="text-sm text-white/70 leading-relaxed">
                                    Predictions are strictly based on historical <span className="text-white font-medium">OHLCV data</span>.
                                    <span className="block mt-2 text-xs text-white/40">
                                        * No external sentiment analysis or fundamental indicators are currently used to ensure pure technical pattern recognition.
                                    </span>
                                </p>
                            </div>

                            {/* Feature Tags */}
                            <div className="flex flex-wrap gap-2">
                                {dataFeatures.map((feature, i) => (
                                    <motion.span
                                        key={feature}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        whileInView={{ opacity: 1, scale: 1 }}
                                        viewport={{ once: true }}
                                        transition={{ delay: 0.4 + i * 0.05 }}
                                        whileHover={{ scale: 1.05, backgroundColor: "rgba(16, 185, 129, 0.15)", borderColor: "rgba(16, 185, 129, 0.4)" }}
                                        className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs font-medium text-white/80 cursor-default transition-colors hover:text-white"
                                    >
                                        {feature}
                                    </motion.span>
                                ))}
                            </div>

                            {/* Disclaimer */}
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: 0.6 }}
                                className="mt-8 pt-6 border-t border-white/10"
                            >
                                <div className="rounded-xl bg-red-500/5 border border-red-500/10 p-4 flex gap-3">
                                    <ShieldAlert className="w-5 h-5 text-red-500/60 flex-shrink-0 mt-0.5" />
                                    <div className="space-y-1">
                                        <p className="text-xs font-bold text-red-400">Educational Purpose Only</p>
                                        <p className="text-[11px] text-red-400/70 leading-relaxed">
                                            This tool demonstrates machine learning capabilities. Past performance does not guarantee future results. Not intended as financial advice.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default InfoSection;
