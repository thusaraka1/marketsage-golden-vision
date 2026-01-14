import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, Code, Terminal, Zap, Database, Lock } from "lucide-react";

const PHI = 1.618;

const endpoints = [
    {
        method: "GET",
        path: "/api/v1/predictions/{symbol}",
        description: "Get AI prediction for a specific stock symbol",
        params: [
            { name: "symbol", type: "string", desc: "Stock symbol (e.g., COMB, CTC, JKH)" },
            { name: "timeframe", type: "string", desc: "Prediction timeframe: 1D, 7D, 30D" },
        ],
    },
    {
        method: "GET",
        path: "/api/v1/stocks",
        description: "List all available stock symbols",
        params: [],
    },
    {
        method: "GET",
        path: "/api/v1/historical/{symbol}",
        description: "Get historical price data for a stock",
        params: [
            { name: "symbol", type: "string", desc: "Stock symbol" },
            { name: "from", type: "date", desc: "Start date (YYYY-MM-DD)" },
            { name: "to", type: "date", desc: "End date (YYYY-MM-DD)" },
        ],
    },
    {
        method: "POST",
        path: "/api/v1/analyze",
        description: "Run custom analysis on provided data",
        params: [
            { name: "data", type: "array", desc: "Array of OHLCV data points" },
            { name: "model", type: "string", desc: "Model type: bilstm, xgboost, ensemble" },
        ],
    },
];

const ApiDocs = () => {
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
                        <div className="flex items-center gap-4 mb-8">
                            <div
                                className="rounded-2xl bg-primary/20 flex items-center justify-center"
                                style={{ width: `${PHI * 3}rem`, height: `${PHI * 3}rem` }}
                            >
                                <Code className="w-8 h-8 text-primary" />
                            </div>
                            <div>
                                <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground">
                                    API Documentation
                                </h1>
                                <p className="text-foreground/50">Version 1.0 • RESTful API</p>
                            </div>
                        </div>

                        {/* Quick Start */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="card-premium mb-8"
                            style={{ padding: `${PHI * 1.5}rem` }}
                        >
                            <h2 className="text-xl font-display font-bold text-foreground mb-4 flex items-center gap-2">
                                <Zap className="w-5 h-5 text-primary" />
                                Quick Start
                            </h2>
                            <div className="bg-black/40 rounded-xl p-4 font-mono text-sm overflow-x-auto">
                                <code className="text-foreground/80">
                                    <span className="text-primary">curl</span> -X GET \<br />
                                    <span className="text-foreground/50">&nbsp;&nbsp;</span>"https://api.marketsage.com/v1/predictions/COMB" \<br />
                                    <span className="text-foreground/50">&nbsp;&nbsp;</span>-H "Authorization: Bearer YOUR_API_KEY"
                                </code>
                            </div>
                        </motion.div>

                        {/* Features Grid */}
                        <div className="grid md:grid-cols-3 gap-6 mb-12">
                            {[
                                { icon: Terminal, title: "RESTful API", desc: "Clean, predictable endpoints" },
                                { icon: Lock, title: "Secure", desc: "JWT authentication" },
                                { icon: Database, title: "Real-time", desc: "Live CSE data feeds" },
                            ].map((feature, i) => (
                                <motion.div
                                    key={feature.title}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 + i * 0.1 }}
                                    className="card-premium text-center"
                                    style={{ padding: `${PHI * 1.2}rem` }}
                                >
                                    <feature.icon className="w-8 h-8 text-primary mx-auto mb-3" />
                                    <h3 className="font-semibold text-foreground mb-1">{feature.title}</h3>
                                    <p className="text-foreground/50 text-sm">{feature.desc}</p>
                                </motion.div>
                            ))}
                        </div>

                        {/* Endpoints */}
                        <h2 className="text-2xl font-display font-bold text-foreground mb-6">Endpoints</h2>
                        <div className="space-y-4">
                            {endpoints.map((endpoint, i) => (
                                <motion.div
                                    key={endpoint.path}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.4 + i * 0.1 }}
                                    className="card-premium overflow-hidden"
                                >
                                    <div
                                        className="flex items-center gap-4 border-b border-white/5"
                                        style={{ padding: `${PHI}rem` }}
                                    >
                                        <span className={`px-3 py-1 rounded-lg text-xs font-bold ${endpoint.method === "GET" ? "bg-primary/20 text-primary" : "bg-yellow-500/20 text-yellow-400"
                                            }`}>
                                            {endpoint.method}
                                        </span>
                                        <code className="font-mono text-foreground">{endpoint.path}</code>
                                    </div>
                                    <div style={{ padding: `${PHI}rem` }}>
                                        <p className="text-foreground/70 mb-4">{endpoint.description}</p>
                                        {endpoint.params.length > 0 && (
                                            <div className="bg-black/20 rounded-lg overflow-hidden">
                                                <table className="w-full text-sm">
                                                    <thead>
                                                        <tr className="border-b border-white/10">
                                                            <th className="text-left px-4 py-2 text-foreground/50">Parameter</th>
                                                            <th className="text-left px-4 py-2 text-foreground/50">Type</th>
                                                            <th className="text-left px-4 py-2 text-foreground/50">Description</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {endpoint.params.map((param) => (
                                                            <tr key={param.name} className="border-b border-white/5">
                                                                <td className="px-4 py-2 font-mono text-primary">{param.name}</td>
                                                                <td className="px-4 py-2 text-foreground/60">{param.type}</td>
                                                                <td className="px-4 py-2 text-foreground/70">{param.desc}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </main>
        </div>
    );
};

export default ApiDocs;
