import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Check, CreditCard, Lock, Mail, User, ArrowRight, ShieldCheck } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

const API_URL = 'http://localhost:3001/api';

const AuthModal = () => {
    const { isAuthModalOpen, closeAuthModal, authModalView, login, register, upgradeToPro } = useAuth();
    const [view, setView] = useState(authModalView);
    const [isLoading, setIsLoading] = useState(false);

    // Form States
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [cardNumber, setCardNumber] = useState("");
    const [expiry, setExpiry] = useState("");
    const [cvc, setCvc] = useState("");

    // Subscription settings from admin
    const [subscriptionPrice, setSubscriptionPrice] = useState("3.00");
    const [subscriptionCurrency, setSubscriptionCurrency] = useState("USD");
    const [subscriptionSymbol, setSubscriptionSymbol] = useState("$");

    // Fetch subscription settings
    useEffect(() => {
        const fetchSettings = async () => {
            try {
                const response = await fetch(`${API_URL}/settings/subscription`);
                const data = await response.json();
                if (data.price) setSubscriptionPrice(data.price);
                if (data.currency) setSubscriptionCurrency(data.currency);
                if (data.symbol) setSubscriptionSymbol(data.symbol);
            } catch (error) {
                console.error('Failed to fetch subscription settings:', error);
            }
        };
        if (isAuthModalOpen) {
            fetchSettings();
        }
    }, [isAuthModalOpen]);

    const handleCardNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        let value = e.target.value.replace(/\D/g, '');
        if (value.length > 16) value = value.slice(0, 16);
        value = value.replace(/(\d{4})(?=\d)/g, '$1 '); // Add space every 4 digits
        setCardNumber(value);
    };

    const handleCvcChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value.replace(/\D/g, '').slice(0, 4);
        setCvc(value);
    };

    const handleExpiryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        let value = e.target.value.replace(/\D/g, ''); // Remove non-digits
        if (value.length > 4) value = value.slice(0, 4); // Limit to 4 digits

        if (value.length >= 2) {
            value = value.slice(0, 2) + '/' + value.slice(2);
        }
        setExpiry(value);
    };

    // Sync internal view state with context
    useEffect(() => {
        setView(authModalView);
    }, [authModalView]);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            await login(email, password);
        } catch (error) {
            // Error is handled in AuthContext
        } finally {
            setIsLoading(false);
        }
    };

    const handlePayment = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        await upgradeToPro();
        setIsLoading(false);
    };

    if (!isAuthModalOpen) return null;

    return (
        <AnimatePresence>
            {isAuthModalOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={closeAuthModal}
                        className="absolute inset-0 bg-black/60 backdrop-blur-md"
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="relative w-full max-w-md overflow-hidden rounded-3xl border border-white/10 bg-[#0B1221] shadow-2xl"
                    >
                        {/* Abstract Background Glow */}
                        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-primary/20 blur-[80px] rounded-full pointer-events-none" />
                        <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-64 h-64 bg-indigo-500/10 blur-[80px] rounded-full pointer-events-none" />

                        <div className="relative p-8">
                            <button
                                onClick={closeAuthModal}
                                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white transition-colors"
                            >
                                <X size={20} />
                            </button>

                            {/* Header */}
                            <div className="text-center mb-6">
                                <motion.div
                                    className="w-16 h-16 bg-primary/20 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-primary/30 glow-emerald-subtle"
                                >
                                    {/* MarketSage Logo */}
                                    <svg
                                        viewBox="0 0 24 24"
                                        className="w-8 h-8 text-primary"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                    >
                                        <path d="M3 3v18h18" />
                                        <path d="M7 16l4-6 4 4 6-8" />
                                    </svg>
                                </motion.div>
                                <h3 className="text-2xl font-bold font-display text-white mb-2">
                                    {view === 'login' && "Welcome Back"}
                                    {view === 'signup' && "Create Pro Account"}
                                    {view === 'payment' && "Upgrade to Pro"}
                                </h3>
                                <p className="text-white/50 text-sm">
                                    {view === 'signup'
                                        ? `Sign up & Subscribe to unlock exclusive reports (${subscriptionSymbol}${subscriptionPrice}/mo)`
                                        : view === 'payment'
                                            ? "Complete your upgrade to access all features"
                                            : "Sign in to access your dashboard"
                                    }
                                </p>
                            </div>

                            {/* Views */}
                            <AnimatePresence mode="wait">
                                {view === 'login' && (
                                    <motion.form
                                        key="login-form"
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        onSubmit={handleLogin}
                                        className="space-y-4"
                                    >
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Email Address</label>
                                            <div className="relative">
                                                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="email"
                                                    value={email}
                                                    onChange={(e) => setEmail(e.target.value)}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors"
                                                    placeholder="you@example.com"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Password</label>
                                            <div className="relative">
                                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="password"
                                                    value={password}
                                                    onChange={(e) => setPassword(e.target.value)}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors"
                                                    placeholder="••••••••"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <button
                                            type="submit"
                                            disabled={isLoading}
                                            className="w-full bg-primary hover:bg-emerald-600 text-primary-foreground font-semibold py-3 rounded-xl transition-all flex items-center justify-center gap-2 group mt-6"
                                        >
                                            {isLoading ? "Processing..." : (
                                                <>
                                                    Sign In
                                                    <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                                                </>
                                            )}
                                        </button>

                                        <div className="text-center mt-4">
                                            <button
                                                type="button"
                                                onClick={() => setView('signup')}
                                                className="text-xs text-white/40 hover:text-primary transition-colors"
                                            >
                                                Don't have an account? Create Pro Account
                                            </button>
                                        </div>
                                    </motion.form>
                                )}

                                {/* Combined Signup & Payment */}
                                {view === 'signup' && (
                                    <motion.form
                                        key="signup-form"
                                        initial={{ opacity: 0, x: 20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: -20 }}
                                        onSubmit={async (e) => {
                                            e.preventDefault();
                                            setIsLoading(true);
                                            try {
                                                await register(name, email, password);
                                                await upgradeToPro();
                                            } catch (error) {
                                                // Error is handled in AuthContext
                                            } finally {
                                                setIsLoading(false);
                                            }
                                        }}
                                        className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar"
                                    >
                                        {/* Basic Info */}
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Full Name</label>
                                            <div className="relative">
                                                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="text"
                                                    value={name}
                                                    onChange={(e) => setName(e.target.value)}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors"
                                                    placeholder="John Doe"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Email Address</label>
                                            <div className="relative">
                                                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="email"
                                                    value={email}
                                                    onChange={(e) => setEmail(e.target.value)}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors"
                                                    placeholder="you@example.com"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Password</label>
                                            <div className="relative">
                                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="password"
                                                    value={password}
                                                    onChange={(e) => setPassword(e.target.value)}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors"
                                                    placeholder="Create a password"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        {/* Payment Info */}
                                        <div className="border-t border-white/10 pt-6 mt-6">
                                            <div className="flex justify-between items-center mb-6 bg-primary/5 p-4 rounded-xl border border-primary/20">
                                                <div>
                                                    <span className="block text-sm font-medium text-white/70">Subscription</span>
                                                    <span className="block text-lg font-bold text-white">Pro Plan</span>
                                                </div>
                                                <div className="text-right">
                                                    <span className="block text-2xl font-bold text-primary text-glow">{subscriptionSymbol}{subscriptionPrice}</span>
                                                    <span className="block text-[10px] text-white/40 uppercase tracking-widest">{subscriptionCurrency}/month</span>
                                                </div>
                                            </div>

                                            <div className="space-y-2 mb-4">
                                                <div className="relative">
                                                    <CreditCard className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                    <input
                                                        type="text"
                                                        value={cardNumber}
                                                        onChange={handleCardNumberChange}
                                                        maxLength={19}
                                                        className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                        placeholder="0000 0000 0000 0000"
                                                        required
                                                    />
                                                </div>
                                            </div>

                                            <div className="grid grid-cols-2 gap-4">
                                                <input
                                                    type="text"
                                                    value={expiry}
                                                    onChange={handleExpiryChange}
                                                    maxLength={5}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-center text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                    placeholder="MM/YY"
                                                    required
                                                />
                                                <input
                                                    type="text"
                                                    value={cvc}
                                                    onChange={handleCvcChange}
                                                    maxLength={4}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-center text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                    placeholder="CVC"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <button
                                            type="submit"
                                            disabled={isLoading}
                                            className="w-full bg-gradient-to-r from-primary to-emerald-600 hover:from-emerald-500 hover:to-emerald-400 text-primary-foreground font-semibold py-3 rounded-xl transition-all shadow-lg shadow-primary/20 flex items-center justify-center gap-2 mt-6"
                                        >
                                            {isLoading ? "Setting up..." : "Create Account & Subscribe"}
                                        </button>

                                        <div className="text-center mt-4">
                                            <button
                                                type="button"
                                                onClick={() => setView('login')}
                                                className="text-xs text-white/40 hover:text-primary transition-colors"
                                            >
                                                Already have an account? Sign in
                                            </button>
                                        </div>
                                    </motion.form>
                                )}

                                {view === 'payment' && (
                                    <motion.form
                                        key="payment-only-form"
                                        initial={{ opacity: 0, x: 20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: -20 }}
                                        onSubmit={handlePayment}
                                        className="space-y-4"
                                    >
                                        {/* ... Existing standalone payment form logic if needed,
                                            but streamlined to use the combined one mostly.
                                            Keeping simplified version just in case accessing from 'Unlock' while logged in but not Pro.
                                        */}
                                        <div className="p-4 rounded-xl bg-primary/10 border border-primary/20 mb-6">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-sm font-medium text-primary">Pro Plan</span>
                                                <span className="text-lg font-bold text-white">{subscriptionSymbol}{subscriptionPrice}<span className="text-xs font-normal text-white/50">/{subscriptionCurrency}/mo</span></span>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs text-primary/80">
                                                <ShieldCheck size={12} />
                                                Secure SSL Encryption
                                            </div>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-white/60 ml-1">Card Number</label>
                                            <div className="relative">
                                                <CreditCard className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                                <input
                                                    type="text"
                                                    value={cardNumber}
                                                    onChange={handleCardNumberChange}
                                                    maxLength={19}
                                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                    placeholder="0000 0000 0000 0000"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <input
                                                type="text"
                                                value={expiry}
                                                onChange={handleExpiryChange}
                                                maxLength={5}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-center text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                placeholder="MM/YY"
                                                required
                                            />
                                            <input
                                                type="text"
                                                value={cvc}
                                                onChange={handleCvcChange}
                                                maxLength={4}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-center text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50 transition-colors font-mono"
                                                placeholder="123"
                                                required
                                            />
                                        </div>

                                        <button
                                            type="submit"
                                            disabled={isLoading}
                                            className="w-full bg-gradient-to-r from-primary to-emerald-600 hover:from-emerald-500 hover:to-emerald-400 text-primary-foreground font-bold py-3 rounded-xl transition-all shadow-lg shadow-primary/20 flex items-center justify-center gap-2 mt-6"
                                        >
                                            {isLoading ? "Processing..." : "Complete Upgrade"}
                                        </button>

                                        <div className="text-center mt-2">
                                            <div className="flex items-center justify-center gap-4 text-white/20">
                                                {/* Simple visual placeholders for card brands */}
                                                <div className="h-4 w-8 bg-current rounded-sm opacity-50" />
                                                <div className="h-4 w-8 bg-current rounded-sm opacity-50" />
                                                <div className="h-4 w-8 bg-current rounded-sm opacity-50" />
                                            </div>
                                        </div>
                                    </motion.form>
                                )}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};

export default AuthModal;
