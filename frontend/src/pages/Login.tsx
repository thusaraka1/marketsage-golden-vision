import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            await login(email, password);

            // Get updated user data from localStorage after login
            const storedUser = localStorage.getItem('marketsage_user');
            if (storedUser) {
                const userData = JSON.parse(storedUser);

                // Only Super Admins can access the admin panel
                if (userData.isAdmin === true) {
                    navigate('/admin');
                } else {
                    // Regular users cannot access admin panel
                    toast.error('Access Denied: Admin privileges required');
                    // Redirect to home page instead
                    navigate('/');
                }
            }
        } catch (error) {
            // Error handling is done in AuthContext, but we can double ensure here if needed
            console.error("Login failed", error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen w-full flex bg-background text-foreground overflow-hidden">
            {/* Major Section - Visuals (Golden Ratio ~62%) */}
            <div className="hidden lg:flex flex-[1.618] relative flex-col justify-between p-12 overflow-hidden bg-gunmetal-deep">
                {/* Background Effects */}
                <div className="absolute inset-0 z-0">
                    <div className="absolute top-[-20%] left-[-10%] w-[70%] h-[70%] rounded-full bg-emerald-glow/10 blur-[120px] animate-pulse-glow" />
                    <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] rounded-full bg-emerald-dim/10 blur-[100px]" />
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-transparent via-background/20 to-background opacity-80" />
                </div>

                {/* Content */}
                <div className="relative z-10">
                    <div className="flex items-center gap-3 mb-8">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-glow to-emerald-dim flex items-center justify-center glow-emerald">
                            <span className="text-white font-bold text-xl">M</span>
                        </div>
                        <h1 className="text-2xl font-bold tracking-tight font-display">MarketSage</h1>
                    </div>
                </div>

                <div className="relative z-10 max-w-2xl">
                    <h2 className="text-5xl font-bold tracking-tight mb-6 leading-tight font-display text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-white/70">
                        Advanced Analytics for <br />
                        <span className="text-emerald-glow">Intelligent Growth</span>
                    </h2>
                    <p className="text-lg text-muted-foreground leading-relaxed max-w-lg">
                        Access real-time market insights, predictive analytics, and comprehensive reporting tools designed for the modern financial landscape.
                    </p>
                </div>

                <div className="relative z-10 flex items-center gap-6 text-sm text-muted-foreground/60">
                    <span>© {new Date().getFullYear()} MarketSage Inc.</span>
                    <span className="w-1 h-1 rounded-full bg-emerald-dim" />
                    <span>Enterprise Edition</span>
                </div>
            </div>

            {/* Minor Section - Login Form (Golden Ratio ~38%) */}
            <div className="flex-[1] flex items-center justify-center p-8 bg-background border-l border-white/5 relative z-20 shadow-2xl">
                <div className="w-full max-w-md space-y-8 animate-fade-in">
                    <div className="text-center lg:text-left space-y-2">
                        <h3 className="text-3xl font-bold tracking-tight font-display text-white">Welcome back</h3>
                        <p className="text-muted-foreground">Enter your credentials to access the admin panel</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="space-y-2 group">
                            <Label htmlFor="email" className="text-sm font-medium text-muted-foreground group-focus-within:text-emerald-glow transition-colors">Username</Label>
                            <input
                                id="email"
                                type="text"
                                placeholder="admin"
                                className="w-full input-premium bg-white/5 border-white/10 focus:border-emerald-glow/50 text-white placeholder:text-muted-foreground/30"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>

                        <div className="space-y-2 group">
                            <div className="flex items-center justify-between">
                                <Label htmlFor="password" className="text-sm font-medium text-muted-foreground group-focus-within:text-emerald-glow transition-colors">Password</Label>
                                <Link to="#" className="text-xs text-emerald-glow hover:text-emerald-400 hover:underline">Forgot password?</Link>
                            </div>
                            <input
                                id="password"
                                type="password"
                                placeholder="••••••••"
                                className="w-full input-premium bg-white/5 border-white/10 focus:border-emerald-glow/50 text-white placeholder:text-muted-foreground/30"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>

                        <Button
                            type="submit"
                            disabled={isSubmitting}
                            className="w-full h-12 bg-emerald-glow hover:bg-emerald-600 text-white font-medium text-base rounded-xl glow-emerald-subtle hover:glow-emerald transition-all duration-300 transform hover:scale-[1.02]"
                        >
                            {isSubmitting ? 'Authenticating...' : 'Sign In'}
                        </Button>
                    </form>


                </div>
            </div>
        </div>
    );
};

export default Login;
