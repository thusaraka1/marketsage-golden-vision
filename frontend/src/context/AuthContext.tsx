import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { toast } from 'sonner';

const API_URL = 'http://localhost:3001/api';

interface User {
    id?: number;
    name: string;
    email: string;
    isPro: boolean;
    isAdmin?: boolean;  // True only for Super Admins from 'admins' table
}

interface AuthContextType {
    user: User | null;
    isLoading: boolean;
    login: (email: string, password: string) => Promise<void>;
    adminLogin: (username: string, password: string) => Promise<void>;
    register: (name: string, email: string, password: string) => Promise<void>;
    logout: () => void;
    upgradeToPro: () => Promise<void>;
    isAuthModalOpen: boolean;
    authModalView: 'login' | 'signup' | 'payment';
    openAuthModal: (view?: 'login' | 'signup' | 'payment') => void;
    closeAuthModal: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
    const [authModalView, setAuthModalView] = useState<'login' | 'signup' | 'payment'>('login');

    useEffect(() => {
        // Check local storage for persisted session
        const storedUser = localStorage.getItem('marketsage_user');
        const token = localStorage.getItem('marketsage_token');
        if (storedUser && token) {
            setUser(JSON.parse(storedUser));
        }
        setIsLoading(false);
    }, []);

    const login = async (email: string, password: string) => {
        try {
            const response = await fetch(`${API_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Login failed');
            }

            setUser(data.user);
            localStorage.setItem('marketsage_user', JSON.stringify(data.user));
            localStorage.setItem('marketsage_token', data.token);
            toast.success(`Welcome back, ${data.user.name}`);
            closeAuthModal();
        } catch (error) {
            toast.error(error instanceof Error ? error.message : 'Login failed');
            throw error;
        }
    };

    // Admin login - calls /api/auth/admin-login (Super Admin only)
    const adminLogin = async (username: string, password: string) => {
        try {
            const response = await fetch(`${API_URL}/auth/admin-login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Admin login failed');
            }

            setUser(data.user);
            localStorage.setItem('marketsage_user', JSON.stringify(data.user));
            localStorage.setItem('marketsage_token', data.token);
            toast.success('Welcome, Super Admin!');
        } catch (error) {
            toast.error(error instanceof Error ? error.message : 'Admin login failed');
            throw error;
        }
    };

    const register = async (name: string, email: string, password: string) => {
        try {
            const response = await fetch(`${API_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Registration failed');
            }

            setUser(data.user);
            localStorage.setItem('marketsage_user', JSON.stringify(data.user));
            localStorage.setItem('marketsage_token', data.token);
            toast.success(`Welcome, ${data.user.name}!`);
            closeAuthModal();
        } catch (error) {
            toast.error(error instanceof Error ? error.message : 'Registration failed');
            throw error;
        }
    };

    const logout = () => {
        setUser(null);
        localStorage.removeItem('marketsage_user');
        localStorage.removeItem('marketsage_token');
        toast.info('Logged out successfully');
    };

    const upgradeToPro = async () => {
        if (!user) return;

        try {
            const response = await fetch(`${API_URL}/auth/upgrade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: user.id }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upgrade failed');
            }

            const updatedUser = { ...user, isPro: true };
            setUser(updatedUser);
            localStorage.setItem('marketsage_user', JSON.stringify(updatedUser));
            toast.success("Welcome to MarketSage Pro!");
            closeAuthModal();
        } catch (error) {
            toast.error(error instanceof Error ? error.message : 'Upgrade failed');
            throw error;
        }
    };

    const openAuthModal = (view: 'login' | 'signup' | 'payment' = 'login') => {
        setAuthModalView(view);
        setIsAuthModalOpen(true);
    };

    const closeAuthModal = () => {
        setIsAuthModalOpen(false);
    };

    return (
        <AuthContext.Provider value={{
            user,
            isLoading,
            login,
            adminLogin,
            register,
            logout,
            upgradeToPro,
            isAuthModalOpen,
            authModalView,
            openAuthModal,
            closeAuthModal
        }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
