import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import {
    ArrowLeft, Users, FileText, CreditCard, Upload, Trash2,
    Download, Search, MoreVertical, Shield, Eye, Ban,
    DollarSign, Save, Plus, BarChart3, PieChart, TrendingUp, Activity, LogOut
} from "lucide-react";

const PHI = 1.618;
const API_URL = 'http://localhost:3001/api';

// Reports with description
interface Report {
    id: number;
    name: string;
    category: string;
    description?: string;
    size: string;
    uploadDate: string;
    downloads: number;
}


type TabType = "users" | "reports" | "subscription";

const AdminPanel = () => {
    const navigate = useNavigate();
    const { logout } = useAuth();

    const [activeTab, setActiveTab] = useState<TabType>("users");
    const [searchQuery, setSearchQuery] = useState("");
    const [users, setUsers] = useState<Array<{ id: number; name: string; email: string; plan: string; status: string; joined: string }>>([]);
    const [reports, setReports] = useState<Report[]>([]);
    const [isLoadingUsers, setIsLoadingUsers] = useState(true);
    const [subscriptionPrice, setSubscriptionPrice] = useState("29.99");
    const [selectedCurrency, setSelectedCurrency] = useState("USD");
    const [isSavingSubscription, setIsSavingSubscription] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [newReportName, setNewReportName] = useState("");
    const [newReportCategory, setNewReportCategory] = useState("");
    const [newReportDescription, setNewReportDescription] = useState("");
    const [isUploading, setIsUploading] = useState(false);

    // Handle Super Admin logout
    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const currencies = [
        // Major Currencies
        { code: "USD", symbol: "$", name: "US Dollar" },
        { code: "EUR", symbol: "€", name: "Euro" },
        { code: "GBP", symbol: "£", name: "British Pound" },
        { code: "JPY", symbol: "¥", name: "Japanese Yen" },
        { code: "CHF", symbol: "Fr", name: "Swiss Franc" },
        { code: "CNY", symbol: "¥", name: "Chinese Yuan" },
        // Asia & Pacific
        { code: "AUD", symbol: "A$", name: "Australian Dollar" },
        { code: "NZD", symbol: "NZ$", name: "New Zealand Dollar" },
        { code: "HKD", symbol: "HK$", name: "Hong Kong Dollar" },
        { code: "SGD", symbol: "S$", name: "Singapore Dollar" },
        { code: "INR", symbol: "₹", name: "Indian Rupee" },
        { code: "LKR", symbol: "Rs.", name: "Sri Lankan Rupee" },
        { code: "PKR", symbol: "₨", name: "Pakistani Rupee" },
        { code: "BDT", symbol: "৳", name: "Bangladeshi Taka" },
        { code: "NPR", symbol: "₨", name: "Nepalese Rupee" },
        { code: "MMK", symbol: "K", name: "Myanmar Kyat" },
        { code: "THB", symbol: "฿", name: "Thai Baht" },
        { code: "VND", symbol: "₫", name: "Vietnamese Dong" },
        { code: "MYR", symbol: "RM", name: "Malaysian Ringgit" },
        { code: "IDR", symbol: "Rp", name: "Indonesian Rupiah" },
        { code: "PHP", symbol: "₱", name: "Philippine Peso" },
        { code: "KRW", symbol: "₩", name: "South Korean Won" },
        { code: "TWD", symbol: "NT$", name: "Taiwan Dollar" },
        // Middle East
        { code: "AED", symbol: "د.إ", name: "UAE Dirham" },
        { code: "SAR", symbol: "﷼", name: "Saudi Riyal" },
        { code: "QAR", symbol: "﷼", name: "Qatari Riyal" },
        { code: "KWD", symbol: "د.ك", name: "Kuwaiti Dinar" },
        { code: "BHD", symbol: "ب.د", name: "Bahraini Dinar" },
        { code: "OMR", symbol: "ر.ع.", name: "Omani Rial" },
        { code: "JOD", symbol: "د.ا", name: "Jordanian Dinar" },
        { code: "ILS", symbol: "₪", name: "Israeli Shekel" },
        { code: "TRY", symbol: "₺", name: "Turkish Lira" },
        { code: "IRR", symbol: "﷼", name: "Iranian Rial" },
        { code: "IQD", symbol: "ع.د", name: "Iraqi Dinar" },
        { code: "LBP", symbol: "ل.ل", name: "Lebanese Pound" },
        { code: "SYP", symbol: "£", name: "Syrian Pound" },
        // Africa
        { code: "ZAR", symbol: "R", name: "South African Rand" },
        { code: "EGP", symbol: "E£", name: "Egyptian Pound" },
        { code: "NGN", symbol: "₦", name: "Nigerian Naira" },
        { code: "KES", symbol: "KSh", name: "Kenyan Shilling" },
        { code: "GHS", symbol: "₵", name: "Ghanaian Cedi" },
        { code: "MAD", symbol: "د.م.", name: "Moroccan Dirham" },
        { code: "TND", symbol: "د.ت", name: "Tunisian Dinar" },
        { code: "DZD", symbol: "د.ج", name: "Algerian Dinar" },
        { code: "TZS", symbol: "TSh", name: "Tanzanian Shilling" },
        { code: "UGX", symbol: "USh", name: "Ugandan Shilling" },
        { code: "ETB", symbol: "Br", name: "Ethiopian Birr" },
        { code: "XOF", symbol: "CFA", name: "West African CFA Franc" },
        { code: "XAF", symbol: "FCFA", name: "Central African CFA Franc" },
        // Americas
        { code: "CAD", symbol: "C$", name: "Canadian Dollar" },
        { code: "MXN", symbol: "$", name: "Mexican Peso" },
        { code: "BRL", symbol: "R$", name: "Brazilian Real" },
        { code: "ARS", symbol: "$", name: "Argentine Peso" },
        { code: "CLP", symbol: "$", name: "Chilean Peso" },
        { code: "COP", symbol: "$", name: "Colombian Peso" },
        { code: "PEN", symbol: "S/", name: "Peruvian Sol" },
        { code: "UYU", symbol: "$U", name: "Uruguayan Peso" },
        { code: "VES", symbol: "Bs.", name: "Venezuelan Bolívar" },
        { code: "BOB", symbol: "Bs.", name: "Bolivian Boliviano" },
        { code: "PYG", symbol: "₲", name: "Paraguayan Guarani" },
        { code: "JMD", symbol: "J$", name: "Jamaican Dollar" },
        { code: "TTD", symbol: "TT$", name: "Trinidad Dollar" },
        { code: "DOP", symbol: "RD$", name: "Dominican Peso" },
        { code: "CRC", symbol: "₡", name: "Costa Rican Colón" },
        { code: "GTQ", symbol: "Q", name: "Guatemalan Quetzal" },
        { code: "HNL", symbol: "L", name: "Honduran Lempira" },
        { code: "NIO", symbol: "C$", name: "Nicaraguan Córdoba" },
        { code: "PAB", symbol: "B/.", name: "Panamanian Balboa" },
        // Europe
        { code: "SEK", symbol: "kr", name: "Swedish Krona" },
        { code: "NOK", symbol: "kr", name: "Norwegian Krone" },
        { code: "DKK", symbol: "kr", name: "Danish Krone" },
        { code: "PLN", symbol: "zł", name: "Polish Zloty" },
        { code: "CZK", symbol: "Kč", name: "Czech Koruna" },
        { code: "HUF", symbol: "Ft", name: "Hungarian Forint" },
        { code: "RON", symbol: "lei", name: "Romanian Leu" },
        { code: "BGN", symbol: "лв", name: "Bulgarian Lev" },
        { code: "HRK", symbol: "kn", name: "Croatian Kuna" },
        { code: "RSD", symbol: "дин.", name: "Serbian Dinar" },
        { code: "UAH", symbol: "₴", name: "Ukrainian Hryvnia" },
        { code: "RUB", symbol: "₽", name: "Russian Ruble" },
        { code: "BYN", symbol: "Br", name: "Belarusian Ruble" },
        { code: "ISK", symbol: "kr", name: "Icelandic Króna" },
        { code: "GEL", symbol: "₾", name: "Georgian Lari" },
        { code: "AMD", symbol: "֏", name: "Armenian Dram" },
        { code: "AZN", symbol: "₼", name: "Azerbaijani Manat" },
        { code: "KZT", symbol: "₸", name: "Kazakhstani Tenge" },
        { code: "UZS", symbol: "сўм", name: "Uzbekistani Som" },
        // Caribbean & Others
        { code: "BSD", symbol: "$", name: "Bahamian Dollar" },
        { code: "BBD", symbol: "$", name: "Barbadian Dollar" },
        { code: "BZD", symbol: "BZ$", name: "Belize Dollar" },
        { code: "XCD", symbol: "$", name: "East Caribbean Dollar" },
        { code: "FJD", symbol: "FJ$", name: "Fijian Dollar" },
        { code: "PGK", symbol: "K", name: "Papua New Guinean Kina" },
        { code: "WST", symbol: "T", name: "Samoan Tala" },
        { code: "TOP", symbol: "T$", name: "Tongan Paʻanga" },
        { code: "VUV", symbol: "VT", name: "Vanuatu Vatu" },
        // Crypto-friendly
        { code: "BTC", symbol: "₿", name: "Bitcoin" },
        { code: "ETH", symbol: "Ξ", name: "Ethereum" },
    ];

    const currentCurrency = currencies.find(c => c.code === selectedCurrency) || currencies[0];

    // Fetch users from API
    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const response = await fetch(`${API_URL}/admin/users`);
                const data = await response.json();
                if (data.users) {
                    setUsers(data.users);
                }
            } catch (error) {
                console.error('Failed to fetch users:', error);
            } finally {
                setIsLoadingUsers(false);
            }
        };
        fetchUsers();
    }, []);

    // Fetch subscription settings from API
    useEffect(() => {
        const fetchSubscriptionSettings = async () => {
            try {
                const response = await fetch(`${API_URL}/settings/subscription`);
                const data = await response.json();
                if (data.price) setSubscriptionPrice(data.price);
                if (data.currency) setSelectedCurrency(data.currency);
            } catch (error) {
                console.error('Failed to fetch subscription settings:', error);
            }
        };
        fetchSubscriptionSettings();
    }, []);

    // Fetch reports from API
    useEffect(() => {
        const fetchReports = async () => {
            try {
                const response = await fetch(`${API_URL}/reports`);
                const data = await response.json();
                if (Array.isArray(data)) {
                    // Map API response to frontend format if needed
                    const formattedReports = data.map((r: any) => ({
                        id: r.id,
                        name: r.name,
                        category: r.category,
                        description: r.description,
                        size: r.size,
                        uploadDate: new Date(r.upload_date).toISOString().split('T')[0],
                        downloads: r.downloads
                    }));
                    setReports(formattedReports);
                }
            } catch (error) {
                console.error('Failed to fetch reports:', error);
            }
        };
        fetchReports();
    }, []);

    const tabs = [
        { id: "users" as TabType, label: "Users", icon: Users, count: users.length },
        { id: "reports" as TabType, label: "Reports", icon: FileText, count: reports.length },
        { id: "subscription" as TabType, label: "Subscription", icon: CreditCard },
    ];

    const filteredUsers = users.filter(
        (u) =>
            u.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            u.email.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleDeleteUser = async (id: number) => {
        try {
            await fetch(`${API_URL}/admin/users/${id}`, { method: 'DELETE' });
            setUsers(users.filter((u) => u.id !== id));
        } catch (error) {
            console.error('Failed to delete user:', error);
        }
    };

    const handleToggleStatus = async (id: number) => {
        try {
            const response = await fetch(`${API_URL}/admin/users/${id}/status`, { method: 'PUT' });
            const data = await response.json();
            setUsers(
                users.map((u) =>
                    u.id === id ? { ...u, status: data.status } : u
                )
            );
        } catch (error) {
            console.error('Failed to toggle status:', error);
        }
    };

    const handleSaveSubscription = async () => {
        setIsSavingSubscription(true);
        try {
            const response = await fetch(`${API_URL}/settings/subscription`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ price: subscriptionPrice, currency: selectedCurrency }),
            });
            if (response.ok) {
                alert('Subscription settings saved successfully!');
            }
        } catch (error) {
            console.error('Failed to save subscription settings:', error);
            alert('Failed to save settings');
        } finally {
            setIsSavingSubscription(false);
        }
    };

    // Icon mapping based on category
    const getIconForCategory = (category: string) => {
        switch (category) {
            case "Quarterly Report": return BarChart3;
            case "Market Analysis": return Activity;
            case "Investment Strategy": return PieChart;
            case "Sector Report": return TrendingUp;
            case "Risk Analysis": return FileText;
            case "Annual Outlook": return TrendingUp;
            default: return FileText;
        }
    };

    const handleUploadReport = async () => {
        if (selectedFile && newReportName && newReportCategory) {
            setIsUploading(true);
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('name', newReportName);
                formData.append('category', newReportCategory);
                formData.append('description', newReportDescription);

                const response = await fetch(`${API_URL}/reports`, {
                    method: 'POST',
                    body: formData, // Browser sets Content-Type to multipart/form-data
                });

                if (response.ok) {
                    const newReport = await response.json();
                    setReports([newReport, ...reports]);
                    setNewReportName("");
                    setNewReportCategory("Market Analysis");
                    setNewReportDescription("");
                    setSelectedFile(null);
                    // Reset file input
                    const fileInput = document.getElementById('file-upload') as HTMLInputElement;
                    if (fileInput) fileInput.value = '';
                } else {
                    console.error('Failed to upload report');
                    alert('Failed to upload report');
                }
            } catch (error) {
                console.error('Error uploading report:', error);
                alert('Error uploading report');
            } finally {
                setIsUploading(false);
            }
        }
    };

    const handleDeleteReport = async (id: number) => {
        try {
            const response = await fetch(`${API_URL}/reports/${id}`, { method: 'DELETE' });
            if (response.ok) {
                setReports(reports.filter((r) => r.id !== id));
            }
        } catch (error) {
            console.error('Failed to delete report:', error);
        }
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
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <Shield className="w-5 h-5 text-primary" />
                            <span className="font-semibold text-foreground">Admin Panel</span>
                        </div>
                        <button
                            onClick={handleLogout}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors font-medium text-sm"
                        >
                            <LogOut className="w-4 h-4" />
                            Logout
                        </button>
                    </div>
                </div>
            </header>

            {/* Content */}
            <main className="pt-32 pb-24">
                <div className="container mx-auto px-6">
                    {/* Header */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8"
                    >
                        <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-2">
                            Admin <span className="text-primary">Dashboard</span>
                        </h1>
                        <p className="text-foreground/50">Manage users, reports, and subscription settings</p>
                    </motion.div>

                    {/* Tabs */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="flex gap-2 mb-8"
                    >
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${activeTab === tab.id
                                    ? "bg-primary text-primary-foreground"
                                    : "glass text-foreground/70 hover:text-foreground"
                                    }`}
                            >
                                <tab.icon className="w-5 h-5" />
                                {tab.label}
                                {tab.count !== undefined && (
                                    <span className={`px-2 py-0.5 rounded-full text-xs ${activeTab === tab.id ? "bg-white/20" : "bg-primary/20 text-primary"
                                        }`}>
                                        {tab.count}
                                    </span>
                                )}
                            </button>
                        ))}
                    </motion.div>

                    {/* Users Tab */}
                    {activeTab === "users" && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="space-y-6"
                        >
                            {/* Search */}
                            <div className="relative max-w-md">
                                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground/50" />
                                <input
                                    type="text"
                                    placeholder="Search users..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full input-premium pl-12"
                                />
                            </div>

                            {/* Users Table */}
                            <div className="card-premium overflow-hidden">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-white/10">
                                            <th className="text-left px-6 py-4 text-foreground/50 font-medium">User</th>
                                            <th className="text-left px-6 py-4 text-foreground/50 font-medium">Plan</th>
                                            <th className="text-left px-6 py-4 text-foreground/50 font-medium">Status</th>
                                            <th className="text-left px-6 py-4 text-foreground/50 font-medium">Joined</th>
                                            <th className="text-right px-6 py-4 text-foreground/50 font-medium">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredUsers.map((user) => (
                                            <tr key={user.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                                <td className="px-6 py-4">
                                                    <div>
                                                        <div className="font-medium text-foreground">{user.name}</div>
                                                        <div className="text-sm text-foreground/50">{user.email}</div>
                                                    </div>
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${user.plan === "Premium"
                                                        ? "bg-primary/20 text-primary"
                                                        : "bg-white/10 text-foreground/70"
                                                        }`}>
                                                        {user.plan}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${user.status === "Active"
                                                        ? "bg-green-500/20 text-green-400"
                                                        : "bg-red-500/20 text-red-400"
                                                        }`}>
                                                        {user.status}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-foreground/70">{user.joined}</td>
                                                <td className="px-6 py-4">
                                                    <div className="flex items-center justify-end gap-2">
                                                        <button
                                                            onClick={() => handleToggleStatus(user.id)}
                                                            className="p-2 rounded-lg hover:bg-white/10 transition-colors text-foreground/50 hover:text-foreground"
                                                            title={user.status === "Active" ? "Suspend" : "Activate"}
                                                        >
                                                            {user.status === "Active" ? <Ban className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                                        </button>
                                                        <button
                                                            onClick={() => handleDeleteUser(user.id)}
                                                            className="p-2 rounded-lg hover:bg-red-500/20 transition-colors text-foreground/50 hover:text-red-400"
                                                            title="Delete"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </motion.div>
                    )}

                    {/* Reports Tab */}
                    {activeTab === "reports" && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="space-y-6"
                        >
                            {/* Upload Section */}
                            <div className="card-premium" style={{ padding: `${PHI * 1.5}rem` }}>
                                <h3 className="text-xl font-display font-bold text-foreground mb-6 flex items-center gap-2">
                                    <Upload className="w-5 h-5 text-primary" />
                                    Upload New Report
                                </h3>
                                <div className="space-y-4 mb-6">
                                    <div className="grid md:grid-cols-2 gap-4">
                                        <input
                                            type="text"
                                            placeholder="Report Name"
                                            value={newReportName}
                                            onChange={(e) => setNewReportName(e.target.value)}
                                            className="w-full input-premium"
                                        />
                                        <select
                                            value={newReportCategory}
                                            onChange={(e) => setNewReportCategory(e.target.value)}
                                            className="w-full input-premium"
                                        >
                                            <option value="">Select Category</option>
                                            <option value="Quarterly Report">Quarterly Report</option>
                                            <option value="Investment Strategy">Investment Strategy</option>
                                            <option value="Risk Analysis">Risk Analysis</option>
                                            <option value="Market Analysis">Market Analysis</option>
                                            <option value="Annual Outlook">Annual Outlook</option>
                                            <option value="Sector Report">Sector Report</option>
                                        </select>
                                    </div>
                                    <textarea
                                        placeholder="Report Description (Optional)"
                                        value={newReportDescription}
                                        onChange={(e) => setNewReportDescription(e.target.value)}
                                        className="w-full input-premium min-h-[100px] resize-none"
                                    />
                                    <div className="relative">
                                        <input
                                            type="file"
                                            accept=".pdf"
                                            onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                                            className="absolute inset-0 opacity-0 cursor-pointer"
                                        />
                                        <div className="input-premium flex items-center gap-2 cursor-pointer bg-white/5 border-dashed">
                                            <FileText className="w-5 h-5 text-foreground/50" />
                                            <span className="text-foreground/50 truncate">
                                                {selectedFile ? selectedFile.name : "Choose PDF file..."}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={handleUploadReport}
                                    disabled={!selectedFile || !newReportName || !newReportCategory}
                                    className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-bold transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                                >
                                    <Plus className="w-5 h-5" />
                                    Upload Report
                                </button>
                            </div>

                            {/* Reports List - Premium Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                                {reports.map((report) => {
                                    const Icon = getIconForCategory(report.category);
                                    return (
                                        <motion.div
                                            layout
                                            initial={{ opacity: 0, scale: 0.9 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            key={report.id}
                                            className="group relative overflow-hidden rounded-2xl bg-white/5 border border-white/10 hover:border-primary/50 transition-all duration-500 hover:shadow-[0_0_30px_rgba(56,189,248,0.1)] hover:-translate-y-1"
                                        >
                                            {/* Green top border */}
                                            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-primary/80 to-primary" />

                                            <div className="p-6 h-full flex flex-col space-y-4">
                                                <div className="flex items-start justify-between">
                                                    <div className="p-3 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors duration-300">
                                                        <Icon className="w-6 h-6 text-primary" />
                                                    </div>
                                                    <div className="flex flex-col items-end gap-1">
                                                        <span className="text-xs text-foreground/40">{report.uploadDate}</span>
                                                        <button
                                                            onClick={() => handleDeleteReport(report.id)}
                                                            className="p-1.5 rounded-lg hover:bg-red-500/20 text-foreground/30 hover:text-red-400 transition-colors"
                                                            title="Delete Report"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </div>

                                                <div>
                                                    <span className="text-xs font-medium text-primary uppercase tracking-wider block mb-1">
                                                        {report.category}
                                                    </span>
                                                    <h4 className="font-display font-bold text-xl text-foreground group-hover:text-primary transition-colors">
                                                        {report.name}
                                                    </h4>
                                                </div>

                                                {report.description && (
                                                    <p className="text-sm text-foreground/60 leading-relaxed line-clamp-3 flex-grow">
                                                        {report.description}
                                                    </p>
                                                )}

                                                <div className="pt-4 border-t border-white/5 flex items-center justify-between text-sm text-foreground/50 mt-auto">
                                                    <span className="font-medium">{report.size}</span>
                                                    <div className="flex items-center gap-2">
                                                        <a
                                                            href={`${API_URL}/reports/${report.id}/download`}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="flex items-center gap-1 hover:text-primary transition-colors"
                                                            download
                                                        >
                                                            <Download className="w-4 h-4" />
                                                            <span>{report.downloads} Downloads</span>
                                                        </a>
                                                    </div>
                                                </div>
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}

                    {/* Subscription Tab */}
                    {activeTab === "subscription" && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="max-w-2xl"
                        >
                            <div className="card-premium" style={{ padding: `${PHI * 2}rem` }}>
                                <h3 className="text-xl font-display font-bold text-foreground mb-6 flex items-center gap-2">
                                    <DollarSign className="w-5 h-5 text-primary" />
                                    Subscription Settings
                                </h3>

                                <div className="space-y-6">
                                    {/* Currency Selector */}
                                    <div>
                                        <label className="block text-sm font-medium text-foreground/70 mb-2">
                                            Currency
                                        </label>
                                        <select
                                            value={selectedCurrency}
                                            onChange={(e) => setSelectedCurrency(e.target.value)}
                                            className="w-full input-premium"
                                        >
                                            {currencies.map((currency) => (
                                                <option key={currency.code} value={currency.code}>
                                                    {currency.code} - {currency.name} ({currency.symbol})
                                                </option>
                                            ))}
                                        </select>
                                    </div>

                                    {/* Monthly Price */}
                                    <div>
                                        <label className="block text-sm font-medium text-foreground/70 mb-2">
                                            Monthly Subscription Price
                                        </label>
                                        <div className="relative">
                                            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-primary font-bold text-lg">
                                                {currentCurrency.symbol}
                                            </span>
                                            <input
                                                type="text"
                                                inputMode="decimal"
                                                pattern="[0-9]*\.?[0-9]*"
                                                value={subscriptionPrice}
                                                onChange={(e) => {
                                                    const value = e.target.value;
                                                    if (/^\d*\.?\d*$/.test(value)) {
                                                        setSubscriptionPrice(value);
                                                    }
                                                }}
                                                className="w-full input-premium pl-12 text-2xl font-bold"
                                            />
                                        </div>
                                    </div>

                                    {/* Preview Card */}
                                    <div className="p-6 rounded-xl bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20">
                                        <div className="text-sm text-primary font-medium mb-2">PREMIUM PLAN</div>
                                        <div className="flex items-baseline gap-1 mb-4">
                                            <span className="text-4xl font-display font-bold text-foreground">
                                                {currentCurrency.symbol}{subscriptionPrice}
                                            </span>
                                            <span className="text-foreground/50">/{selectedCurrency}/month</span>
                                        </div>
                                        <ul className="space-y-2 text-foreground/70 text-sm">
                                            <li className="flex items-center gap-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                                Unlimited AI predictions
                                            </li>
                                            <li className="flex items-center gap-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                                Access to all reports
                                            </li>
                                            <li className="flex items-center gap-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                                Priority support
                                            </li>
                                            <li className="flex items-center gap-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                                Real-time alerts
                                            </li>
                                        </ul>
                                    </div>

                                    {/* Save Button */}
                                    <button
                                        onClick={handleSaveSubscription}
                                        disabled={isSavingSubscription}
                                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-bold transition-all duration-300 hover:scale-[1.02] disabled:opacity-50"
                                    >
                                        <Save className="w-5 h-5" />
                                        {isSavingSubscription ? 'Saving...' : 'Save Changes'}
                                    </button>
                                </div>
                            </div>

                            {/* Stats */}
                            <div className="grid grid-cols-3 gap-4 mt-6">
                                {[
                                    { label: "Total Subscribers", value: "1,247" },
                                    { label: "Monthly Revenue", value: "$37,284" },
                                    { label: "Churn Rate", value: "2.3%" },
                                ].map((stat) => (
                                    <div key={stat.label} className="card-premium text-center" style={{ padding: `${PHI}rem` }}>
                                        <div className="text-2xl font-display font-bold text-primary">{stat.value}</div>
                                        <div className="text-sm text-foreground/50">{stat.label}</div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default AdminPanel;
