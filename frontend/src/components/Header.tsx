import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { User as UserIcon, LogIn, Crown } from "lucide-react";

const navLinks = [
  { name: "Home", href: "#home" },
  { name: "Prediction", href: "#prediction" },
  { name: "About", href: "#info" },
  { name: "Heatmap", href: "#heatmap" },
  { name: "Reports", href: "#reports" },
  { name: "News", href: "#news" },
  { name: "Reviews", href: "#reviews" },
];

const AuthHandler = () => {
  const { user, openAuthModal, logout } = useAuth();

  if (user) {
    return (
      <div className="flex items-center gap-3 ml-4">
        {user.isPro && (
          <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-500/30 text-amber-400 text-xs font-bold uppercase tracking-wider">
            <Crown size={12} />
            <span>PRO</span>
          </div>
        )}
        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <div className="text-sm font-medium text-white">{user.name}</div>
            <button onClick={logout} className="text-xs text-white/50 hover:text-white transition-colors">Sign Out</button>
          </div>
          <div className="w-9 h-9 rounded-full bg-white/10 flex items-center justify-center border border-white/10 text-white">
            <UserIcon size={16} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => openAuthModal('login')}
      className="ml-4 flex items-center gap-2 px-5 py-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-primary/50 text-white font-medium transition-all group"
    >
      <LogIn size={16} className="text-primary group-hover:text-white transition-colors" />
      <span>Login</span>
    </motion.button>
  );
};

const Header = () => {
  const [activeSection, setActiveSection] = useState("home");

  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: "-20% 0px -80% 0px", // Highlighting kicks in when section is near top
      threshold: 0
    };

    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    navLinks.forEach((link) => {
      const element = document.querySelector(link.href);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50 frosted-header"
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="flex items-center gap-2"
          >
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center glow-emerald-subtle">
              <svg
                viewBox="0 0 24 24"
                className="w-5 h-5 text-primary"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M3 3v18h18" />
                <path d="M7 16l4-6 4 4 6-8" />
              </svg>
            </div>
            <span className="font-display text-xl font-bold text-foreground">
              MarketSage
            </span>
          </motion.div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {navLinks.map((link, index) => {
              const sectionId = link.href.substring(1);
              const isActive = activeSection === sectionId;

              const handleClick = () => {
                const element = document.getElementById(sectionId);
                if (element) {
                  element.scrollIntoView({ behavior: 'smooth' });
                }
              };

              return (
                <motion.button
                  key={link.name}
                  onClick={handleClick}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index, duration: 0.4 }}
                  className={`text-sm font-medium transition-colors duration-300 relative group ${isActive ? "text-primary" : "text-foreground/70 hover:text-primary"
                    }`}
                >
                  {link.name}
                  <span className={`absolute -bottom-1 left-0 h-0.5 bg-primary transition-all duration-300 ${isActive ? "w-full" : "w-0 group-hover:w-full"
                    }`} />
                </motion.button>
              )
            })}
          </nav>

          {/* Live Indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex items-center gap-2"
          >
            <div className="w-2.5 h-2.5 rounded-full bg-primary pulse-live" />
            <span className="text-xs font-medium text-primary uppercase tracking-wider">
              Live
            </span>
          </motion.div>

          {/* Auth Button */}
          <AuthHandler />

        </div>
      </div>
    </motion.header>
  );
};

export default Header;
