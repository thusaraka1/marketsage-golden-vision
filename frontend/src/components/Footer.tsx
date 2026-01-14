import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="py-16 border-t border-white/5">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-8">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="flex items-center gap-2"
          >
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
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

          {/* Links */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="flex items-center gap-8 text-sm text-foreground/50"
          >
            <Link to="/privacy" className="hover:text-primary transition-colors">
              Privacy Policy
            </Link>
            <Link to="/terms" className="hover:text-primary transition-colors">
              Terms of Service
            </Link>
            <Link to="/api-docs" className="hover:text-primary transition-colors">
              API Documentation
            </Link>
            <Link to="/contact" className="hover:text-primary transition-colors">
              Contact
            </Link>
          </motion.div>

          {/* Copyright */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-sm text-foreground/30"
          >
            © 2025 MarketSage. All rights reserved.
          </motion.div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

