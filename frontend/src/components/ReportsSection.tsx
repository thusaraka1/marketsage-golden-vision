import { motion } from "framer-motion";
import { FileText, Download, TrendingUp, BarChart3, PieChart, Activity, Lock } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

const API_URL = 'http://localhost:3001/api';

interface Report {
  id: number;
  title: string;
  category: string;
  date: string;
  description: string;
  icon?: any;
}

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

import { useState, useEffect } from "react";

const ReportsSection = () => {
  const { user, openAuthModal } = useAuth();
  const [reports, setReports] = useState<Report[]>([]);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const response = await fetch(`${API_URL}/reports`);
        const data = await response.json();
        if (Array.isArray(data)) {
          const formattedReports = data.map((r: any) => ({
            id: r.id,
            title: r.name,
            category: r.category,
            date: new Date(r.upload_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
            description: r.description || "In-depth market analysis and insights.",
            icon: getIconForCategory(r.category)
          }));
          setReports(formattedReports);
        }
      } catch (error) {
        console.error('Failed to fetch reports:', error);
      }
    };
    fetchReports();
  }, []);



  return (
    <section id="reports" className="min-h-screen py-32 relative">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
            <span className="text-foreground">Research</span>{" "}
            <span className="text-primary text-glow">Reports</span>
          </h2>
          <p className="text-lg text-foreground/50 max-w-2xl mx-auto">
            In-depth analysis and actionable insights from our team of experts
          </p>
        </motion.div>

        {/* Masonry Grid */}
        <div className="max-w-6xl mx-auto columns-1 md:columns-2 lg:columns-3 gap-6 space-y-6">
          {reports.map((report, index) => (
            <motion.div
              key={report.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="break-inside-avoid group"
            >
              <div className="relative rounded-2xl bg-card border border-white/5 overflow-hidden hover-lift">
                {/* Green top border */}
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-primary/80 to-primary" />

                {/* Content */}
                <div className="p-6 space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="p-3 rounded-xl bg-primary/10">
                      <report.icon className="w-6 h-6 text-primary" />
                    </div>
                    <span className="text-xs text-foreground/40">{report.date}</span>
                  </div>

                  <div>
                    <span className="text-xs font-medium text-primary uppercase tracking-wider">
                      {report.category}
                    </span>
                    <h3 className="font-display text-xl font-bold text-foreground mt-1">
                      {report.title}
                    </h3>
                  </div>

                  <p className="text-sm text-foreground/50 leading-relaxed">
                    {report.description}
                  </p>

                  {user ? (
                    <a
                      href={`${API_URL}/reports/${report.id}/download`}
                      target="_blank"
                      rel="noopener noreferrer"
                      download
                      className="w-full flex items-center justify-center gap-2 py-3 rounded-xl
                                 bg-transparent border border-white/10 text-foreground/50
                                 group-hover:bg-primary/10 group-hover:border-primary/30 
                                 group-hover:text-primary transition-all duration-300"
                    >
                      <Download className="w-4 h-4" />
                      <span className="text-sm font-medium">Download PDF</span>
                    </a>
                  ) : (
                    <button
                      onClick={() => openAuthModal('login')}
                      className="w-full flex items-center justify-center gap-2 py-3 rounded-xl
                                 bg-transparent border border-white/10 text-foreground/50
                                 group-hover:bg-primary/10 group-hover:border-primary/30 
                                 group-hover:text-primary transition-all duration-300"
                    >
                      <Lock size={14} />
                      <span className="text-sm font-medium">Login to Download</span>
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ReportsSection;
