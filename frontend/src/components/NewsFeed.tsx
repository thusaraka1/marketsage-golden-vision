import { motion, AnimatePresence } from "framer-motion";
import { ExternalLink, Clock, TrendingUp, TrendingDown, ArrowRight, RefreshCw } from "lucide-react";
import { useState } from "react";

const newsItems = [
  {
    id: 1,
    time: "2 min ago",
    headline: "Federal Reserve signals potential rate cuts in Q1 2025",
    summary: "The central bank's latest meeting minutes suggest a shift in monetary policy stance as inflation metrics align with long-term targets, sending positive ripples through equity markets.",
    sentiment: "positive",
    source: "Reuters",
    // Switched to a reliable Unsplash image (Stock Charts/Data)
    imageUrl: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop",
    tags: ["Monetary Policy", "USD"]
  },
  {
    id: 2,
    time: "15 min ago",
    headline: "NVIDIA announces next-gen AI accelerator with 3x performance gains",
    sentiment: "positive",
    source: "Bloomberg",
    imageUrl: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1965&auto=format&fit=crop",
  },
  {
    id: 3,
    time: "32 min ago",
    headline: "Treasury yields drop to 6-month low on inflation data",
    sentiment: "positive",
    source: "WSJ",
    imageUrl: "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?q=80&w=1742&auto=format&fit=crop",
  },
  {
    id: 4,
    time: "1 hour ago",
    headline: "China PMI data misses expectations, manufacturing contracts",
    sentiment: "negative",
    source: "CNBC",
    imageUrl: "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=2070&auto=format&fit=crop",
  },
  {
    id: 5,
    time: "2 hours ago",
    headline: "Apple reportedly in talks to expand AI partnerships",
    sentiment: "positive",
    source: "TechCrunch",
    imageUrl: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?q=80&w=2070&auto=format&fit=crop",
  },
];

const NewsFeed = () => {
  const [featuredNews, setFeaturedNews] = useState(newsItems[0]);

  // Filter out the featured news from the sidebar list so it doesn't appear twice
  const sidebarNews = newsItems.filter(item => item.id !== featuredNews.id);

  return (
    <section id="news" className="min-h-screen py-32 relative bg-background">
      {/* Ambient background glow */}
      <div className="absolute top-1/4 right-0 w-[500px] h-[500px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6 border border-primary/20">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            Live Market Updates
          </div>
          <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
            <span className="text-foreground">Global</span>{" "}
            <span className="text-primary text-glow">Intelligence</span>
          </h2>
          <p className="text-lg text-foreground/50 max-w-2xl mx-auto">
            Select any news item to view detailed analysis
          </p>
        </motion.div>

        {/* Golden Ratio Layout */}
        <div className="flex flex-col lg:flex-row gap-8">

          {/* Main Feature - 61.8% */}
          <motion.div
            layout
            className="lg:flex-[1.618] relative group"
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={featuredNews.id}
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.4 }}
                className="relative h-[600px] w-full overflow-hidden rounded-3xl border border-white/10 card-premium p-0 group-hover:border-primary/50 transition-colors duration-500"
              >
                <div className="absolute inset-0">
                  <img
                    src={featuredNews.imageUrl}
                    alt={featuredNews.headline}
                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                    referrerPolicy="no-referrer"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      // Fallback to a safe color or pattern if image fails
                      target.style.display = 'none';
                      target.parentElement!.style.backgroundColor = '#0f172a'; // Deep Void Gunmetal fallback
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background via-background/90 to-transparent" />
                  <div className="absolute inset-0 bg-gradient-to-r from-background/50 to-transparent" />
                </div>

                <div className="absolute bottom-0 left-0 p-8 md:p-12 w-full z-10">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex items-center gap-3 mb-4"
                  >
                    <span className={`pill-${featuredNews.sentiment === 'positive' ? 'bullish' : 'bearish'} flex items-center gap-1.5`}>
                      {featuredNews.sentiment === 'positive' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                      {featuredNews.sentiment === 'positive' ? 'Bullish' : 'Bearish'}
                    </span>
                    <span className="text-emerald-400 text-xs font-semibold px-2 py-1 bg-emerald-950/50 rounded border border-emerald-900/50">
                      FEATURED
                    </span>
                    {featuredNews.tags?.map(tag => (
                      <span key={tag} className="hidden md:inline-block text-white/40 text-xs border border-white/10 px-2 py-1 rounded">
                        {tag}
                      </span>
                    ))}
                  </motion.div>

                  <motion.h3
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="font-display text-3xl md:text-5xl font-bold text-white mb-6 leading-tight group-hover:text-primary transition-colors duration-300"
                  >
                    {featuredNews.headline}
                  </motion.h3>

                  <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="text-lg text-white/70 mb-8 max-w-2xl line-clamp-3 leading-relaxed"
                  >
                    {featuredNews.summary || "Click to read the full analysis and market impact of this developing story."}
                  </motion.p>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="flex items-center justify-between border-t border-white/10 pt-6"
                  >
                    <div className="flex items-center gap-4 text-sm text-white/50">
                      <span className="flex items-center gap-1.5">
                        <Clock size={16} />
                        {featuredNews.time}
                      </span>
                      <span>•</span>
                      <span>{featuredNews.source}</span>
                    </div>

                    <button className="flex items-center gap-2 text-primary font-medium group/btn hover:text-emerald-300 transition-colors">
                      Read Full Story
                      <ArrowRight size={18} className="transition-transform group-hover/btn:translate-x-1" />
                    </button>
                  </motion.div>
                </div>
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Sidebar List - 38.2% */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="lg:flex-1 space-y-4"
          >
            <div className="flex items-center justify-between mb-4 px-2">
              <h4 className="text-xl font-bold font-display text-foreground">Latest Updates</h4>
              <div className="flex gap-2">
                <button className="h-8 w-8 rounded-full border border-white/10 flex items-center justify-center hover:bg-white/5 transition-colors">
                  <RefreshCw size={14} className="text-white/70" />
                </button>
              </div>
            </div>

            <div className="space-y-4 overflow-y-auto pr-2 max-h-[600px] custom-scrollbar">
              {sidebarNews.map((item) => (
                <motion.div
                  layoutId={`news-item-${item.id}`}
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  onClick={() => setFeaturedNews(item)}
                  className="group relative overflow-hidden rounded-xl border border-white/5 bg-white/5 p-4 hover:bg-white/10 hover:border-primary/30 transition-all duration-300 cursor-pointer"
                >
                  <div className="flex gap-4">
                    <div className="h-24 w-24 shrink-0 overflow-hidden rounded-lg bg-white/5 relative">
                      <img
                        src={item.imageUrl}
                        alt={item.headline}
                        className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.style.display = 'none';
                          target.parentElement!.style.backgroundColor = '#1e293b';
                        }}
                      />
                      <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors" />
                    </div>
                    <div className="flex flex-col justify-between flex-1">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className={`text-[10px] uppercase font-bold tracking-wider ${item.sentiment === 'positive' ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                            {item.sentiment}
                          </span>
                          <span className="text-xs text-white/40">{item.time}</span>
                        </div>
                        <h5 className="font-semibold text-white/90 leading-snug group-hover:text-primary transition-colors line-clamp-2">
                          {item.headline}
                        </h5>
                      </div>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-white/40 bg-white/5 px-2 py-0.5 rounded">
                          {item.source}
                        </span>
                        <span className="text-xs text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                          View Details
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}

              <button className="w-full py-4 mt-2 rounded-xl border border-white/10 bg-white/5 text-sm font-medium text-white/70 hover:bg-white/10 hover:text-primary hover:border-primary/30 transition-all flex items-center justify-center gap-2 group">
                View All News
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  );
};

export default NewsFeed;
