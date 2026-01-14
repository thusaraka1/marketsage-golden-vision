import { motion } from "framer-motion";

const heatmapData = [
  { symbol: "AAPL", change: 2.4, size: "large" },
  { symbol: "MSFT", change: 1.8, size: "large" },
  { symbol: "GOOGL", change: 0.9, size: "medium" },
  { symbol: "AMZN", change: -0.5, size: "medium" },
  { symbol: "NVDA", change: 5.2, size: "large" },
  { symbol: "META", change: 1.2, size: "medium" },
  { symbol: "TSLA", change: -2.1, size: "medium" },
  { symbol: "BRK.B", change: 0.3, size: "small" },
  { symbol: "JPM", change: 1.5, size: "small" },
  { symbol: "V", change: 0.8, size: "small" },
  { symbol: "UNH", change: -0.3, size: "small" },
  { symbol: "HD", change: 1.1, size: "small" },
  { symbol: "MA", change: 0.6, size: "small" },
  { symbol: "PG", change: 0.2, size: "small" },
  { symbol: "DIS", change: -1.8, size: "small" },
  { symbol: "PYPL", change: 3.1, size: "small" },
  { symbol: "NFLX", change: 2.8, size: "medium" },
  { symbol: "ADBE", change: 1.4, size: "small" },
  { symbol: "CRM", change: 0.7, size: "small" },
  { symbol: "INTC", change: -0.9, size: "small" },
];

const getColorClass = (change: number) => {
  if (change >= 3) return "bg-primary/90";
  if (change >= 1.5) return "bg-primary/70";
  if (change >= 0) return "bg-primary/40";
  if (change >= -1) return "bg-white/10";
  return "bg-destructive/50";
};

const getSizeClass = (size: string) => {
  switch (size) {
    case "large":
      return "col-span-2 row-span-2";
    case "medium":
      return "col-span-2 row-span-1";
    default:
      return "col-span-1 row-span-1";
  }
};

const HeatmapSection = () => {
  return (
    <section id="heatmap" className="min-h-screen py-32 relative">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
            <span className="text-foreground">Market</span>{" "}
            <span className="text-primary text-glow">Heatmap</span>
          </h2>
          <p className="text-lg text-foreground/50 max-w-2xl mx-auto">
            Real-time visualization of market movements across major sectors
          </p>
        </motion.div>

        {/* Legend */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="flex justify-center gap-6 mb-12 text-sm"
        >
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-primary/90" />
            <span className="text-foreground/60">Strong Growth</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-primary/40" />
            <span className="text-foreground/60">Moderate</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-white/10" />
            <span className="text-foreground/60">Neutral</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-destructive/50" />
            <span className="text-foreground/60">Decline</span>
          </div>
        </motion.div>

        {/* Heatmap Grid */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="max-w-5xl mx-auto"
        >
          <div className="grid grid-cols-6 md:grid-cols-8 gap-2 p-6 rounded-2xl glass">
            {heatmapData.map((item, index) => (
              <motion.div
                key={item.symbol}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 + index * 0.03 }}
                whileHover={{ scale: 1.05, zIndex: 10 }}
                className={`
                  ${getSizeClass(item.size)}
                  ${getColorClass(item.change)}
                  rounded-lg p-4 flex flex-col justify-between
                  cursor-pointer transition-all duration-300
                  hover:shadow-lg hover:shadow-primary/20
                  border border-white/5 hover:border-primary/30
                `}
              >
                <span className="font-display font-bold text-foreground text-sm md:text-base">
                  {item.symbol}
                </span>
                <span
                  className={`text-xs md:text-sm font-medium ${
                    item.change >= 0 ? "text-primary" : "text-destructive"
                  }`}
                >
                  {item.change >= 0 ? "+" : ""}
                  {item.change.toFixed(1)}%
                </span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Market Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="flex justify-center gap-8 mt-12"
        >
          {[
            { label: "S&P 500", value: "+1.24%", positive: true },
            { label: "NASDAQ", value: "+1.89%", positive: true },
            { label: "DOW", value: "+0.67%", positive: true },
          ].map((index) => (
            <div key={index.label} className="text-center">
              <div className="text-sm text-foreground/50 mb-1">{index.label}</div>
              <div
                className={`font-display font-bold text-xl ${
                  index.positive ? "text-primary" : "text-destructive"
                }`}
              >
                {index.value}
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default HeatmapSection;
