import { motion } from "framer-motion";
import WireframeChart from "./WireframeChart";

const PHI = 1.618;

const HeroSection = () => {
  return (
    <section
      id="home"
      className="min-h-screen flex items-center relative overflow-hidden"
      style={{
        paddingTop: `${PHI * 6}rem`,
        paddingBottom: `${PHI * 2}rem`
      }}
    >
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-radial from-primary/5 via-transparent to-transparent pointer-events-none" />

      <div className="container mx-auto px-6 h-full flex items-center">
        <div className="flex flex-col lg:flex-row items-center w-full" style={{ gap: `${PHI * 2}rem` }}>

          {/* Left side - Golden Ratio Major (61.8%) */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="lg:w-[61.8%] flex flex-col justify-center"
            style={{ gap: `${PHI}rem` }}
          >
            <div className="space-y-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass"
                style={{ marginBottom: `${PHI}rem` }}
              >
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span className="text-xs font-medium text-primary uppercase tracking-wider">
                  AI-Powered Analytics
                </span>
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.8 }}
                className="font-display font-bold leading-[0.95] tracking-tight"
                style={{ fontSize: `clamp(3rem, ${PHI * 3}vw, 6rem)` }}
              >
                <span className="text-foreground">Predict</span>
                <br />
                <span className="text-foreground/90">the</span>{" "}
                <span className="text-primary text-glow">Unseen</span>
              </motion.h1>
            </div>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-foreground/60 max-w-lg leading-relaxed"
              style={{
                fontSize: `${PHI * 0.75}rem`, // ~1.2rem
                marginBottom: `${PHI}rem`
              }}
            >
              Harness the power of advanced machine learning to anticipate market
              movements with unprecedented accuracy. Your edge in the financial
              frontier.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="flex flex-wrap items-center"
              style={{ gap: `${PHI}rem` }}
            >
              <a
                href="#prediction"
                className="inline-flex items-center justify-center rounded-2xl bg-primary text-primary-foreground font-bold hover:shadow-emerald transition-all duration-300 hover:scale-[1.05] active:scale-[0.98] group relative overflow-hidden"
                style={{
                  height: `${PHI * 2.5}rem`, // ~4rem height (Phi scaled)
                  paddingLeft: `${PHI * 2.5}rem`,
                  paddingRight: `${PHI * 2.5}rem`,
                  fontSize: `${PHI * 0.7}rem`, // ~1.13rem text
                  boxShadow: `0 0 ${PHI * 10}px hsla(160, 84%, 39%, 0.4)` // Glowing aura
                }}
              >
                <div className="absolute inset-0 bg-white/20 translate-y-[100%] group-hover:translate-y-[0%] transition-transform duration-500" />
                <span className="relative z-10 flex items-center gap-2">
                  Start Trading Now
                  <svg
                    className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2.5}
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </span>
              </a>
              <button
                onClick={() => {
                  document.getElementById('reports')?.scrollIntoView({ behavior: 'smooth' });
                }}
                className="inline-flex items-center justify-center rounded-2xl glass text-foreground font-semibold hover:bg-white/10 transition-all duration-300 hover:scale-[1.02]"
                style={{
                  height: `${PHI * 2.5}rem`,
                  paddingLeft: `${PHI * 2}rem`,
                  paddingRight: `${PHI * 2}rem`,
                  fontSize: `${PHI * 0.7}rem`
                }}
              >
                View Analytics
              </button>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="flex pt-8 border-t border-white/10 w-full"
              style={{
                marginTop: `${PHI}rem`,
                gap: `${PHI * 2}rem`
              }}
            >
              {[
                { value: "94.7%", label: "Accuracy Rate" },
                { value: "2.4M+", label: "Predictions" },
                { value: "$847B", label: "Assets Analyzed" },
              ].map((stat, i) => (
                <div key={i} className="space-y-1">
                  <div className="font-display font-bold text-primary" style={{ fontSize: `${PHI}rem` }}>
                    {stat.value}
                  </div>
                  <div className="text-sm text-foreground/50">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* Right side - Golden Ratio Minor (38.2%) */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 1 }}
            className="lg:w-[38.2%] relative flex justify-center items-center"
            style={{ minHeight: `${PHI * 20}rem` }}
          >
            <div className="absolute inset-0 bg-gradient-radial from-primary/20 via-transparent to-transparent blur-3xl" />
            {/* Golden Rectangle Container */}
            <div
              className="w-full glass-strong rounded-3xl border border-white/10 overflow-hidden relative shadow-2xl"
              style={{
                height: `${PHI * 18}rem`,
                maxHeight: '550px'
              }}
            >
              <img
                src={`${import.meta.env.BASE_URL}hero-dashboard.png`}
                alt="Trading Dashboard"
                className="w-full h-full object-cover"
              />
              {/* Overlay gradient for blending */}
              <div className="absolute inset-0 bg-gradient-to-t from-background/20 via-transparent to-transparent" />
            </div>

          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
