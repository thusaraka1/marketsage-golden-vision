/**
 * ================================================================================
 * PredictionSection Component
 * ================================================================================
 * 
 * This React component displays real-time AI-powered stock predictions for
 * Colombo Stock Exchange (CSE) companies. It fetches predictions from the
 * Python ML backend API and displays them with an interactive chart and
 * confidence metrics.
 * 
 * Features:
 * - Company search and selection
 * - Interactive price chart with timeframe filters
 * - Real-time prediction display with confidence ring
 * - Model analysis breakdown (BiLSTM + XGBoost probabilities)
 * - Technical indicator display
 * 
 * Dependencies:
 * - React (useState, useMemo, useEffect hooks)
 * - Framer Motion (animations)
 * - Recharts (price chart)
 * - Lucide React (icons)
 * 
 * Author: MarketSage Team
 * ================================================================================
 */

// =============================================================================
// IMPORTS
// =============================================================================

// React hooks for state management and side effects
import { useState, useMemo, useEffect } from "react";

// Framer Motion for smooth animations
import { motion } from "framer-motion";

// Lucide React icons for UI elements
import {
  Search,       // Search icon for search bar
  TrendingUp,   // Upward arrow for bullish signals
  TrendingDown, // Downward arrow for bearish signals
  Sparkles,     // Sparkle icon for AI badge
  Loader2,      // Spinning loader for loading states
  Brain,        // Brain icon for model analysis
  BarChart3,    // Chart icon for analysis panel
  Activity      // Activity icon for technical indicators
} from "lucide-react";

// Recharts components for interactive price chart
import {
  AreaChart,           // Area chart component
  Area,                // Area fill for the chart
  XAxis,               // X-axis (dates)
  YAxis,               // Y-axis (prices)
  Tooltip,             // Hover tooltip
  ResponsiveContainer  // Makes chart responsive to container size
} from "recharts";

// =============================================================================
// CONSTANTS
// =============================================================================

// Golden Ratio - used for aesthetically pleasing proportions in the UI
const PHI = 1.618;

// List of Sri Lankan (CSE) companies available for prediction
// These are the 5 companies the ML model was trained on
const companies = [
  { symbol: "COMB", name: "Commercial Bank" },  // Banking sector
  { symbol: "CTC", name: "Ceylon Tobacco" },    // Consumer goods
  { symbol: "DIAL", name: "Dialog Axiata" },    // Telecommunications
  { symbol: "DIST", name: "Distilleries" },     // Beverages
  { symbol: "JKH", name: "John Keells" },       // Conglomerate
];

// Base URL for the Python Prediction API
// The Flask server runs on port 5000
const PREDICTION_API_URL = 'http://localhost:5000/api';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Prediction interface - defines the structure of API response
 * This type ensures TypeScript knows what fields to expect from the ML backend
 */
interface Prediction {
  symbol: string;                              // Stock ticker symbol (e.g., "COMB")
  name: string;                                // Full company name
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";   // Trading signal from ensemble model
  confidence: number;                          // Confidence percentage (50-95%)
  target: number;                              // Predicted target price
  currentPrice: number;                        // Current stock price
  timeframe: string;                           // Prediction timeframe (e.g., "1-5 Days")
  risk: "Low" | "Medium" | "High";             // Risk level based on volatility

  // Model probability outputs (0 to 1 range)
  probabilities: {
    bilstm: number;     // BiLSTM neural network probability
    xgboost: number;    // XGBoost gradient boosting probability
    ensemble: number;   // Weighted ensemble probability (0.4*bilstm + 0.6*xgboost)
  };

  // Technical indicators used by the model
  features: {
    close: number;        // Latest closing price
    rsi: number;          // Relative Strength Index (0-100)
    macd: number;         // Moving Average Convergence Divergence
    macd_signal: number;  // MACD Signal Line
    bb_position: number;  // Bollinger Bands Position (0-1)
    log_return: number;   // Daily logarithmic return
    volatility: number;   // 20-day rolling volatility
    volume_ratio: number; // Volume ratio vs 20-day average
  };

  // Historical price data for chart
  chartData: Array<{ Date: string; price: number }>;

  // When the analysis was performed
  analysisDate: string;
}

// Available timeframe filters for the chart
const timeFilters = ["1W", "1M", "3M"];  // 1 Week, 1 Month, 3 Months

// =============================================================================
// MAIN COMPONENT
// =============================================================================

const PredictionSection = () => {
  // ===========================================================================
  // STATE MANAGEMENT
  // ===========================================================================

  // Search bar state - for filtering companies by name/symbol
  const [searchValue, setSearchValue] = useState("");

  // Currently selected company symbol
  const [activeCompany, setActiveCompany] = useState("COMB");

  // Currently selected timeframe for chart display
  const [activeTimeframe, setActiveTimeframe] = useState("1M");

  // API state management
  const [prediction, setPrediction] = useState<Prediction | null>(null);  // Prediction data from API
  const [loading, setLoading] = useState(false);                          // Loading indicator
  const [error, setError] = useState<string | null>(null);                // Error message if any
  const [showAnalysis, setShowAnalysis] = useState(false);                // Toggle for analysis panel

  // ===========================================================================
  // MEMOIZED VALUES (optimized calculations that only recompute when dependencies change)
  // ===========================================================================

  /**
   * Filter companies based on search input
   * Uses useMemo to avoid recalculating on every render
   */
  const filteredCompanies = useMemo(() => {
    // If no search query, return all companies
    if (!searchValue.trim()) return companies;

    // Filter by symbol or name (case-insensitive)
    const query = searchValue.toLowerCase();
    return companies.filter(
      (c) =>
        c.symbol.toLowerCase().includes(query) ||
        c.name.toLowerCase().includes(query)
    );
  }, [searchValue]);  // Only recalculate when searchValue changes

  // ===========================================================================
  // SIDE EFFECTS
  // ===========================================================================

  /**
   * Fetch prediction whenever the selected company changes
   * useEffect runs after render when dependencies change
   */
  useEffect(() => {
    fetchPrediction(activeCompany);
  }, [activeCompany]);  // Dependency: runs when activeCompany changes

  // ===========================================================================
  // API FUNCTIONS
  // ===========================================================================

  /**
   * Fetch prediction from the Python ML API
   * 
   * @param symbol - Stock ticker symbol (e.g., "COMB")
   * 
   * This function:
   * 1. Sets loading state to show spinner
   * 2. Clears any previous errors
   * 3. Makes async fetch request to /api/predict/<symbol>
   * 4. Handles success/error responses
   * 5. Updates state with prediction data or error message
   */
  const fetchPrediction = async (symbol: string) => {
    setLoading(true);     // Show loading spinner
    setError(null);       // Clear previous errors

    try {
      // Make GET request to prediction API
      const response = await fetch(`${PREDICTION_API_URL}/predict/${symbol}`);

      // Check if response was successful (status 200-299)
      if (!response.ok) {
        throw new Error(`Failed to fetch prediction for ${symbol}`);
      }

      // Parse JSON response
      const data = await response.json();

      // Update prediction state with API data
      setPrediction(data);

    } catch (err) {
      // Log error for debugging
      console.error('Prediction fetch error:', err);

      // Set user-friendly error message
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction');

      // Clear prediction data on error
      setPrediction(null);

    } finally {
      // Always hide loading spinner when done (success or error)
      setLoading(false);
    }
  };

  // ===========================================================================
  // EVENT HANDLERS
  // ===========================================================================

  /**
   * Handle company tab selection
   * @param symbol - Selected company symbol
   */
  const handleCompanyChange = (symbol: string) => {
    setActiveCompany(symbol);  // This triggers useEffect to fetch new prediction
  };

  /**
   * Handle timeframe filter selection
   * @param tf - Selected timeframe ("1W", "1M", or "3M")
   */
  const handleTimeframeChange = (tf: string) => {
    setActiveTimeframe(tf);  // Update chart to show different time period
  };

  /**
   * Handle analyze button click (from search bar)
   * Selects the first matching company from search results
   */
  const handleAnalyze = () => {
    if (filteredCompanies.length > 0) {
      const firstMatch = filteredCompanies[0];
      handleCompanyChange(firstMatch.symbol);
    }
  };

  // ===========================================================================
  // DERIVED DATA
  // ===========================================================================

  /**
   * Prepare chart data based on selected timeframe
   * Filters the historical price data from the API response
   */
  const chartData = useMemo(() => {
    // Return empty array if no prediction data
    if (!prediction?.chartData) return [];

    // Get the full chart data from API
    const data = prediction.chartData;

    // Filter based on selected timeframe
    if (activeTimeframe === "1W") return data.slice(-7);   // Last 7 days
    if (activeTimeframe === "1M") return data.slice(-30);  // Last 30 days
    return data;  // Full data for 3M

  }, [prediction, activeTimeframe]);

  // Convenience booleans for signal-based styling
  const isBullish = prediction?.signal === "BULLISH";  // Green styling
  const isBearish = prediction?.signal === "BEARISH";  // Red styling

  /**
   * Calculate SVG circle stroke-dashoffset for confidence ring
   * 
   * The ring is an SVG circle with circumference 282.7 (2 * PI * 45 radius)
   * strokeDashoffset controls how much of the circle is "hidden"
   * - 282.7 = completely hidden (0% confidence)
   * - 0 = fully visible (100% confidence)
   */
  const ringOffset = 282.7 - (282.7 * (prediction?.confidence || 0) / 100);

  // ===========================================================================
  // RENDER
  // ===========================================================================

  return (
    <section
      id="prediction"
      className="min-h-screen flex items-center py-32 relative"
    >
      {/* ---------------------------------------------------------------------
          Background Spotlight Effect
          A radial gradient that creates a subtle glow behind the content
      --------------------------------------------------------------------- */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-gradient-radial from-primary/10 via-transparent to-transparent pointer-events-none" />

      <div className="container mx-auto px-6">

        {/* -------------------------------------------------------------------
            Section Header
            Animated title with "Market" and "Intelligence" text
        ------------------------------------------------------------------- */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}           // Start invisible and below
          whileInView={{ opacity: 1, y: 0 }}        // Animate to visible and in place
          viewport={{ once: true }}                  // Only animate once when scrolled into view
          transition={{ duration: 0.6 }}             // Animation duration
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
            <span className="text-foreground">Market</span>{" "}
            <span className="text-primary text-glow">Intelligence</span>
          </h2>
          <p className="text-lg text-foreground/50 max-w-2xl mx-auto">
            Real-time AI predictions powered by BiLSTM + XGBoost ensemble model
          </p>
        </motion.div>

        {/* -------------------------------------------------------------------
            Search Bar
            Allows users to search for companies by symbol or name
        ------------------------------------------------------------------- */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="max-w-3xl mx-auto mb-16"
        >
          {/* Glowing border effect container */}
          <div className="relative group">
            {/* Blur glow effect on hover */}
            <div className="absolute -inset-1 bg-gradient-to-r from-primary/30 via-primary/50 to-primary/30 rounded-2xl blur-lg opacity-15 group-hover:opacity-25 transition-opacity duration-500" />

            {/* Search input container */}
            <div className="relative flex items-center">
              {/* Search icon */}
              <Search className="absolute left-6 w-6 h-6 text-primary" />

              {/* Search input field */}
              <input
                type="text"
                placeholder="Search stock symbol (e.g., COMB, CTC, JKH)..."
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}  // Update search value on type
                onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}  // Submit on Enter
                className="w-full input-premium pl-16 text-lg"
              />

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={loading}  // Disable while loading
                className="absolute right-3 px-6 py-2 rounded-lg bg-primary text-primary-foreground font-semibold hover:shadow-emerald transition-all duration-300 disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Analyze"}
              </button>
            </div>
          </div>

          {/* Search results count */}
          {searchValue && (
            <p className="text-sm text-foreground/40 mt-2 text-center">
              Found {filteredCompanies.length} matching {filteredCompanies.length === 1 ? "company" : "companies"}
            </p>
          )}
        </motion.div>

        {/* -------------------------------------------------------------------
            Interactive Price Chart Section
            Displays historical price data with company tabs and time filters
        ------------------------------------------------------------------- */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="max-w-5xl mx-auto mb-16"
        >
          <div className="card-premium overflow-hidden" style={{ padding: `${PHI * 1.5}rem` }}>

            {/* Header: Company Tabs + Time Filters */}
            <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8" style={{ gap: `${PHI}rem` }}>

              {/* Company Selection Tabs */}
              <div className="flex flex-wrap items-center" style={{ gap: `${PHI * 0.5}rem` }}>
                {filteredCompanies.length > 0 ? (
                  filteredCompanies.map((company) => (
                    <button
                      key={company.symbol}
                      onClick={() => handleCompanyChange(company.symbol)}
                      disabled={loading}
                      className={`relative px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-300 ${activeCompany === company.symbol
                          ? "text-primary bg-primary/10"           // Active tab styling
                          : "text-foreground/60 hover:text-foreground hover:bg-white/5"  // Inactive
                        } disabled:opacity-50`}
                      style={{ height: `${PHI * 2}rem` }}
                    >
                      {company.symbol}

                      {/* Animated underline for active tab */}
                      {activeCompany === company.symbol && (
                        <motion.div
                          layoutId="companyUnderline"  // Shared layout animation ID
                          className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"
                          style={{ boxShadow: "0 0 10px hsl(160 84% 39%)" }}
                        />
                      )}
                    </button>
                  ))
                ) : (
                  <p className="text-foreground/50 text-sm">No companies found. Try a different search.</p>
                )}
              </div>

              {/* Timeframe Filter Buttons */}
              <div className="flex items-center" style={{ gap: `${PHI * 0.3}rem` }}>
                {timeFilters.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => handleTimeframeChange(tf)}
                    className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all duration-300 ${activeTimeframe === tf
                        ? "bg-primary text-primary-foreground"  // Active filter
                        : "glass text-foreground/70 hover:text-foreground"  // Inactive
                      }`}
                    style={{ height: `${PHI * 1.5}rem` }}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            {/* Company Name Display */}
            <div className="mb-4">
              <h3 className="text-2xl font-display font-bold text-foreground">
                {companies.find((c) => c.symbol === activeCompany)?.name}
              </h3>
              <p className="text-foreground/50 text-sm">
                CSE: {activeCompany}.N0000
                {prediction && (
                  <span className="ml-4 text-foreground/40">
                    Current: Rs. {prediction.currentPrice.toFixed(2)}
                  </span>
                )}
              </p>
            </div>

            {/* Chart Container */}
            <div
              className="relative"
              style={{ height: `calc(50vw / ${PHI})`, maxHeight: "400px", minHeight: "250px" }}
            >
              {/* Loading State */}
              {loading ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto mb-4" />
                    <p className="text-foreground/50">Analyzing {activeCompany}...</p>
                    <p className="text-sm text-foreground/30 mt-2">Running BiLSTM + XGBoost ensemble</p>
                  </div>
                </div>
              ) : error ? (
                /* Error State */
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-red-400">
                    <p className="text-lg font-semibold">Analysis Error</p>
                    <p className="text-sm opacity-70">{error}</p>
                    <button
                      onClick={() => fetchPrediction(activeCompany)}
                      className="mt-4 px-4 py-2 bg-primary/20 text-primary rounded-lg hover:bg-primary/30 transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                </div>
              ) : (
                /* Chart Display */
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    {/* Gradient definition for area fill */}
                    <defs>
                      <linearGradient id="emeraldGradient" x1="0" y1="0" x2="0" y2="1">
                        {/* Top color: Bullish=Green, Bearish=Red */}
                        <stop offset="0%" stopColor={isBearish ? "hsl(0 84% 60%)" : "hsl(160 84% 39%)"} stopOpacity={0.4} />
                        {/* Bottom color: Fades to transparent */}
                        <stop offset="100%" stopColor={isBearish ? "hsl(0 84% 60%)" : "hsl(160 84% 39%)"} stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    {/* X-Axis: Dates */}
                    <XAxis
                      dataKey="Date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "hsl(210 40% 98% / 0.4)", fontSize: 11 }}
                      tickFormatter={(value) => value.slice(5)}  // Show only MM-DD
                    />

                    {/* Y-Axis: Prices */}
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "hsl(210 40% 98% / 0.4)", fontSize: 11 }}
                      domain={["dataMin - 5", "dataMax + 5"]}  // Add padding to range
                      tickFormatter={(value) => `${value.toFixed(0)}`}
                    />

                    {/* Tooltip on hover */}
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(222.2 47.4% 11.2% / 0.9)",
                        border: `1px solid ${isBearish ? "hsl(0 84% 60% / 0.3)" : "hsl(160 84% 39% / 0.3)"}`,
                        borderRadius: "12px",
                        boxShadow: `0 0 20px ${isBearish ? "hsl(0 84% 60% / 0.2)" : "hsl(160 84% 39% / 0.2)"}`,
                      }}
                      labelStyle={{ color: "hsl(210 40% 98%)", fontWeight: "bold" }}
                      itemStyle={{ color: isBearish ? "hsl(0 84% 60%)" : "hsl(160 84% 39%)" }}
                      formatter={(value: number) => [`Rs. ${value.toFixed(2)}`, "Price"]}
                    />

                    {/* Area chart with gradient fill */}
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke={isBearish ? "hsl(0 84% 60%)" : "hsl(160 84% 39%)"}
                      strokeWidth={2}
                      fill="url(#emeraldGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </motion.div>

        {/* -------------------------------------------------------------------
            Prediction Result Card
            Shows signal, confidence, target price, and model analysis
        ------------------------------------------------------------------- */}
        {prediction && !loading && !error && (
          <motion.div
            key={activeCompany}  // Re-animate when company changes
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="max-w-4xl mx-auto"
          >
            <div className="card-premium relative overflow-hidden">

              {/* Glassmorphism overlay effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />

              <div className="relative z-10 flex flex-col lg:flex-row items-center gap-12 p-8">

                {/* ---------------------------------------------------------
                    Confidence Ring
                    SVG circular progress indicator showing prediction confidence
                --------------------------------------------------------- */}
                <div className="relative flex-shrink-0">
                  <svg className="w-48 h-48" viewBox="0 0 100 100">
                    {/* Background ring (gray) */}
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className="text-white/10"
                    />

                    {/* Progress ring (colored based on signal) */}
                    <motion.circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke={isBearish ? "hsl(0 84% 60%)" : isBullish ? "hsl(160 84% 39%)" : "hsl(45 84% 50%)"}
                      strokeWidth="4"
                      strokeLinecap="round"
                      strokeDasharray="282.7"  // Full circle circumference
                      initial={{ strokeDashoffset: 282.7 }}         // Start empty
                      animate={{ strokeDashoffset: ringOffset }}    // Animate to confidence level
                      transition={{ duration: 1, ease: "easeOut" }}
                      style={{
                        filter: `drop-shadow(0 0 10px ${isBearish ? "hsla(0, 84%, 60%, 0.5)" : "hsla(160, 84%, 39%, 0.5)"})`,
                      }}
                    />
                  </svg>

                  {/* Arrow icon inside the ring */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                      initial={{ scale: 0.8 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200 }}
                    >
                      {isBearish ? (
                        <TrendingDown className="w-16 h-16 text-red-500" strokeWidth={2.5} />
                      ) : (
                        <TrendingUp className={`w-16 h-16 ${isBullish ? "text-primary" : "text-yellow-500"}`} strokeWidth={2.5} />
                      )}
                    </motion.div>
                  </div>
                </div>

                {/* ---------------------------------------------------------
                    Prediction Content
                    Signal, confidence, target, timeframe, risk level
                --------------------------------------------------------- */}
                <div className="flex-1 text-center lg:text-left space-y-6">

                  {/* Badges row */}
                  <div className="flex items-center justify-center lg:justify-start gap-3">
                    {/* AI Analysis badge */}
                    <span className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${isBullish ? "bg-primary/20 text-primary" :
                        isBearish ? "bg-red-500/20 text-red-400" :
                          "bg-yellow-500/20 text-yellow-400"
                      }`}>
                      <Sparkles className="w-3 h-3" />
                      AI Analysis
                    </span>

                    {/* Stock symbol badge */}
                    <span className="px-3 py-1 rounded-full bg-white/10 text-xs font-medium text-foreground/70">
                      {activeCompany}
                    </span>

                    {/* Toggle analysis panel button */}
                    <button
                      onClick={() => setShowAnalysis(!showAnalysis)}
                      className="px-3 py-1 rounded-full bg-primary/10 text-xs font-medium text-primary hover:bg-primary/20 transition-colors flex items-center gap-1"
                    >
                      <Brain className="w-3 h-3" />
                      {showAnalysis ? "Hide" : "Show"} Details
                    </button>
                  </div>

                  {/* Signal and Confidence Display */}
                  <div>
                    <h3 className={`font-display text-3xl md:text-4xl font-bold mb-2 ${isBullish ? "text-primary" :
                        isBearish ? "text-red-400" :
                          "text-yellow-400"
                      }`}>
                      {prediction.signal} SIGNAL
                    </h3>
                    <p className="text-5xl md:text-6xl font-display font-bold text-foreground">
                      {prediction.confidence}%{" "}
                      <span className="text-2xl text-foreground/50 font-normal">
                        Confidence
                      </span>
                    </p>
                  </div>

                  {/* Metrics row: Target, Timeframe, Risk */}
                  <div className="flex flex-wrap gap-6 text-sm text-foreground/60">
                    <div>
                      <span className="block text-foreground/40">Target</span>
                      <span className="text-lg font-semibold text-foreground">
                        Rs. {prediction.target.toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="block text-foreground/40">Timeframe</span>
                      <span className="text-lg font-semibold text-foreground">
                        {prediction.timeframe}
                      </span>
                    </div>
                    <div>
                      <span className="block text-foreground/40">Risk Level</span>
                      <span className={`text-lg font-semibold ${prediction.risk === "Low" ? "text-primary" :
                          prediction.risk === "High" ? "text-red-400" :
                            "text-yellow-400"
                        }`}>
                        {prediction.risk}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ---------------------------------------------------------
                  Model Analysis Panel (Expandable)
                  Shows detailed breakdown of model predictions and features
              --------------------------------------------------------- */}
              {showAnalysis && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="border-t border-white/10 p-6"
                >
                  <h4 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-primary" />
                    Model Analysis
                  </h4>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                    {/* Ensemble Breakdown: BiLSTM and XGBoost probabilities */}
                    <div className="space-y-3">
                      <p className="text-sm text-foreground/50 font-medium">Ensemble Breakdown</p>
                      <div className="space-y-2">
                        {/* BiLSTM contribution (40% weight) */}
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-foreground/70">BiLSTM (40%)</span>
                          <span className="text-sm font-mono text-primary">{(prediction.probabilities.bilstm * 100).toFixed(1)}%</span>
                        </div>
                        {/* XGBoost contribution (60% weight) */}
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-foreground/70">XGBoost (60%)</span>
                          <span className="text-sm font-mono text-primary">{(prediction.probabilities.xgboost * 100).toFixed(1)}%</span>
                        </div>
                        {/* Combined ensemble result */}
                        <div className="flex justify-between items-center border-t border-white/10 pt-2">
                          <span className="text-sm text-foreground font-medium">Ensemble</span>
                          <span className="text-sm font-mono text-primary font-bold">{(prediction.probabilities.ensemble * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Technical Indicators: Features used by the model */}
                    <div className="space-y-3">
                      <p className="text-sm text-foreground/50 font-medium flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Technical Indicators
                      </p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {/* RSI: Relative Strength Index (momentum indicator) */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">RSI</span>
                          <span className="font-mono text-foreground">{prediction.features.rsi}</span>
                        </div>
                        {/* MACD: Moving Average Convergence Divergence (trend indicator) */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">MACD</span>
                          <span className="font-mono text-foreground">{prediction.features.macd}</span>
                        </div>
                        {/* BB Position: Position within Bollinger Bands (0-1) */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">BB Position</span>
                          <span className="font-mono text-foreground">{prediction.features.bb_position}</span>
                        </div>
                        {/* Volatility: 20-day rolling standard deviation */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">Volatility</span>
                          <span className="font-mono text-foreground">{prediction.features.volatility}</span>
                        </div>
                        {/* Log Return: Daily logarithmic return */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">Log Return</span>
                          <span className="font-mono text-foreground">{prediction.features.log_return}</span>
                        </div>
                        {/* Volume Ratio: Volume vs 20-day average */}
                        <div className="flex justify-between">
                          <span className="text-foreground/50">Vol Ratio</span>
                          <span className="font-mono text-foreground">{prediction.features.volume_ratio}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Footer with analysis timestamp and model accuracy disclaimer */}
                  <p className="text-xs text-foreground/30 mt-4 text-center">
                    Analysis performed: {prediction.analysisDate} | Model accuracy: ~62%
                  </p>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
};

// Export the component for use in other parts of the application
export default PredictionSection;
