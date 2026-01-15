"""
================================================================================
MarketSage Prediction API
================================================================================
This Flask application serves real-time stock predictions for the Colombo Stock
Exchange (CSE) using a trained ensemble model combining BiLSTM (Deep Learning) 
and XGBoost (Gradient Boosting).

The API loads pre-trained models at startup and provides REST endpoints for:
- Health checks
- Listing available companies
- Getting predictions for specific stocks

Author: MarketSage Team
Version: 1.0.0
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os                      # Operating system interface (file paths)
import pickle                  # Serialize/deserialize Python objects
from datetime import datetime  # Date and time handling

# Data processing libraries
import numpy as np             # Numerical computing (arrays, math operations)
import pandas as pd            # Data manipulation and analysis (DataFrames)

# Web framework
from flask import Flask, jsonify, request  # Flask web framework for REST API
from flask_cors import CORS    # Cross-Origin Resource Sharing (allows frontend access)

# Machine Learning - TensorFlow/Keras for BiLSTM neural network
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning logs
import tensorflow as tf        # TensorFlow deep learning framework
from tensorflow import keras   # High-level neural network API

# Machine Learning - XGBoost for gradient boosting
import xgboost as xgb          # XGBoost gradient boosting library

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is two levels up from backend/prediction_api/
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Directory containing trained model files (BiLSTM, XGBoost, Scaler)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ensemble")

# Directory containing historical stock data CSV files
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "historical")

# Model hyperparameters (must match training configuration)
SEQ_LENGTH = 20      # Number of days of historical data to use for prediction (20-day window)
BEST_THRESHOLD = 0.3 # Threshold for converting probability to binary prediction

# Dictionary mapping stock symbols to full company names
# These are the 5 CSE companies the model was trained on
COMPANIES = {
    "COMB": "Commercial Bank of Ceylon PLC - COMB",   # Banking sector
    "CTC": "Ceylon Tobacco Company PLC - CTC",        # Consumer goods
    "DIAL": "Dialog Axiata PLC - DIAL",               # Telecommunications
    "DIST": "Distilleries Company of Sri Lanka PLC - DIST",  # Beverages
    "JKH": "John Keells Holdings PLC - JKH",          # Conglomerate
}

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

# Create Flask application instance
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows the frontend (running on different port) to access this API
CORS(app)

# =============================================================================
# GLOBAL MODEL VARIABLES
# =============================================================================
# These are loaded once at startup and reused for all predictions

bilstm_model = None  # BiLSTM (Bidirectional LSTM) deep learning model
xgb_model = None     # XGBoost gradient boosting model
scaler = None        # RobustScaler for normalizing input features


def load_models():
    """
    Load all machine learning models and the scaler at application startup.
    
    This function:
    1. Loads the BiLSTM model (.keras file) for sequence-based predictions
    2. Loads the XGBoost model (.json file) for feature-based predictions
    3. Loads the scaler (.pkl file) or creates a new one if loading fails
    
    Models are loaded into global variables for efficient reuse.
    """
    global bilstm_model, xgb_model, scaler
    
    print("📦 Loading Models...")
    
    # ----------------------------
    # Load BiLSTM Neural Network
    # ----------------------------
    # BiLSTM (Bidirectional Long Short-Term Memory) is a recurrent neural network
    # that can learn patterns in sequential data by processing it in both directions
    bilstm_path = os.path.join(MODEL_DIR, "bi_lstm_model.keras")
    bilstm_model = keras.models.load_model(bilstm_path)
    print(f"✅ BiLSTM Model loaded")
    
    # ----------------------------
    # Load XGBoost Model
    # ----------------------------
    # XGBoost (eXtreme Gradient Boosting) is an ensemble of decision trees
    # that excel at tabular/structured data predictions
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    xgb_model = xgb.Booster()  # Create empty Booster object
    xgb_model.load_model(xgb_path)  # Load trained parameters
    print(f"✅ XGBoost Model loaded")
    
    # ----------------------------
    # Load Feature Scaler
    # ----------------------------
    # The scaler normalizes input features to have similar ranges
    # RobustScaler is used because it handles outliers better than StandardScaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)  # Load pre-fitted scaler from pickle file
        print(f"✅ Scaler loaded")
    except Exception as e:
        # If scaler pickle is incompatible (different Python/sklearn version),
        # create a fresh scaler that will be fitted on first prediction
        print(f"⚠️ Could not load scaler: {e}")
        print("  Creating fresh RobustScaler...")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler._fitted = False  # Custom flag to track if fitting is needed
        print(f"✅ Fresh scaler created (will be fitted on first prediction)")
    
    print("🚀 All models loaded successfully!")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_company_folder(symbol):
    """
    Find the folder containing historical data for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., "COMB", "JKH")
    
    Returns:
        str: Full path to the company's data folder, or None if not found
    
    The data is organized in two period folders:
    - Historical data 2020-2025
    - Historical data 2015-2020
    """
    # Search in both historical data periods
    for period in ["Historical data 2020-2025", "Historical data 2015-2020"]:
        period_path = os.path.join(DATA_DIR, period)
        if not os.path.exists(period_path):
            continue  # Skip if period folder doesn't exist
        
        # Look for folder matching the stock symbol
        for company_dir in os.listdir(period_path):
            if company_dir.startswith(symbol + " -") or company_dir.startswith(symbol + " "):
                return os.path.join(period_path, company_dir)
    
    return None  # Company not found


def load_company_data(symbol):
    """
    Load and combine daily price data for a company from all available CSV files.
    
    Args:
        symbol: Stock ticker symbol (e.g., "COMB", "JKH")
    
    Returns:
        pandas.DataFrame: Combined historical data with columns: Date, Close, etc.
                         Returns None if no data found
    
    This function:
    1. Searches both historical data periods for matching folders
    2. Reads Daily.csv files from matching folders
    3. Combines data from multiple periods
    4. Cleans and standardizes column names
    5. Converts date strings to datetime objects
    6. Sorts data chronologically
    """
    all_data = []  # List to collect DataFrames from multiple sources
    
    # Search in both historical data periods
    for period in ["Historical data 2020-2025", "Historical data 2015-2020"]:
        period_path = os.path.join(DATA_DIR, period)
        if not os.path.exists(period_path):
            continue
        
        # Look for company folders matching the symbol
        for company_dir in os.listdir(period_path):
            # Folder naming: "COMB.N0000 - Commercial Bank" or similar
            if not (company_dir.startswith(symbol + ".N0000") or 
                    company_dir.startswith(symbol + " -") or 
                    company_dir.startswith(symbol + " ")):
                continue
            
            # Load the Daily.csv file
            csv_path = os.path.join(period_path, company_dir, "Daily.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
    
    # Return None if no data found
    if not all_data:
        return None
    
    # Combine all DataFrames into one
    df = pd.concat(all_data, ignore_index=True)
    
    # Clean column names (remove extra whitespace)
    df.columns = df.columns.str.strip()
    
    # Standardize date column name
    if 'Trade Date' in df.columns:
        df.rename(columns={'Trade Date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Standardize close price column name
    if 'Close (Rs.)' in df.columns:
        df.rename(columns={'Close (Rs.)': 'Close'}, inplace=True)
    
    # Convert Close to numeric (handles any formatting issues)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Remove rows with missing Close prices
    df = df.dropna(subset=['Close'])
    
    # Sort by date (oldest first) and reset index
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# IMPORTANT: These features MUST exactly match what was used during model training!

def create_features(df):
    """
    Create technical indicator features from raw price data.
    
    This function calculates the same 8 features that were used during model training.
    These are standard technical indicators used in stock market analysis.
    
    Args:
        df: DataFrame with at least 'Date' and 'Close' columns
    
    Returns:
        DataFrame with added feature columns and NaN rows dropped
    
    Features created:
    1. RSI (Relative Strength Index) - Momentum indicator (0-100)
    2. MACD (Moving Average Convergence Divergence) - Trend-following indicator
    3. MACD_Signal - 9-day EMA of MACD
    4. BB_Position - Position within Bollinger Bands (0-1)
    5. Log_Return - Daily logarithmic return
    6. Volatility - 20-day rolling standard deviation of returns
    7. Volume_Ratio - Current volume vs 20-day average
    8. Target - Binary: 1 if next day's close > today's close
    """
    df = df.copy()  # Don't modify original DataFrame
    df = df.sort_values('Date')  # Ensure chronological order
    
    # --------------------------------
    # 1. RSI (Relative Strength Index)
    # --------------------------------
    # RSI measures the magnitude of recent price changes
    # RSI > 70: Overbought (potential sell signal)
    # RSI < 30: Oversold (potential buy signal)
    delta = df['Close'].diff()  # Daily price changes
    gain = delta.where(delta > 0, 0).rolling(14).mean()   # Average gains over 14 days
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean() # Average losses over 14 days
    rs = gain / (loss + 1e-10)  # Relative Strength (add small number to avoid division by zero)
    df['RSI'] = 100 - (100 / (1 + rs))  # RSI formula
    
    # --------------------------------
    # 2. MACD (Moving Average Convergence Divergence)
    # --------------------------------
    # MACD shows the relationship between two moving averages
    # Positive MACD: Short-term trend is up
    # Negative MACD: Short-term trend is down
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()  # 12-day exponential moving average
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()  # 26-day exponential moving average
    df['MACD'] = ema12 - ema26  # MACD line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line (9-day EMA of MACD)
    
    # --------------------------------
    # 3. Bollinger Bands Position
    # --------------------------------
    # Bollinger Bands show price volatility
    # BB_Position of 0: Price at lower band
    # BB_Position of 1: Price at upper band
    # BB_Position of 0.5: Price at middle (20-day SMA)
    window = 20
    rolling_mean = df['Close'].rolling(window).mean()  # 20-day simple moving average
    rolling_std = df['Close'].rolling(window).std()    # 20-day standard deviation
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)  # Upper band (mean + 2*std)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)  # Lower band (mean - 2*std)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # --------------------------------
    # 4. Price Transformations
    # --------------------------------
    # Log returns are preferred over simple returns because they are:
    # - Additive (can sum daily returns for total return)
    # - More normally distributed
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility: Rolling standard deviation of log returns
    # Higher volatility = higher risk
    df['Volatility'] = df['Log_Return'].rolling(20).std()
    
    # --------------------------------
    # 5. Volume Analysis
    # --------------------------------
    # Volume ratio compares current volume to 20-day average
    # > 1: Higher than average volume (potential significant move)
    # < 1: Lower than average volume
    if 'ShareVolume' in df.columns:
        df['Volume_SMA'] = df['ShareVolume'].rolling(20).mean()
        df['Volume_Ratio'] = df['ShareVolume'] / (df['Volume_SMA'] + 1e-10)
    else:
        df['Volume_Ratio'] = 0  # Default if volume data not available
    
    # --------------------------------
    # 6. Target Variable (for training format compatibility)
    # --------------------------------
    # Binary classification target:
    # 1 = Price went UP the next day
    # 0 = Price went DOWN the next day
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove rows with NaN values (created by rolling calculations)
    df = df.dropna()
    
    return df


# List of feature column names (must match training exactly)
# 8 features + 1 target = 9 columns used by the model
FEATURE_COLS = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'Log_Return', 'Volatility', 'Volume_Ratio']
TARGET_COL = 'Target'


def get_smart_features(X_seq):
    """
    Extract summary statistics from sequences for XGBoost model.
    
    XGBoost expects a flat 2D array, not sequences, so we extract:
    1. Last values of each feature in the sequence
    2. Lagged close prices (1, 3, 5 days ago)
    3. Mean of each feature across the sequence
    4. Standard deviation of each feature across the sequence
    
    Args:
        X_seq: 3D numpy array of shape (n_samples, seq_length, n_features+1)
               The +1 is for the target column
    
    Returns:
        2D numpy array with extracted features for XGBoost
    
    This function MUST match exactly what was used during training!
    """
    # X_seq shape: (Samples, SEQ_LENGTH, 9) - 8 features + 1 target column
    
    # Last values of all features (excluding target column)
    last_val = X_seq[:, -1, :-1]  # Shape: (n_samples, 8)
    
    # Lagged close prices (Close is at index 0)
    lag_1 = X_seq[:, -2, 0:1]  # Close price 1 day ago
    lag_3 = X_seq[:, -4, 0:1]  # Close price 3 days ago
    lag_5 = X_seq[:, -6, 0:1]  # Close price 5 days ago
    
    # Mean of each feature across the 20-day sequence
    mean_val = np.mean(X_seq[:, :, :-1], axis=1)  # Shape: (n_samples, 8)
    
    # Standard deviation of each feature across the sequence
    std_val = np.std(X_seq[:, :, :-1], axis=1)  # Shape: (n_samples, 8)
    
    # Combine all features horizontally
    # Total: 8 + 8 + 8 + 1 + 1 + 1 = 27 features
    return np.hstack([last_val, mean_val, std_val, lag_1, lag_3, lag_5])


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def make_prediction(symbol):
    """
    Generate a stock prediction for the given company symbol.
    
    This is the main prediction function that:
    1. Loads historical price data for the company
    2. Engineers features (technical indicators)
    3. Scales the features
    4. Runs BiLSTM model for probability estimate
    5. Runs XGBoost model for probability estimate
    6. Combines predictions using weighted ensemble (40% BiLSTM, 60% XGBoost)
    7. Determines signal (BULLISH/BEARISH/NEUTRAL) based on ensemble probability
    8. Calculates confidence and risk levels
    9. Returns comprehensive prediction result
    
    Args:
        symbol: Stock ticker symbol (e.g., "COMB", "JKH")
    
    Returns:
        tuple: (prediction_dict, error_message)
               prediction_dict is None if error occurred
               error_message is None if successful
    """
    
    # ================================
    # Step 1: Load Historical Data
    # ================================
    df = load_company_data(symbol)
    
    # Validate we have enough data (need SEQ_LENGTH + extra for feature calculation)
    if df is None or len(df) < SEQ_LENGTH + 30:
        return None, "Insufficient data for prediction"
    
    # ================================
    # Step 2: Feature Engineering
    # ================================
    df_features = create_features(df)
    
    if len(df_features) < SEQ_LENGTH:
        return None, "Insufficient processed data"
    
    # ================================
    # Step 3: Prepare Input Sequence
    # ================================
    # Get the most recent 20 days of data
    # Include both features (8) AND target (1) = 9 columns total
    # (This matches the training data format)
    all_cols = FEATURE_COLS + [TARGET_COL]
    features = df_features[all_cols].values[-SEQ_LENGTH:]  # Shape: (20, 9)
    
    # ================================
    # Step 4: Scale Features
    # ================================
    global scaler
    feature_vals = df_features[FEATURE_COLS].values
    
    # If using a fresh scaler (pickle loading failed), fit it on available data
    if hasattr(scaler, '_fitted') and scaler._fitted == False:
        scaler.fit(feature_vals)  # Fit on all available feature data
        scaler._fitted = True
        print(f"  Scaler fitted on {len(feature_vals)} samples")
    
    # Scale the 8 feature columns (don't scale the target column)
    features_for_scaling = features[:, :-1]  # Remove target column
    targets = features[:, -1:]               # Keep target column
    features_scaled = scaler.transform(features_for_scaling)
    
    # Recombine scaled features with unscaled target
    data_for_seq = np.hstack((features_scaled, targets))
    
    # Reshape for LSTM: (batch_size=1, sequence_length=20, features=9)
    X_seq = data_for_seq.reshape(1, SEQ_LENGTH, len(all_cols))
    
    # ================================
    # Step 5: BiLSTM Prediction
    # ================================
    # The BiLSTM model outputs a probability between 0 and 1
    # Higher values indicate higher likelihood of price increase
    bilstm_prob = bilstm_model.predict(X_seq, verbose=0).flatten()[0]
    
    # ================================
    # Step 6: XGBoost Prediction
    # ================================
    # XGBoost uses "smart features" extracted from the sequence
    X_smart = get_smart_features(X_seq)
    dmatrix = xgb.DMatrix(X_smart)  # XGBoost's optimized data structure
    xgb_prob = xgb_model.predict(dmatrix)[0]
    
    # ================================
    # Step 7: Ensemble Combination
    # ================================
    # Weighted average: 40% BiLSTM + 60% XGBoost
    # XGBoost gets more weight because it typically performs better on this task
    ensemble_prob = 0.4 * float(bilstm_prob) + 0.6 * float(xgb_prob)
    
    # ================================
    # Step 8: Determine Trading Signal
    # ================================
    # Convert probability to actionable signal
    if ensemble_prob > 0.55:
        signal = "BULLISH"   # Strong indication of price increase
    elif ensemble_prob < 0.45:
        signal = "BEARISH"   # Strong indication of price decrease
    else:
        signal = "NEUTRAL"   # Uncertain, no clear direction
    
    # ================================
    # Step 9: Calculate Confidence
    # ================================
    # Confidence is based on how far the probability is from 0.5 (neutral)
    # Range: 50% (uncertain) to 95% (very confident)
    raw_confidence = abs(ensemble_prob - 0.5) * 2  # 0 to 1 scale
    confidence = int(50 + raw_confidence * 45)     # 50% to 95% scale
    
    # ================================
    # Step 10: Assess Risk Level
    # ================================
    # Risk is based on recent volatility
    recent_volatility = df_features['Volatility'].iloc[-1]
    
    if recent_volatility < 0.02:
        risk = "Low"      # Stable price movement
    elif recent_volatility < 0.05:
        risk = "Medium"   # Moderate price swings
    else:
        risk = "High"     # Large price swings
    
    # ================================
    # Step 11: Calculate Price Target
    # ================================
    latest_price = float(df_features['Close'].iloc[-1])
    
    # Estimate target price based on signal and confidence
    if signal == "BULLISH":
        target = latest_price * (1 + 0.02 * (confidence / 100))  # Up to 2% gain
    elif signal == "BEARISH":
        target = latest_price * (1 - 0.02 * (confidence / 100))  # Up to 2% loss
    else:
        target = latest_price  # No change expected
    
    # ================================
    # Step 12: Prepare Response Data
    # ================================
    
    # Get latest feature values for analysis display
    latest_features = df_features[FEATURE_COLS].iloc[-1].to_dict()
    
    # Prepare chart data (last 30 days of prices)
    chart_data = df_features[['Date', 'Close']].tail(30).to_dict(orient='records')
    for item in chart_data:
        # Format date as string and add 'price' key for frontend compatibility
        item['Date'] = item['Date'].strftime('%Y-%m-%d') if hasattr(item['Date'], 'strftime') else str(item['Date'])
        item['price'] = float(item['Close'])
    
    # ================================
    # Return Complete Prediction
    # ================================
    return {
        "symbol": symbol,                          # Stock symbol
        "name": COMPANIES.get(symbol, symbol),     # Full company name
        "signal": signal,                          # BULLISH/BEARISH/NEUTRAL
        "confidence": confidence,                  # 50-95%
        "target": round(target, 2),               # Predicted price target
        "currentPrice": round(latest_price, 2),   # Current stock price
        "timeframe": "1-5 Days",                  # Prediction timeframe
        "risk": risk,                             # Low/Medium/High
        "probabilities": {                        # Model probability outputs
            "bilstm": round(float(bilstm_prob), 4),
            "xgboost": round(float(xgb_prob), 4),
            "ensemble": round(ensemble_prob, 4)
        },
        "features": {                             # Technical indicators used
            "close": round(float(latest_features['Close']), 2),
            "rsi": round(float(latest_features['RSI']), 2),
            "macd": round(float(latest_features['MACD']), 4),
            "macd_signal": round(float(latest_features['MACD_Signal']), 4),
            "bb_position": round(float(latest_features['BB_Position']), 4),
            "log_return": round(float(latest_features['Log_Return']), 4),
            "volatility": round(float(latest_features['Volatility']), 4),
            "volume_ratio": round(float(latest_features['Volume_Ratio']), 4),
        },
        "chartData": chart_data,                  # Historical prices for chart
        "analysisDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Prediction timestamp
    }, None  # No error


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON: {status: "healthy", models_loaded: true/false, timestamp: ISO string}
    
    Use this endpoint to verify the API is running and models are loaded.
    """
    return jsonify({
        "status": "healthy",
        "models_loaded": bilstm_model is not None and xgb_model is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/companies', methods=['GET'])
def get_companies():
    """
    Get the list of available companies for prediction.
    
    Returns:
        JSON array: [{symbol: "COMB", name: "Commercial Bank..."}, ...]
    
    These are the 5 CSE companies the model was trained on.
    """
    return jsonify([
        {"symbol": symbol, "name": name}
        for symbol, name in COMPANIES.items()
    ])


@app.route('/api/predict/<symbol>', methods=['GET'])
def predict(symbol):
    """
    Get AI prediction for a specific stock.
    
    Args:
        symbol: Stock ticker symbol in URL path (e.g., /api/predict/COMB)
    
    Returns:
        JSON: Full prediction object including signal, confidence, target price,
              probabilities from each model, technical indicators, and chart data
    
    Status Codes:
        200: Success - prediction returned
        404: Unknown stock symbol
        500: Prediction error (insufficient data, etc.)
    """
    # Normalize symbol to uppercase
    symbol = symbol.upper()
    
    # Validate symbol is in our list
    if symbol not in COMPANIES:
        return jsonify({
            "error": f"Unknown symbol: {symbol}",
            "available": list(COMPANIES.keys())
        }), 404
    
    # Generate prediction
    result, error = make_prediction(symbol)
    
    # Handle errors
    if error:
        return jsonify({"error": error}), 500
    
    # Return successful prediction
    return jsonify(result)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    """
    Main entry point when running the script directly.
    
    1. Loads all ML models into memory
    2. Prints startup information
    3. Starts the Flask development server on port 5000
    """
    
    # Load models at startup (done once, reused for all requests)
    load_models()
    
    # Print startup banner
    print("\n" + "=" * 50)
    print("🚀 MarketSage Prediction API")
    print("=" * 50)
    print("📍 Running on: http://localhost:5000")
    print("📋 Endpoints:")
    print("   GET /api/health     - Health check")
    print("   GET /api/companies  - List companies")
    print("   GET /api/predict/<SYMBOL> - Get prediction")
    print("=" * 50 + "\n")
    
    # Start Flask server
    # host='0.0.0.0' allows connections from other machines on the network
    # debug=False for production (set True for development with auto-reload)
    app.run(host='0.0.0.0', port=5000, debug=False)
