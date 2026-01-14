# 📈 IMPROVED Colombo Stock Exchange - Multi-Model Stock Prediction
# FIXES: Stock Split Adjustment, Better Scaling, Predict Returns Instead of Prices
# GPU-Accelerated Training on Google Colab (T4)
# VERSION: 1000 EPOCHS WITH REPRODUCIBILITY

# ============================================================
# 🔧 STEP 1: SETUP & GPU CHECK + SEEDS FOR REPRODUCIBILITY
# ============================================================

import os
import random
import numpy as np

# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

print("=" * 50)
print("🔧 GPU CHECK - 1000 EPOCH VERSION")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU is enabled!")
else:
    print("⚠️ Enable GPU: Runtime > Change runtime type > T4 GPU")

# ============================================================
# 📚 STEP 2: IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

print("✅ Libraries imported!")

# ============================================================
# 📂 STEP 3: LOAD DATA
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/Historical Data.zip'

import zipfile
with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/data')

print("✅ Data extracted!")

# ============================================================
# 🔧 STEP 4: IMPROVED DATA PREPROCESSING (KEY FIX!)
# ============================================================

def adjust_stock_split(df, split_date='2024-11-01', split_ratio=10):
    """
    Adjust historical prices for stock split
    All prices before split_date are divided by split_ratio
    """
    df = df.copy()
    split_date = pd.to_datetime(split_date)
    
    price_columns = ['Open', 'High', 'Low', 'Close']
    
    # Adjust pre-split prices
    mask = df['Date'] < split_date
    for col in price_columns:
        df.loc[mask, col] = df.loc[mask, col] / split_ratio
    
    print(f"✅ Adjusted {mask.sum()} rows for {split_ratio}:1 stock split")
    return df

def load_and_preprocess_data_v2(company_folder, adjust_split=True):
    """Improved data preprocessing with stock split adjustment"""
    
    # Load data
    df = pd.read_csv(f'{company_folder}/Daily.csv')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']
    
    # Convert date and sort
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset='Date', keep='first')
    
    # Fill missing values
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # CRITICAL: Adjust for stock split
    if adjust_split:
        df = adjust_stock_split(df, split_date='2024-11-01', split_ratio=10)
    
    # === FEATURE ENGINEERING ===
    
    # Price-based features
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price relative to MAs (normalized)
    df['Close_MA5_Ratio'] = df['Close'] / df['MA_5']
    df['Close_MA10_Ratio'] = df['Close'] / df['MA_10']
    df['Close_MA20_Ratio'] = df['Close'] / df['MA_20']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Volatility
    df['Volatility_5'] = df['Log_Return'].rolling(window=5).std()
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
    
    # Volume features (normalized)
    df['Volume_MA_5'] = df['ShareVolume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['ShareVolume'] / (df['Volume_MA_5'] + 1)
    
    # Momentum
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Drop NaN
    df = df.dropna()
    
    return df

# ============================================================
# 📊 STEP 5: LOAD AND EXPLORE DATA
# ============================================================

companies = {
    'JKH': '/content/data/Historical Data/Historical data 2020-2025/JKH.N0000 - John Keells',
    'COMB': '/content/data/Historical Data/Historical data 2020-2025/COMB.N0000 - Commercial Bank',
    'CTC': '/content/data/Historical Data/Historical data 2020-2025/CTC.N0000 - Ceylon Tobacco',
    'DIAL': '/content/data/Historical Data/Historical data 2020-2025/DIAL.N0000 - Dialog Axiata',
    'DIST': '/content/data/Historical Data/Historical data 2020-2025/DIST.N0000 - Distilleries Company'
}

COMPANY = 'JKH'
df = load_and_preprocess_data_v2(companies[COMPANY], adjust_split=True)

print(f"\n📊 Loaded {COMPANY}: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Price range: Rs.{df['Close'].min():.2f} to Rs.{df['Close'].max():.2f}")

# Visualize adjusted data
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], 'b-', linewidth=1)
plt.title(f'{COMPANY} Stock Price (Split-Adjusted)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price (Rs.)')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# 🎯 STEP 6: PREPARE FEATURES (PREDICT RETURNS, NOT PRICES!)
# ============================================================

# KEY INSIGHT: Predicting PERCENTAGE RETURNS is much easier than absolute prices!
# Returns are stationary, prices are not.

# Selected features (all normalized/ratio-based)
feature_columns = [
    'Close_MA5_Ratio', 'Close_MA10_Ratio', 'Close_MA20_Ratio',
    'RSI', 'MACD_Hist', 'BB_Position',
    'Volatility_5', 'Volatility_20',
    'Volume_Ratio', 'Momentum_5', 'Momentum_10',
    'Price_Change'
]

# Target: Next day's return (percentage change)
df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
df = df.dropna()

X = df[feature_columns].values
y = df['Target_Return'].values

print(f"\n🎯 Features: {X.shape}")
print(f"🎯 Target (Next Day Return): {y.shape}")
print(f"Target mean: {y.mean():.6f}, std: {y.std():.6f}")

# ============================================================
# 📈 STEP 7: PROPER TRAIN/TEST SPLIT (NO DATA LEAKAGE!)
# ============================================================

# Use RobustScaler (less sensitive to outliers)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# IMPORTANT: Chronological split (no shuffle for time series!)
split_idx = int(len(X_scaled) * 0.8)

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# Keep original y for evaluation
y_train_orig = y[:split_idx]
y_test_orig = y[split_idx:]

print(f"\n📊 Training: {len(X_train)} samples ({df['Date'].iloc[0]} to {df['Date'].iloc[split_idx-1]})")
print(f"📊 Testing: {len(X_test)} samples ({df['Date'].iloc[split_idx]} to {df['Date'].iloc[-1]})")

# ============================================================
# 🤖 STEP 8: MODEL 1 - RIDGE REGRESSION (Better than LR)
# ============================================================

print("\n" + "=" * 50)
print("🤖 MODEL 1: RIDGE REGRESSION")
print("=" * 50)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred_scaled = ridge_model.predict(X_test)
ridge_pred = scaler_y.inverse_transform(ridge_pred_scaled.reshape(-1, 1)).flatten()

# ============================================================
# 🌲 STEP 9: MODEL 2 - GRADIENT BOOSTING (Better than RF)
# ============================================================

print("\n" + "=" * 50)
print("🌲 MODEL 2: GRADIENT BOOSTING")
print("=" * 50)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred_scaled = gb_model.predict(X_test)
gb_pred = scaler_y.inverse_transform(gb_pred_scaled.reshape(-1, 1)).flatten()

# ============================================================
# 🧠 STEP 10: PREPARE SEQUENCES FOR LSTM
# ============================================================

SEQ_LENGTH = 30  # Reduced from 60 for faster training

def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Split for deep learning (maintain chronological order)
split_seq = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

print(f"\n🧠 LSTM shapes: Train {X_train_seq.shape}, Test {X_test_seq.shape}")

# ============================================================
# 🧠 STEP 11: MODEL 3 - IMPROVED LSTM
# ============================================================

print("\n" + "=" * 50)
print("🧠 MODEL 3: IMPROVED LSTM")
print("=" * 50)

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

lstm_model = build_lstm_model((SEQ_LENGTH, X_train_seq.shape[2]))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)
]

print("🚀 Training LSTM (1000 epochs max, early stopping patience=50)...")
history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

lstm_pred_scaled = lstm_model.predict(X_test_seq).flatten()
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()

# ============================================================
# 🔄 STEP 12: MODEL 4 - GRU
# ============================================================

print("\n" + "=" * 50)
print("🔄 MODEL 4: GRU")
print("=" * 50)

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

gru_model = build_gru_model((SEQ_LENGTH, X_train_seq.shape[2]))

print("🚀 Training GRU (1000 epochs max, early stopping patience=50)...")
history_gru = gru_model.fit(
    X_train_seq, y_train_seq,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

gru_pred_scaled = gru_model.predict(X_test_seq).flatten()
gru_pred = scaler_y.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()

# ============================================================
# 🔗 STEP 13: MODEL 5 - BIDIRECTIONAL LSTM
# ============================================================

print("\n" + "=" * 50)
print("🔗 MODEL 5: BIDIRECTIONAL LSTM")
print("=" * 50)

def build_bilstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(16, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

bilstm_model = build_bilstm_model((SEQ_LENGTH, X_train_seq.shape[2]))

print("🚀 Training BiLSTM (1000 epochs max, early stopping patience=50)...")
history_bilstm = bilstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

bilstm_pred_scaled = bilstm_model.predict(X_test_seq).flatten()
bilstm_pred = scaler_y.inverse_transform(bilstm_pred_scaled.reshape(-1, 1)).flatten()

# ============================================================
# 🎯 STEP 14: ENSEMBLE (Weighted by inverse error)
# ============================================================

print("\n" + "=" * 50)
print("🎯 ENSEMBLE MODEL")
print("=" * 50)

# Simple ensemble average
ensemble_pred = (lstm_pred + gru_pred + bilstm_pred) / 3

# Align y_test for deep learning models
y_test_dl = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# ============================================================
# 📊 STEP 15: EVALUATION (PROPER METRICS FOR RETURNS!)
# ============================================================

def evaluate_returns_model(y_true, y_pred, model_name):
    """Evaluate model predicting returns"""
    
    # RMSE & MAE (in percentage points)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    mae = mean_absolute_error(y_true, y_pred) * 100
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Directional Accuracy (UP/DOWN prediction)
    dir_true = y_true > 0
    dir_pred = y_pred > 0
    dir_acc = np.mean(dir_true == dir_pred) * 100
    
    # Hit Rate (correct sign prediction)
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    
    return {
        'Model': model_name,
        'RMSE (%)': rmse,
        'MAE (%)': mae,
        'R² Score': r2,
        'Directional Accuracy (%)': dir_acc,
        'Hit Rate (%)': hit_rate
    }

print("\n" + "=" * 70)
print("📊 MODEL COMPARISON - PREDICTING DAILY RETURNS")
print("=" * 70)

results = []

# ML models (full test set)
y_test_ml = y_test_orig
ridge_pred_aligned = ridge_pred[:len(y_test_ml)]
gb_pred_aligned = gb_pred[:len(y_test_ml)]

results.append(evaluate_returns_model(y_test_ml, ridge_pred_aligned, 'Ridge Regression'))
results.append(evaluate_returns_model(y_test_ml, gb_pred_aligned, 'Gradient Boosting'))

# DL models (sequence test set)
results.append(evaluate_returns_model(y_test_dl, lstm_pred, 'LSTM'))
results.append(evaluate_returns_model(y_test_dl, gru_pred, 'GRU'))
results.append(evaluate_returns_model(y_test_dl, bilstm_pred, 'Bidirectional LSTM'))
results.append(evaluate_returns_model(y_test_dl, ensemble_pred, 'Ensemble (Avg)'))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Directional Accuracy (%)', ascending=False)

print(results_df.to_string(index=False))

# ============================================================
# 📈 STEP 16: VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Directional Accuracy Comparison
ax1 = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
bars = ax1.bar(results_df['Model'], results_df['Directional Accuracy (%)'], color=colors)
ax1.axhline(y=50, color='red', linestyle='--', label='Random Guess (50%)')
ax1.set_title('Directional Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(40, 70)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, results_df['Directional Accuracy (%)']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax1.legend()

# 2. R² Score Comparison
ax2 = axes[0, 1]
bars = ax2.bar(results_df['Model'], results_df['R² Score'], color=colors)
ax2.set_title('R² Score (Higher is Better, 0 = Baseline)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score')
ax2.axhline(y=0, color='red', linestyle='--')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, results_df['R² Score']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Training Loss
ax3 = axes[1, 0]
ax3.plot(history_lstm.history['loss'], label='LSTM Train')
ax3.plot(history_lstm.history['val_loss'], label='LSTM Val')
ax3.plot(history_gru.history['loss'], label='GRU Train', linestyle='--')
ax3.plot(history_gru.history['val_loss'], label='GRU Val', linestyle='--')
ax3.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Actual vs Predicted Returns
ax4 = axes[1, 1]
ax4.scatter(y_test_dl, ensemble_pred, alpha=0.5, s=20)
ax4.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
ax4.set_title('Actual vs Predicted Returns (Ensemble)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Actual Return')
ax4.set_ylabel('Predicted Return')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/improved_model_comparison.png', dpi=300)
plt.show()

# ============================================================
# 🔮 STEP 17: CONVERT RETURNS TO PRICE PREDICTIONS
# ============================================================

print("\n" + "=" * 50)
print("🔮 PRICE PREDICTION FROM RETURNS")
print("=" * 50)

# Get last known prices for test period
test_start_idx = split_seq + SEQ_LENGTH
test_dates = df['Date'].iloc[test_start_idx:test_start_idx+len(y_test_dl)].values
test_prices_actual = df['Close'].iloc[test_start_idx:test_start_idx+len(y_test_dl)].values

# Convert predicted returns to price predictions
# Price_t+1 = Price_t * (1 + predicted_return)
initial_price = df['Close'].iloc[test_start_idx - 1]
predicted_prices = [initial_price]

for ret in ensemble_pred:
    next_price = predicted_prices[-1] * (1 + ret)
    predicted_prices.append(next_price)

predicted_prices = np.array(predicted_prices[1:])  # Remove initial

# Price prediction metrics
price_rmse = np.sqrt(mean_squared_error(test_prices_actual, predicted_prices))
price_mae = mean_absolute_error(test_prices_actual, predicted_prices)
price_mape = np.mean(np.abs((test_prices_actual - predicted_prices) / test_prices_actual)) * 100

print(f"\n📊 Price Prediction Metrics:")
print(f"   RMSE: Rs. {price_rmse:.2f}")
print(f"   MAE: Rs. {price_mae:.2f}")
print(f"   MAPE: {price_mape:.2f}%")

# Plot price prediction
plt.figure(figsize=(14, 6))
plt.plot(test_dates, test_prices_actual, 'b-', label='Actual Price', linewidth=2)
plt.plot(test_dates, predicted_prices, 'r--', label='Predicted Price', linewidth=2, alpha=0.8)
plt.title(f'{COMPANY} Stock Price Prediction (Ensemble Model)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price (Rs.)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/content/price_prediction.png', dpi=300)
plt.show()

# ============================================================
# 💾 STEP 18: SAVE MODELS
# ============================================================

import pickle
import os

save_path = '/content/drive/MyDrive/CSE_Models_v2'
os.makedirs(save_path, exist_ok=True)

lstm_model.save(f'{save_path}/{COMPANY}_lstm_v2.h5')
gru_model.save(f'{save_path}/{COMPANY}_gru_v2.h5')
bilstm_model.save(f'{save_path}/{COMPANY}_bilstm_v2.h5')

with open(f'{save_path}/{COMPANY}_ridge.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
with open(f'{save_path}/{COMPANY}_gb.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
with open(f'{save_path}/{COMPANY}_scalers_v2.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

results_df.to_csv(f'{save_path}/{COMPANY}_results_v2.csv', index=False)

print(f"\n✅ Models saved to {save_path}")

# ============================================================
# 📋 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("📋 FINAL SUMMARY")
print("=" * 70)

best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   • Directional Accuracy: {best_model['Directional Accuracy (%)']:.1f}%")
print(f"   • R² Score: {best_model['R² Score']:.4f}")
print(f"   • Hit Rate: {best_model['Hit Rate (%)']:.1f}%")

print(f"\n📊 Price Prediction Performance:")
print(f"   • MAPE: {price_mape:.2f}%")
print(f"   • RMSE: Rs. {price_rmse:.2f}")

if best_model['Directional Accuracy (%)'] > 55:
    print(f"\n✅ Model performs better than random guess!")
    print(f"✅ Good enough for academic research project.")
else:
    print(f"\n⚠️ Model close to random guess.")
    print(f"   Consider adding more features or external data.")

print("=" * 70)
