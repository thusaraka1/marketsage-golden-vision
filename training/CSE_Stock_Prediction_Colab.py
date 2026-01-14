# 📈 Colombo Stock Exchange - Multi-Model Stock Prediction
# GPU-Accelerated Training on Google Colab (T4)
# Models: Linear Regression, Random Forest, LSTM, GRU, Ensemble

# ============================================================
# 🔧 STEP 1: SETUP & GPU CHECK
# ============================================================

# Check GPU availability
import tensorflow as tf
print("=" * 50)
print("🔧 GPU CHECK")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU is enabled! Training will be accelerated.")
else:
    print("⚠️ No GPU found. Go to Runtime > Change runtime type > T4 GPU")

# Install required packages
# !pip install -q scikit-learn pandas numpy matplotlib seaborn ta

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

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

print("✅ All libraries imported successfully!")

# ============================================================
# 📂 STEP 3: LOAD DATA FROM GOOGLE DRIVE
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# Update this path to your data location
DATA_PATH = '/content/drive/MyDrive/Historical Data.zip'

# Unzip data
import zipfile
with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/data')

print("✅ Data extracted successfully!")

# ============================================================
# 📊 STEP 4: DATA PREPROCESSING
# ============================================================

def load_and_preprocess_data(company_folder):
    """Load and preprocess data for a company"""
    
    # Load daily data
    df = pd.read_csv(f'{company_folder}/Daily.csv')
    
    # Rename columns for easier handling
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Fill missing values
    df['Open'] = df['Open'].fillna(df['Close'])
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Add Technical Indicators
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Price Change
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=21).std()
    
    # Drop NaN rows created by indicators
    df = df.dropna()
    
    return df

# Load data for all companies
companies = {
    'JKH': '/content/data/Historical Data/Historical data 2020-2025/JKH.N0000 - John Keells',
    'COMB': '/content/data/Historical Data/Historical data 2020-2025/COMB.N0000 - Commercial Bank',
    'CTC': '/content/data/Historical Data/Historical data 2020-2025/CTC.N0000 - Ceylon Tobacco',
    'DIAL': '/content/data/Historical Data/Historical data 2020-2025/DIAL.N0000 - Dialog Axiata',
    'DIST': '/content/data/Historical Data/Historical data 2020-2025/DIST.N0000 - Distilleries Company'
}

# Select company to train (change this)
COMPANY = 'JKH'
df = load_and_preprocess_data(companies[COMPANY])
print(f"\n📊 Loaded {COMPANY} data: {len(df)} records")
print(df.head())

# ============================================================
# 🔀 STEP 5: PREPARE FEATURES & TARGET
# ============================================================

# Features for ML models
feature_columns = ['Open', 'High', 'Low', 'Close', 'ShareVolume', 
                   'MA_7', 'MA_21', 'RSI', 'MACD', 'Volatility']

X = df[feature_columns].values
y = df['Close'].shift(-1).dropna().values  # Predict next day's close
X = X[:-1]  # Remove last row to match y

print(f"\n🎯 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# ============================================================
# 📈 STEP 6: TIME SERIES SPLIT (PROPER VALIDATION)
# ============================================================

# For time series, we use forward-chaining validation
# This prevents data leakage from future to past

# Simple train/test split (80/20 chronologically)
split_idx = int(len(X_scaled) * 0.8)

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

print(f"\n📊 Training set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# ============================================================
# 🤖 STEP 7: MODEL 1 - LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 50)
print("🤖 MODEL 1: LINEAR REGRESSION")
print("=" * 50)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Cross-validation score
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=tscv, scoring='r2')
print(f"Cross-Validation R² Scores: {lr_cv_scores}")
print(f"Mean CV R²: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# ============================================================
# 🌲 STEP 8: MODEL 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 50)
print("🌲 MODEL 2: RANDOM FOREST")
print("=" * 50)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Cross-validation score
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=tscv, scoring='r2')
print(f"Cross-Validation R² Scores: {rf_cv_scores}")
print(f"Mean CV R²: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n🔑 Feature Importance:")
print(feature_importance)

# ============================================================
# 🧠 STEP 9: MODEL 3 - LSTM (GPU ACCELERATED)
# ============================================================

print("\n" + "=" * 50)
print("🧠 MODEL 3: LSTM (GPU ACCELERATED)")
print("=" * 50)

# Prepare sequences for LSTM
def create_sequences(X, y, seq_length=60):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

SEQ_LENGTH = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Split for LSTM
split_idx_seq = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
y_train_seq, y_test_seq = y_seq[:split_idx_seq], y_seq[split_idx_seq:]

print(f"LSTM Training shape: {X_train_seq.shape}")
print(f"LSTM Test shape: {X_test_seq.shape}")

# Build LSTM Model
with tf.device('/GPU:0'):
    lstm_model = Sequential([
        Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

lstm_model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
]

# Train LSTM
print("\n🚀 Training LSTM on GPU...")
history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

lstm_pred = lstm_model.predict(X_test_seq).flatten()

# ============================================================
# 🔄 STEP 10: MODEL 4 - GRU (GPU ACCELERATED)
# ============================================================

print("\n" + "=" * 50)
print("🔄 MODEL 4: GRU (GPU ACCELERATED)")
print("=" * 50)

with tf.device('/GPU:0'):
    gru_model = Sequential([
        Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    gru_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

# Train GRU
print("\n🚀 Training GRU on GPU...")
history_gru = gru_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

gru_pred = gru_model.predict(X_test_seq).flatten()

# ============================================================
# 🔗 STEP 11: MODEL 5 - BIDIRECTIONAL LSTM
# ============================================================

print("\n" + "=" * 50)
print("🔗 MODEL 5: BIDIRECTIONAL LSTM")
print("=" * 50)

with tf.device('/GPU:0'):
    bilstm_model = Sequential([
        Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    bilstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

# Train BiLSTM
print("\n🚀 Training Bidirectional LSTM on GPU...")
history_bilstm = bilstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

bilstm_pred = bilstm_model.predict(X_test_seq).flatten()

# ============================================================
# 🎯 STEP 12: ENSEMBLE MODEL (WEIGHTED AVERAGE)
# ============================================================

print("\n" + "=" * 50)
print("🎯 ENSEMBLE MODEL (WEIGHTED AVERAGE)")
print("=" * 50)

# Simple ensemble: average of all deep learning models
ensemble_pred = (lstm_pred + gru_pred + bilstm_pred) / 3

# ============================================================
# 📊 STEP 13: EVALUATION & COMPARISON
# ============================================================

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display metrics"""
    # Inverse transform to original scale
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # Directional Accuracy (predicting UP or DOWN correctly)
    direction_true = np.diff(y_true_orig) > 0
    direction_pred = np.diff(y_pred_orig) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'Directional Accuracy (%)': directional_accuracy,
        'MAPE (%)': mape
    }

# Evaluate all models
# Note: For fair comparison, we need to align predictions with same test set
# Deep learning models use sequence data, so test set is shorter

print("\n" + "=" * 70)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 70)

# DL models evaluation (same test set)
results = []
results.append(evaluate_model(y_test_seq, lstm_pred, 'LSTM'))
results.append(evaluate_model(y_test_seq, gru_pred, 'GRU'))
results.append(evaluate_model(y_test_seq, bilstm_pred, 'Bidirectional LSTM'))
results.append(evaluate_model(y_test_seq, ensemble_pred, 'Ensemble (Avg)'))

# ML models (using aligned portion)
lr_pred_aligned = lr_pred[-len(y_test_seq):]
rf_pred_aligned = rf_pred[-len(y_test_seq):]
y_test_aligned = y_test[-len(y_test_seq):]

results.append(evaluate_model(y_test_aligned, lr_pred_aligned, 'Linear Regression'))
results.append(evaluate_model(y_test_aligned, rf_pred_aligned, 'Random Forest'))

# Create comparison dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² Score', ascending=False)
print(results_df.to_string(index=False))

# ============================================================
# 📈 STEP 14: VISUALIZATION
# ============================================================

# Plot 1: Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# R² Score Comparison
ax1 = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
bars = ax1.bar(results_df['Model'], results_df['R² Score'], color=colors)
ax1.set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim(0, 1)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, results_df['R² Score']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# RMSE Comparison
ax2 = axes[0, 1]
bars = ax2.bar(results_df['Model'], results_df['RMSE'], color=colors)
ax2.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, results_df['RMSE']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# Directional Accuracy
ax3 = axes[1, 0]
bars = ax3.bar(results_df['Model'], results_df['Directional Accuracy (%)'], color=colors)
ax3.set_title('Directional Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy (%)')
ax3.set_ylim(0, 100)
ax3.axhline(y=50, color='r', linestyle='--', label='Random Guess (50%)')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, results_df['Directional Accuracy (%)']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# Training Loss (LSTM)
ax4 = axes[1, 1]
ax4.plot(history_lstm.history['loss'], label='LSTM Train Loss')
ax4.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
ax4.plot(history_gru.history['loss'], label='GRU Train Loss', linestyle='--')
ax4.plot(history_gru.history['val_loss'], label='GRU Val Loss', linestyle='--')
ax4.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss (MSE)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(15, 6))

# Inverse transform for visualization
y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
lstm_pred_orig = scaler_y.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
ensemble_pred_orig = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()

plt.plot(y_test_orig, label='Actual Price', color='blue', linewidth=2)
plt.plot(lstm_pred_orig, label='LSTM Prediction', color='red', alpha=0.7)
plt.plot(ensemble_pred_orig, label='Ensemble Prediction', color='green', alpha=0.7)
plt.title(f'{COMPANY} Stock Price - Actual vs Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps')
plt.ylabel('Price (Rs.)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/content/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 💾 STEP 15: SAVE MODELS
# ============================================================

print("\n" + "=" * 50)
print("💾 SAVING MODELS")
print("=" * 50)

# Save to Google Drive
import pickle
import os

save_path = '/content/drive/MyDrive/CSE_Models'
os.makedirs(save_path, exist_ok=True)

# Save deep learning models
lstm_model.save(f'{save_path}/{COMPANY}_lstm_model.h5')
gru_model.save(f'{save_path}/{COMPANY}_gru_model.h5')
bilstm_model.save(f'{save_path}/{COMPANY}_bilstm_model.h5')

# Save ML models
with open(f'{save_path}/{COMPANY}_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open(f'{save_path}/{COMPANY}_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save scalers
with open(f'{save_path}/{COMPANY}_scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

# Save results
results_df.to_csv(f'{save_path}/{COMPANY}_results.csv', index=False)

print(f"✅ All models saved to {save_path}")

# ============================================================
# 🔮 STEP 16: MAKE 30-DAY PREDICTION
# ============================================================

print("\n" + "=" * 50)
print("🔮 30-DAY FUTURE PREDICTION")
print("=" * 50)

def predict_future(model, last_sequence, n_days=30):
    """Predict n days into the future"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_days):
        pred = model.predict(current_seq.reshape(1, SEQ_LENGTH, -1), verbose=0)[0, 0]
        predictions.append(pred)
        
        # Update sequence (simplified - just shift and add prediction as Close)
        new_row = current_seq[-1].copy()
        new_row[3] = pred  # Update Close price
        current_seq = np.vstack([current_seq[1:], new_row])
    
    return np.array(predictions)

# Use the best model (ensemble approach)
last_seq = X_seq[-1]

lstm_future = predict_future(lstm_model, last_seq, 30)
gru_future = predict_future(gru_model, last_seq, 30)
bilstm_future = predict_future(bilstm_model, last_seq, 30)

# Ensemble future prediction
ensemble_future = (lstm_future + gru_future + bilstm_future) / 3
ensemble_future_orig = scaler_y.inverse_transform(ensemble_future.reshape(-1, 1)).flatten()

# Generate future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

# Create prediction dataframe
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': ensemble_future_orig
})

print("\n📈 30-Day Prediction:")
print(future_df)

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(future_df['Date'], future_df['Predicted_Close'], 'g-o', label='Predicted Price')
plt.title(f'{COMPANY} - 30-Day Price Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Predicted Price (Rs.)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/content/30_day_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 📋 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("📋 FINAL SUMMARY")
print("=" * 70)
print(f"\n🏆 BEST MODEL: {results_df.iloc[0]['Model']}")
print(f"   • R² Score: {results_df.iloc[0]['R² Score']:.4f}")
print(f"   • RMSE: {results_df.iloc[0]['RMSE']:.2f}")
print(f"   • Directional Accuracy: {results_df.iloc[0]['Directional Accuracy (%)']:.1f}%")
print(f"\n📊 All models trained and saved successfully!")
print(f"📁 Models saved to: {save_path}")
print("=" * 70)
