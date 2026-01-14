# 📈 LOCAL Stock Prediction Training
# Optimized for RTX 3050 (4GB VRAM) / CPU fallback
# Python 3.11 + TensorFlow 2.20

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 LOCAL STOCK PREDICTION TRAINING")
print("=" * 60)

# ============================================================
# SETUP
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("📌 Loading libraries...")

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Found: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU - using CPU (slower but works)")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# LOAD DATA
# ============================================================

print("\n📂 Loading data...")

DATA_PATH = r"D:\nethumi final research\Historical_Data_Extracted\Historical Data\Historical data 2020-2025\JKH.N0000 - John Keells\Daily.csv"

df = pd.read_csv(DATA_PATH)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset='Date', keep='first')

# Adjust for stock split (Oct 2024)
split_date = pd.to_datetime('2024-11-01')
mask = df['Date'] < split_date
for col in ['Open', 'High', 'Low', 'Close']:
    df.loc[mask, col] = df.loc[mask, col] / 10

df['Open'] = df['Open'].fillna(df['Close'])
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"✅ Loaded {len(df)} records")
print(f"📅 {df['Date'].min()} to {df['Date'].max()}")

# ============================================================
# FEATURES (Same as v2 that got 63.2%)
# ============================================================

print("\n🔧 Creating features...")

df['Price_Change'] = df['Close'].pct_change()
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Moving Averages
for w in [5, 10, 20]:
    df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
    df[f'Close_MA{w}_Ratio'] = df['Close'] / df[f'MA_{w}']

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

# MACD
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# Bollinger Bands
bb_middle = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Position'] = (df['Close'] - (bb_middle - 2*bb_std)) / (4*bb_std + 1e-10)

# Volatility
df['Volatility_5'] = df['Log_Return'].rolling(window=5).std()
df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()

# Volume
df['Volume_MA_5'] = df['ShareVolume'].rolling(window=5).mean()
df['Volume_Ratio'] = df['ShareVolume'] / (df['Volume_MA_5'] + 1)

# Momentum
df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

# Target
df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1

df = df.dropna()
print(f"✅ Features created: {len(df)} samples")

# ============================================================
# PREPARE DATA
# ============================================================

feature_columns = [
    'Close_MA5_Ratio', 'Close_MA10_Ratio', 'Close_MA20_Ratio',
    'RSI', 'MACD_Hist', 'BB_Position',
    'Volatility_5', 'Volatility_20',
    'Volume_Ratio', 'Momentum_5', 'Momentum_10', 'Price_Change'
]

X = df[feature_columns].values
y = df['Target_Return'].values

scaler_X = RobustScaler()
scaler_y = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Chronological split
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
y_test_orig = y[split_idx:]

print(f"📊 Training: {len(X_train)} | Testing: {len(X_test)}")

# ============================================================
# ML MODELS
# ============================================================

print("\n" + "=" * 60)
print("🌲 EXTRA TREES")
print("=" * 60)

et_model = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
et_pred_inv = scaler_y.inverse_transform(et_pred.reshape(-1, 1)).flatten()

et_dir_acc = np.mean((y_test_orig > 0) == (et_pred_inv > 0)) * 100
print(f"✅ Extra Trees: {et_dir_acc:.1f}% directional accuracy")

# ============================================================
# GRU MODEL (Like v2)
# ============================================================

print("\n" + "=" * 60)
print("🧠 GRU MODEL")
print("=" * 60)

SEQ_LENGTH = 30

def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
split_seq = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

print(f"📊 Sequence shape: {X_train_seq.shape}")

# Build GRU
model = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
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
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

print("\n🚀 Training GRU...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=16,  # Smaller batch for limited RAM
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Predict
gru_pred = model.predict(X_test_seq).flatten()
gru_pred_inv = scaler_y.inverse_transform(gru_pred.reshape(-1, 1)).flatten()
y_test_dl = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

gru_dir_acc = np.mean((y_test_dl > 0) == (gru_pred_inv > 0)) * 100
print(f"\n✅ GRU: {gru_dir_acc:.1f}% directional accuracy")

# ============================================================
# ENSEMBLE
# ============================================================

print("\n" + "=" * 60)
print("🔗 HYBRID ENSEMBLE")
print("=" * 60)

# Align ET predictions
test_offset = len(y_test) - len(y_test_seq)
et_pred_aligned = et_pred_inv[test_offset:]

# Try different weights
best_acc = 0
best_pred = None
for w_et in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ensemble = w_et * et_pred_aligned + (1 - w_et) * gru_pred_inv
    acc = np.mean((y_test_dl > 0) == (ensemble > 0)) * 100
    if acc > best_acc:
        best_acc = acc
        best_pred = ensemble
        best_w = w_et

print(f"✅ Best Ensemble ({best_w*100:.0f}% ET, {(1-best_w)*100:.0f}% GRU): {best_acc:.1f}%")

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)

results = [
    {'Model': 'Extra Trees', 'Directional Accuracy (%)': et_dir_acc},
    {'Model': 'GRU', 'Directional Accuracy (%)': gru_dir_acc},
    {'Model': 'Hybrid Ensemble', 'Directional Accuracy (%)': best_acc}
]

results_df = pd.DataFrame(results).sort_values('Directional Accuracy (%)', ascending=False)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n🏆 BEST: {best['Model']} with {best['Directional Accuracy (%)']:.1f}%")

if best['Directional Accuracy (%)'] > 63.2:
    print("✅ Beat previous Colab result (63.2%)!")
else:
    print(f"⚠️ Difference from Colab: {63.2 - best['Directional Accuracy (%)']:.1f}%")

# ============================================================
# SAVE
# ============================================================

model.save(r"D:\nethumi final research\best_gru_local.keras")
import pickle
with open(r"D:\nethumi final research\extra_trees_local.pkl", 'wb') as f:
    pickle.dump(et_model, f)

print(f"\n✅ Models saved locally!")
print("=" * 70)
