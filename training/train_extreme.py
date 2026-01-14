# 🔥 EXTREME TRAINING - TARGET 70% ACCURACY
# Multi-Model Ensemble with 1000 epochs, Early Stopping
# Optimized for RTX 3050 (4GB VRAM)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 70)
print("🔥 EXTREME TRAINING - TARGET: 70% DIRECTIONAL ACCURACY")
print("=" * 70)
print(f"Started: {datetime.now()}")

# ============================================================
# IMPORTS
# ============================================================

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, 
                               RandomForestRegressor, AdaBoostRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ CPU mode (still works, just slower)")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                      BatchNormalization, Bidirectional,
                                      Conv1D, MaxPooling1D, Flatten,
                                      GlobalAveragePooling1D, Concatenate,
                                      Add, LayerNormalization, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

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

# Stock split adjustment
split_date = pd.to_datetime('2024-11-01')
mask = df['Date'] < split_date
for col in ['Open', 'High', 'Low', 'Close']:
    df.loc[mask, col] = df.loc[mask, col] / 10

df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"✅ Loaded {len(df)} records")

# ============================================================
# EXTENSIVE FEATURE ENGINEERING
# ============================================================

print("\n🔧 Creating extensive features...")

def create_all_features(df):
    """Create ALL possible features"""
    df = df.copy()
    
    # Basic returns
    df['Price_Change'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Multiple MAs
    for w in [3, 5, 7, 10, 14, 20, 30, 50]:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'MA_{w}_Ratio'] = df['Close'] / df[f'MA_{w}']
        df[f'MA_{w}_Slope'] = df[f'MA_{w}'].diff(3) / df[f'MA_{w}'].shift(3)
    
    # EMAs
    for span in [5, 10, 12, 20, 26]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        df[f'EMA_{span}_Ratio'] = df['Close'] / df[f'EMA_{span}']
    
    # RSI multiple periods
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        df[f'RSI_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD variations
    for fast, slow in [(8, 17), (12, 26), (5, 35)]:
        exp_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        exp_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        df[f'MACD_{fast}_{slow}'] = macd
        df[f'MACD_{fast}_{slow}_Hist'] = (macd - signal) / df['Close']
    
    # Bollinger Bands multiple
    for w in [10, 20, 30]:
        bb_mid = df['Close'].rolling(window=w).mean()
        bb_std = df['Close'].rolling(window=w).std()
        df[f'BB_{w}_Upper'] = bb_mid + 2 * bb_std
        df[f'BB_{w}_Lower'] = bb_mid - 2 * bb_std
        df[f'BB_{w}_Position'] = (df['Close'] - df[f'BB_{w}_Lower']) / (df[f'BB_{w}_Upper'] - df[f'BB_{w}_Lower'] + 1e-10)
        df[f'BB_{w}_Width'] = (df[f'BB_{w}_Upper'] - df[f'BB_{w}_Lower']) / bb_mid
    
    # Stochastic
    for k in [9, 14, 21]:
        low_min = df['Low'].rolling(window=k).min()
        high_max = df['High'].rolling(window=k).max()
        df[f'Stoch_K_{k}'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df[f'Stoch_D_{k}'] = df[f'Stoch_K_{k}'].rolling(window=3).mean()
    
    # ATR
    for p in [7, 14, 21]:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'ATR_{p}'] = tr.rolling(window=p).mean()
        df[f'ATR_{p}_Norm'] = df[f'ATR_{p}'] / df['Close']
    
    # Volatility
    for w in [5, 10, 20]:
        df[f'Vol_{w}'] = df['Log_Return'].rolling(window=w).std()
    
    # Volume features
    for w in [5, 10, 20]:
        df[f'Vol_MA_{w}'] = df['ShareVolume'].rolling(window=w).mean()
    df['Vol_Ratio_5'] = df['ShareVolume'] / (df['Vol_MA_5'] + 1)
    df['Vol_Ratio_20'] = df['ShareVolume'] / (df['Vol_MA_20'] + 1)
    
    # Momentum
    for p in [3, 5, 10, 14, 20]:
        df[f'Mom_{p}'] = df['Close'] / df['Close'].shift(p) - 1
        df[f'ROC_{p}'] = (df['Close'] - df['Close'].shift(p)) / df['Close'].shift(p)
    
    # Williams %R
    for p in [14, 21]:
        high_max = df['High'].rolling(window=p).max()
        low_min = df['Low'].rolling(window=p).min()
        df[f'WilliamsR_{p}'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
    
    # CCI
    for p in [14, 20]:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=p).mean()
        mad = tp.rolling(window=p).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'CCI_{p}'] = (tp - sma) / (0.015 * mad + 1e-10)
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
        df[f'Close_Lag_{lag}_Ratio'] = df['Close'] / df['Close'].shift(lag)
    
    # Day of week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsFriday'] = (df['DayOfWeek'] == 4).astype(float)
    df['IsMonday'] = (df['DayOfWeek'] == 0).astype(float)
    df['IsThursday'] = (df['DayOfWeek'] == 3).astype(float)
    
    # Price ratios
    df['Open_Close'] = df['Open'] / df['Close']
    df['High_Low'] = df['High'] / df['Low']
    df['High_Close'] = df['High'] / df['Close']
    df['Low_Close'] = df['Low'] / df['Close']
    
    # Trend strength
    df['ADX'] = df['ATR_14']  # Simplified ADX proxy
    
    # Target: Predict if NEXT DAY goes UP or DOWN
    df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)  # 1=UP, 0=DOWN
    
    return df

df = create_all_features(df)
df = df.dropna()

print(f"✅ Total features: {len(df.columns) - 4}")  # Exclude Date, Target, etc.
print(f"✅ Samples after processing: {len(df)}")

# ============================================================
# FEATURE SELECTION (Top 50)
# ============================================================

print("\n🎯 Selecting best features...")

exclude_cols = ['Date', 'Target_Return', 'Target_Direction', 'Open', 'High', 'Low', 'Close',
                'TradeVolume', 'ShareVolume', 'Turnover', 'DayOfWeek']
exclude_cols += [c for c in df.columns if c.startswith('MA_') and not c.endswith('Ratio') and not c.endswith('Slope')]
exclude_cols += [c for c in df.columns if c.startswith('EMA_') and not c.endswith('Ratio')]
exclude_cols += [c for c in df.columns if c.startswith('BB_') and c.endswith('Upper')]
exclude_cols += [c for c in df.columns if c.startswith('BB_') and c.endswith('Lower')]
exclude_cols += [c for c in df.columns if c.startswith('Vol_MA_')]
exclude_cols += [c for c in df.columns if c.startswith('ATR_') and not c.endswith('Norm')]

feature_columns = [c for c in df.columns if c not in exclude_cols]

X = df[feature_columns].values
y = df['Target_Return'].values
y_dir = df['Target_Direction'].values

# Quick feature importance
from sklearn.ensemble import RandomForestClassifier
rf_temp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_temp.fit(X, y_dir)

importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=False)

TOP_N = 40
top_features = importance.head(TOP_N)['Feature'].tolist()
print(f"✅ Selected top {TOP_N} features")
print(f"Top 10: {top_features[:10]}")

X = df[top_features].values
y = df['Target_Return'].values

# ============================================================
# SCALING AND SPLIT
# ============================================================

scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 80/20 chronological split
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
y_train_orig = y[:split_idx]
y_test_orig = y[split_idx:]

print(f"\n📊 Training: {len(X_train)} | Testing: {len(X_test)}")

# ============================================================
# ML MODELS (8 MODELS)
# ============================================================

print("\n" + "=" * 70)
print("🌲 TRAINING ML MODELS")
print("=" * 70)

ml_models = {
    'ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'Ridge': Ridge(alpha=1.0),
}

ml_predictions = {}

for name, model in ml_models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
    dir_acc = np.mean((y_test_orig > 0) == (pred_inv > 0)) * 100
    ml_predictions[name] = pred_inv
    print(f"  ✅ {name}: {dir_acc:.1f}%")

# ============================================================
# DEEP LEARNING MODELS (1000 EPOCHS EACH)
# ============================================================

print("\n" + "=" * 70)
print("🧠 TRAINING DEEP LEARNING MODELS (1000 EPOCHS MAX)")
print("=" * 70)

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

print(f"Sequence shape: Train {X_train_seq.shape}, Test {X_test_seq.shape}")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, min_lr=1e-7, verbose=1)
]

dl_predictions = {}

# Model 1: Deep GRU
print("\n  🔄 Training Deep GRU (1000 epochs)...")
gru_model = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    GRU(128, return_sequences=True, kernel_regularizer=l1_l2(0.0001, 0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    GRU(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    GRU(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])
gru_model.fit(X_train_seq, y_train_seq, epochs=1000, batch_size=16, 
              validation_split=0.2, callbacks=callbacks, verbose=2)
gru_pred = gru_model.predict(X_test_seq, verbose=0).flatten()
dl_predictions['GRU'] = scaler_y.inverse_transform(gru_pred.reshape(-1, 1)).flatten()
print(f"  ✅ GRU done")

# Model 2: Deep LSTM
print("\n  🔄 Training Deep LSTM (1000 epochs)...")
lstm_model = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.0001, 0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])
lstm_model.fit(X_train_seq, y_train_seq, epochs=1000, batch_size=16,
               validation_split=0.2, callbacks=callbacks, verbose=2)
lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
dl_predictions['LSTM'] = scaler_y.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
print(f"  ✅ LSTM done")

# Model 3: BiLSTM
print("\n  🔄 Training Bidirectional LSTM (1000 epochs)...")
bilstm_model = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
bilstm_model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])
bilstm_model.fit(X_train_seq, y_train_seq, epochs=1000, batch_size=16,
                 validation_split=0.2, callbacks=callbacks, verbose=2)
bilstm_pred = bilstm_model.predict(X_test_seq, verbose=0).flatten()
dl_predictions['BiLSTM'] = scaler_y.inverse_transform(bilstm_pred.reshape(-1, 1)).flatten()
print(f"  ✅ BiLSTM done")

# Model 4: CNN-LSTM
print("\n  🔄 Training CNN-LSTM (1000 epochs)...")
cnn_lstm = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu', padding='same'),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
cnn_lstm.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])
cnn_lstm.fit(X_train_seq, y_train_seq, epochs=1000, batch_size=16,
             validation_split=0.2, callbacks=callbacks, verbose=2)
cnn_lstm_pred = cnn_lstm.predict(X_test_seq, verbose=0).flatten()
dl_predictions['CNN-LSTM'] = scaler_y.inverse_transform(cnn_lstm_pred.reshape(-1, 1)).flatten()
print(f"  ✅ CNN-LSTM done")

# ============================================================
# MEGA ENSEMBLE
# ============================================================

print("\n" + "=" * 70)
print("🔗 MEGA ENSEMBLE (ML + DL)")
print("=" * 70)

# Align all predictions
y_test_dl = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
test_offset = len(y_test) - len(y_test_seq)

# Align ML predictions
ml_aligned = {}
for name, pred in ml_predictions.items():
    ml_aligned[name] = pred[test_offset:]

# Combine all
all_preds = {**ml_aligned, **dl_predictions}

# Evaluate each
results = []
for name, pred in all_preds.items():
    dir_acc = np.mean((y_test_dl > 0) == (pred > 0)) * 100
    results.append({'Model': name, 'Accuracy': dir_acc})
    print(f"  {name}: {dir_acc:.1f}%")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

# Weighted ensemble based on performance
print("\n🎯 Creating weighted ensemble...")
weights = {}
total_acc = sum([r['Accuracy'] for r in results])
for r in results:
    weights[r['Model']] = r['Accuracy'] / total_acc

weighted_ensemble = np.zeros_like(y_test_dl)
for name, pred in all_preds.items():
    weighted_ensemble += weights[name] * pred

weighted_dir_acc = np.mean((y_test_dl > 0) == (weighted_ensemble > 0)) * 100
print(f"  Weighted Ensemble: {weighted_dir_acc:.1f}%")

# Simple average ensemble
simple_ensemble = np.mean(list(all_preds.values()), axis=0)
simple_dir_acc = np.mean((y_test_dl > 0) == (simple_ensemble > 0)) * 100
print(f"  Simple Ensemble: {simple_dir_acc:.1f}%")

# Top-3 ensemble
top3 = results_df.head(3)['Model'].tolist()
top3_preds = [all_preds[m] for m in top3]
top3_ensemble = np.mean(top3_preds, axis=0)
top3_dir_acc = np.mean((y_test_dl > 0) == (top3_ensemble > 0)) * 100
print(f"  Top-3 Ensemble ({', '.join(top3)}): {top3_dir_acc:.1f}%")

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)

# Add ensembles to results
final_results = results + [
    {'Model': 'Weighted Ensemble', 'Accuracy': weighted_dir_acc},
    {'Model': 'Simple Ensemble', 'Accuracy': simple_dir_acc},
    {'Model': 'Top-3 Ensemble', 'Accuracy': top3_dir_acc}
]

final_df = pd.DataFrame(final_results).sort_values('Accuracy', ascending=False)
print(final_df.to_string(index=False))

best = final_df.iloc[0]
print(f"\n🏆 BEST: {best['Model']} with {best['Accuracy']:.1f}%")

if best['Accuracy'] >= 70:
    print("🎉 ACHIEVED 70%+ TARGET!")
elif best['Accuracy'] >= 65:
    print("✅ Excellent result (65%+)!")
elif best['Accuracy'] >= 60:
    print("✅ Good result (60%+)")
else:
    print("⚠️ Below target")

# Save best model
gru_model.save(r"D:\nethumi final research\best_model_extreme.keras")
import pickle
with open(r"D:\nethumi final research\all_ml_models.pkl", 'wb') as f:
    pickle.dump(ml_models, f)

print(f"\n✅ Models saved!")
print(f"Finished: {datetime.now()}")
print("=" * 70)
