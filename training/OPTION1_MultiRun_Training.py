# 📈 OPTION 1: MULTI-RUN TRAINING - KEEP BEST MODEL
# Trains 5 times, keeps the best performing model
# GPU-Accelerated on Google Colab T4

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🔄 MULTI-RUN TRAINING - TRAIN 5x, KEEP BEST")
print("=" * 60)

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# LOAD DATA
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import zipfile
DATA_PATH = '/content/drive/MyDrive/Historical Data.zip'
with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/data')

def load_and_process():
    df = pd.read_csv('/content/data/Historical Data/Historical data 2020-2025/JKH.N0000 - John Keells/Daily.csv')
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
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Features
    df['Price_Change'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    for w in [5, 10, 20]:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'Close_MA{w}_Ratio'] = df['Close'] / df[f'MA_{w}']
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp12 - exp26) - (exp12 - exp26).ewm(span=9, adjust=False).mean()
    
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Position'] = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    df['Volatility_5'] = df['Log_Return'].rolling(window=5).std()
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
    df['Volume_Ratio'] = df['ShareVolume'] / (df['ShareVolume'].rolling(window=5).mean() + 1)
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df = df.dropna()
    
    return df

df = load_and_process()
print(f"✅ Loaded {len(df)} samples")

# Features
feature_columns = ['Close_MA5_Ratio', 'Close_MA10_Ratio', 'Close_MA20_Ratio',
                   'RSI', 'MACD_Hist', 'BB_Position', 'Volatility_5', 'Volatility_20',
                   'Volume_Ratio', 'Momentum_5', 'Momentum_10', 'Price_Change']

X = df[feature_columns].values
y = df['Target_Return'].values

scaler_X = RobustScaler()
scaler_y = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Sequences
SEQ_LENGTH = 30

def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
split_seq = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]

y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# MULTI-RUN TRAINING (5 RUNS)
# ============================================================

NUM_RUNS = 5
best_accuracy = 0
best_model = None
all_results = []

print("\n" + "=" * 60)
print(f"🔄 TRAINING {NUM_RUNS} TIMES - KEEPING BEST")
print("=" * 60)

for run in range(1, NUM_RUNS + 1):
    print(f"\n{'='*20} RUN {run}/{NUM_RUNS} {'='*20}")
    
    # Clear session for fresh start
    tf.keras.backend.clear_session()
    
    # Set different seed for each run to get variety
    tf.random.set_seed(run * 42)
    np.random.seed(run * 42)
    
    # Build GRU model
    model = Sequential([
        Input(shape=(SEQ_LENGTH, X_train.shape[2])),
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
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    pred = model.predict(X_test, verbose=0).flatten()
    pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
    
    dir_acc = np.mean((y_test_inv > 0) == (pred_inv > 0)) * 100
    
    print(f"   Run {run}: Directional Accuracy = {dir_acc:.2f}%")
    all_results.append({'Run': run, 'Accuracy': dir_acc})
    
    # Keep best
    if dir_acc > best_accuracy:
        best_accuracy = dir_acc
        best_model = model
        best_run = run
        print(f"   ⭐ NEW BEST!")

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("📊 ALL RUNS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

print(f"\n📈 Statistics:")
print(f"   Mean: {results_df['Accuracy'].mean():.2f}%")
print(f"   Std:  {results_df['Accuracy'].std():.2f}%")
print(f"   Min:  {results_df['Accuracy'].min():.2f}%")
print(f"   Max:  {results_df['Accuracy'].max():.2f}%")

print(f"\n🏆 BEST MODEL: Run {best_run} with {best_accuracy:.2f}%")

# Save best model
save_path = '/content/drive/MyDrive/CSE_Best_Model'
os.makedirs(save_path, exist_ok=True)
best_model.save(f'{save_path}/best_gru_multirun.keras')
print(f"\n✅ Best model saved to {save_path}")

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(results_df['Run'], results_df['Accuracy'], color='steelblue')
plt.axhline(y=50, color='red', linestyle='--', label='Random (50%)')
plt.axhline(y=results_df['Accuracy'].mean(), color='green', linestyle='--', label=f'Mean ({results_df["Accuracy"].mean():.1f}%)')
plt.xlabel('Run')
plt.ylabel('Directional Accuracy (%)')
plt.title('Multi-Run Training Results')
plt.legend()
plt.savefig('/content/multirun_results.png', dpi=300)
plt.show()

print("=" * 60)
