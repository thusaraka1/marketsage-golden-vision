# 📈 OPTION 2: CLASSIFICATION MODEL - PREDICT UP/DOWN DIRECTLY
# Binary classification instead of regression
# GPU-Accelerated on Google Colab T4

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📊 CLASSIFICATION MODEL - PREDICT UP/DOWN")
print("=" * 60)

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    
    # Williams %R
    df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-10)
    
    # Lag features
    for lag in [1, 2, 3]:
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
    
    # Day of week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
    
    # TARGET: Binary classification (1=UP, 0=DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    
    return df

df = load_and_process()
print(f"✅ Loaded {len(df)} samples")

# Check class balance
print(f"\n📊 Class Distribution:")
print(f"   UP (1):   {(df['Target']==1).sum()} ({(df['Target']==1).mean()*100:.1f}%)")
print(f"   DOWN (0): {(df['Target']==0).sum()} ({(df['Target']==0).mean()*100:.1f}%)")

# Features
feature_columns = ['Close_MA5_Ratio', 'Close_MA10_Ratio', 'Close_MA20_Ratio',
                   'RSI', 'MACD_Hist', 'BB_Position', 'Volatility_5', 'Volatility_20',
                   'Volume_Ratio', 'Momentum_5', 'Momentum_10', 'Price_Change',
                   'Stoch_K', 'Williams_R', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'IsFriday']

X = df[feature_columns].values
y = df['Target'].values

scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)

# Split
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# ML CLASSIFIERS
# ============================================================

print("\n" + "=" * 60)
print("🌲 ML CLASSIFICATION MODELS")
print("=" * 60)

# Random Forest
print("\n1️⃣ Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100
print(f"   ✅ Random Forest: {rf_acc:.2f}%")

# Gradient Boosting
print("\n2️⃣ Gradient Boosting...")
gb_clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred) * 100
print(f"   ✅ Gradient Boosting: {gb_acc:.2f}%")

# ============================================================
# DEEP LEARNING CLASSIFIERS
# ============================================================

print("\n" + "=" * 60)
print("🧠 DEEP LEARNING CLASSIFICATION")
print("=" * 60)

SEQ_LENGTH = 30

def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)
split_seq = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

print(f"Sequence shape: {X_train_seq.shape}")

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=1e-6, mode='max', verbose=1)
]

# GRU Classifier
print("\n3️⃣ GRU Classifier...")
gru_clf = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    GRU(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    GRU(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])
gru_clf.compile(optimizer=Adam(learning_rate=0.001), 
                loss='binary_crossentropy',  # Binary cross-entropy for classification
                metrics=['accuracy'])

history_gru = gru_clf.fit(
    X_train_seq, y_train_seq,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

gru_pred_prob = gru_clf.predict(X_test_seq).flatten()
gru_pred = (gru_pred_prob > 0.5).astype(int)
gru_acc = accuracy_score(y_test_seq, gru_pred) * 100
print(f"   ✅ GRU Classifier: {gru_acc:.2f}%")

# LSTM Classifier
print("\n4️⃣ LSTM Classifier...")
lstm_clf = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_clf.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

history_lstm = lstm_clf.fit(
    X_train_seq, y_train_seq,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

lstm_pred_prob = lstm_clf.predict(X_test_seq).flatten()
lstm_pred = (lstm_pred_prob > 0.5).astype(int)
lstm_acc = accuracy_score(y_test_seq, lstm_pred) * 100
print(f"   ✅ LSTM Classifier: {lstm_acc:.2f}%")

# BiLSTM Classifier
print("\n5️⃣ BiLSTM Classifier...")
bilstm_clf = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    Bidirectional(LSTM(32, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(16, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
bilstm_clf.compile(optimizer=Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

history_bilstm = bilstm_clf.fit(
    X_train_seq, y_train_seq,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

bilstm_pred_prob = bilstm_clf.predict(X_test_seq).flatten()
bilstm_pred = (bilstm_pred_prob > 0.5).astype(int)
bilstm_acc = accuracy_score(y_test_seq, bilstm_pred) * 100
print(f"   ✅ BiLSTM Classifier: {bilstm_acc:.2f}%")

# ============================================================
# ENSEMBLE (Voting)
# ============================================================

print("\n" + "=" * 60)
print("🔗 ENSEMBLE VOTING")
print("=" * 60)

# Average probability then threshold
ensemble_prob = (gru_pred_prob + lstm_pred_prob + bilstm_pred_prob) / 3
ensemble_pred = (ensemble_prob > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test_seq, ensemble_pred) * 100
print(f"   ✅ DL Ensemble: {ensemble_acc:.2f}%")

# Majority voting
majority_vote = ((gru_pred + lstm_pred + bilstm_pred) >= 2).astype(int)
majority_acc = accuracy_score(y_test_seq, majority_vote) * 100
print(f"   ✅ Majority Voting: {majority_acc:.2f}%")

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("📊 CLASSIFICATION RESULTS")
print("=" * 70)

results = [
    {'Model': 'Random Forest', 'Accuracy (%)': rf_acc},
    {'Model': 'Gradient Boosting', 'Accuracy (%)': gb_acc},
    {'Model': 'GRU Classifier', 'Accuracy (%)': gru_acc},
    {'Model': 'LSTM Classifier', 'Accuracy (%)': lstm_acc},
    {'Model': 'BiLSTM Classifier', 'Accuracy (%)': bilstm_acc},
    {'Model': 'DL Ensemble (Avg)', 'Accuracy (%)': ensemble_acc},
    {'Model': 'Majority Voting', 'Accuracy (%)': majority_acc},
]

results_df = pd.DataFrame(results).sort_values('Accuracy (%)', ascending=False)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n🏆 BEST: {best['Model']} with {best['Accuracy (%)']:.2f}%")

# Classification Report for best model
print("\n📋 Classification Report (Best Model):")
if best['Model'] == 'GRU Classifier':
    print(classification_report(y_test_seq, gru_pred, target_names=['DOWN', 'UP']))
elif best['Model'] == 'LSTM Classifier':
    print(classification_report(y_test_seq, lstm_pred, target_names=['DOWN', 'UP']))

# Save best model
save_path = '/content/drive/MyDrive/CSE_Classification'
os.makedirs(save_path, exist_ok=True)
gru_clf.save(f'{save_path}/gru_classifier.keras')
lstm_clf.save(f'{save_path}/lstm_classifier.keras')

print(f"\n✅ Models saved to {save_path}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_df)))
bars = ax1.barh(results_df['Model'], results_df['Accuracy (%)'], color=colors)
ax1.axvline(x=50, color='red', linestyle='--', label='Random (50%)')
ax1.set_xlabel('Accuracy (%)')
ax1.set_title('Classification Accuracy Comparison', fontweight='bold')
ax1.legend()

# Training curves
ax2 = axes[1]
ax2.plot(history_gru.history['accuracy'], label='GRU Train')
ax2.plot(history_gru.history['val_accuracy'], label='GRU Val')
ax2.plot(history_lstm.history['accuracy'], label='LSTM Train', linestyle='--')
ax2.plot(history_lstm.history['val_accuracy'], label='LSTM Val', linestyle='--')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training History', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/classification_results.png', dpi=300)
plt.show()

print("=" * 70)
