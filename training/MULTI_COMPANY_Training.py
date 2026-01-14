# 📈 MULTI-COMPANY TRAINING - ALL 5 CSE STOCKS TOGETHER
# Combines: JKH, COMB, CTC, DIAL, DIST
# 5x more data = better pattern learning
# GPU-Accelerated on Google Colab T4

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏢 MULTI-COMPANY TRAINING - 5 CSE STOCKS COMBINED")
print("=" * 70)

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Company paths
companies = {
    'JKH': '/content/data/Historical Data/Historical data 2020-2025/JKH.N0000 - John Keells',
    'COMB': '/content/data/Historical Data/Historical data 2020-2025/COMB.N0000 - Commercial Bank',
    'CTC': '/content/data/Historical Data/Historical data 2020-2025/CTC.N0000 - Ceylon Tobacco',
    'DIAL': '/content/data/Historical Data/Historical data 2020-2025/DIAL.N0000 - Dialog Axiata',
    'DIST': '/content/data/Historical Data/Historical data 2020-2025/DIST.N0000 - Distilleries Company'
}

# Stock split dates (only JKH had a split)
stock_splits = {
    'JKH': ('2024-11-01', 10),  # 10:1 split
    'COMB': None,
    'CTC': None,
    'DIAL': None,
    'DIST': None
}

def load_and_process(company_path, company_name):
    """Load and process a single company's data"""
    df = pd.read_csv(f'{company_path}/Daily.csv')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset='Date', keep='first')
    
    # Stock split adjustment
    split_info = stock_splits.get(company_name)
    if split_info:
        split_date, split_ratio = split_info
        mask = df['Date'] < pd.to_datetime(split_date)
        for col in ['Open', 'High', 'Low', 'Close']:
            df.loc[mask, col] = df.loc[mask, col] / split_ratio
    
    # Fill missing
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Add company identifier
    df['Company'] = company_name
    
    return df

# Load all companies
print("\n📂 Loading all 5 companies...")
all_dfs = []
for name, path in companies.items():
    df = load_and_process(path, name)
    print(f"  ✅ {name}: {len(df)} records")
    all_dfs.append(df)

# Combine all
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n📊 Total combined: {len(combined_df)} records")

# Sort by date within each company
combined_df = combined_df.sort_values(['Company', 'Date']).reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING (Applied per company)
# ============================================================

print("\n🔧 Creating features per company...")

def create_features(df):
    """Create features for a single company"""
    df = df.copy()
    
    # Price features
    df['Price_Change'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
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
    macd = exp12 - exp26
    df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Position'] = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # Volatility
    df['Volatility_5'] = df['Log_Return'].rolling(window=5).std()
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
    
    # Volume
    df['Volume_Ratio'] = df['ShareVolume'] / (df['ShareVolume'].rolling(window=5).mean() + 1)
    
    # Momentum
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    
    # Lag features
    for lag in [1, 2, 3]:
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
    
    # TARGET: Binary classification (1=UP, 0=DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

# Apply features per company
featured_dfs = []
for name in companies.keys():
    company_df = combined_df[combined_df['Company'] == name].copy()
    company_df = create_features(company_df)
    company_df = company_df.dropna()
    featured_dfs.append(company_df)
    print(f"  ✅ {name}: {len(company_df)} samples after features")

# Combine featured data
final_df = pd.concat(featured_dfs, ignore_index=True)
print(f"\n📊 Total samples with features: {len(final_df)}")

# ============================================================
# PREPARE DATA
# ============================================================

feature_columns = [
    'Close_MA5_Ratio', 'Close_MA10_Ratio', 'Close_MA20_Ratio',
    'RSI', 'MACD_Hist', 'BB_Position', 'Volatility_5', 'Volatility_20',
    'Volume_Ratio', 'Momentum_5', 'Momentum_10', 'Price_Change',
    'Stoch_K', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3'
]

X = final_df[feature_columns].values
y = final_df['Target'].values

# Scale
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)

# Train/Test split (chronological per company, then combine)
# Use last 20% of each company as test
train_dfs = []
test_dfs = []

for name in companies.keys():
    company_data = final_df[final_df['Company'] == name]
    split_idx = int(len(company_data) * 0.8)
    train_dfs.append(company_data.iloc[:split_idx])
    test_dfs.append(company_data.iloc[split_idx:])

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

X_train = scaler_X.fit_transform(train_df[feature_columns].values)
X_test = scaler_X.transform(test_df[feature_columns].values)
y_train = train_df['Target'].values
y_test = test_df['Target'].values

print(f"\n📊 Training: {len(X_train)} | Testing: {len(X_test)}")
print(f"📊 Class balance (train): UP={y_train.sum()}, DOWN={len(y_train)-y_train.sum()}")

# ============================================================
# ML MODELS
# ============================================================

print("\n" + "=" * 60)
print("🌲 ML MODELS ON COMBINED DATA")
print("=" * 60)

# Random Forest
print("\n1️⃣ Random Forest (500 trees)...")
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, 
                                 n_jobs=-1, class_weight='balanced')
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100
print(f"   ✅ Random Forest: {rf_acc:.2f}%")

# Gradient Boosting
print("\n2️⃣ Gradient Boosting...")
gb_clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, 
                                     learning_rate=0.05, random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred) * 100
print(f"   ✅ Gradient Boosting: {gb_acc:.2f}%")

# ============================================================
# DEEP LEARNING
# ============================================================

print("\n" + "=" * 60)
print("🧠 DEEP LEARNING ON COMBINED DATA")
print("=" * 60)

SEQ_LENGTH = 30

def create_sequences_per_company(df, feature_cols, scaler, seq_length=30):
    """Create sequences maintaining company boundaries"""
    all_X, all_y = [], []
    
    for name in companies.keys():
        company_data = df[df['Company'] == name].copy()
        X_company = scaler.transform(company_data[feature_cols].values)
        y_company = company_data['Target'].values
        
        for i in range(len(X_company) - seq_length):
            all_X.append(X_company[i:i+seq_length])
            all_y.append(y_company[i+seq_length])
    
    return np.array(all_X), np.array(all_y)

X_train_seq, y_train_seq = create_sequences_per_company(train_df, feature_columns, scaler_X, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences_per_company(test_df, feature_columns, scaler_X, SEQ_LENGTH)

print(f"Sequence shapes: Train {X_train_seq.shape}, Test {X_test_seq.shape}")

# Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=1e-6, mode='max', verbose=1)
]

# GRU
print("\n3️⃣ GRU Classifier...")
gru = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    GRU(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    GRU(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
gru.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

gru.fit(X_train_seq, y_train_seq, epochs=500, batch_size=64, validation_split=0.2,
        callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

gru_pred = (gru.predict(X_test_seq).flatten() > 0.5).astype(int)
gru_acc = accuracy_score(y_test_seq, gru_pred) * 100
print(f"   ✅ GRU: {gru_acc:.2f}%")

# LSTM
print("\n4️⃣ LSTM Classifier...")
lstm = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

lstm.fit(X_train_seq, y_train_seq, epochs=500, batch_size=64, validation_split=0.2,
         callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

lstm_pred = (lstm.predict(X_test_seq).flatten() > 0.5).astype(int)
lstm_acc = accuracy_score(y_test_seq, lstm_pred) * 100
print(f"   ✅ LSTM: {lstm_acc:.2f}%")

# BiLSTM
print("\n5️⃣ BiLSTM Classifier...")
bilstm = Sequential([
    Input(shape=(SEQ_LENGTH, X_train_seq.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
bilstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

bilstm.fit(X_train_seq, y_train_seq, epochs=500, batch_size=64, validation_split=0.2,
           callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

bilstm_pred = (bilstm.predict(X_test_seq).flatten() > 0.5).astype(int)
bilstm_acc = accuracy_score(y_test_seq, bilstm_pred) * 100
print(f"   ✅ BiLSTM: {bilstm_acc:.2f}%")

# ============================================================
# PER-COMPANY EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("📊 PER-COMPANY ACCURACY (Random Forest)")
print("=" * 60)

for name in companies.keys():
    company_test = test_df[test_df['Company'] == name]
    X_comp = scaler_X.transform(company_test[feature_columns].values)
    y_comp = company_test['Target'].values
    pred_comp = rf_clf.predict(X_comp)
    acc_comp = accuracy_score(y_comp, pred_comp) * 100
    print(f"   {name}: {acc_comp:.2f}%")

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("📊 FINAL RESULTS - MULTI-COMPANY TRAINING")
print("=" * 70)

results = [
    {'Model': 'Random Forest', 'Accuracy (%)': rf_acc},
    {'Model': 'Gradient Boosting', 'Accuracy (%)': gb_acc},
    {'Model': 'GRU', 'Accuracy (%)': gru_acc},
    {'Model': 'LSTM', 'Accuracy (%)': lstm_acc},
    {'Model': 'BiLSTM', 'Accuracy (%)': bilstm_acc},
]

results_df = pd.DataFrame(results).sort_values('Accuracy (%)', ascending=False)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n🏆 BEST: {best['Model']} with {best['Accuracy (%)']:.2f}%")

# Compare with single company
print(f"\n📈 Improvement over single company (~60%): {best['Accuracy (%)'] - 60:.2f}%")

# Save models
save_path = '/content/drive/MyDrive/CSE_MultiCompany'
os.makedirs(save_path, exist_ok=True)
gru.save(f'{save_path}/multi_company_gru.keras')
lstm.save(f'{save_path}/multi_company_lstm.keras')

import pickle
with open(f'{save_path}/multi_company_rf.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
with open(f'{save_path}/multi_company_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

print(f"\n✅ Models saved to {save_path}")

# Visualization
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_df)))
bars = plt.barh(results_df['Model'], results_df['Accuracy (%)'], color=colors)
plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
plt.axvline(x=60, color='orange', linestyle='--', linewidth=2, label='Single Company (~60%)')
plt.xlabel('Accuracy (%)')
plt.title('Multi-Company Training Results (5 CSE Stocks)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('/content/multi_company_results.png', dpi=300)
plt.show()

print("=" * 70)
