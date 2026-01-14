"""
Ensemble Model Verification Script
- Loads the extracted BiLSTM + XGBoost ensemble model
- Tests it against historical CSE stock data
- Reports accuracy metrics
"""

import os
import numpy as np
import pandas as pd
import pickle
import glob
from datetime import datetime

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras

# XGBoost
import xgboost as xgb

# SKLearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========================
# CONFIGURATION
# ========================
MODEL_DIR = r"d:\nethumi final research\ENSEMBLE_EXTRACTED\ENSEMBLE_ACC_0.6214_20260106_0727_e50425"
DATA_DIR = r"d:\nethumi final research\Historical_Data_Reextracted\Historical Data"

SEQ_LENGTH = 20  # From metadata: Window: 20 Days

# ========================
# 1. LOAD MODELS
# ========================
print("=" * 60)
print("🔍 ENSEMBLE MODEL VERIFICATION")
print("=" * 60)

print("\n📦 Loading Models...")

# Load BiLSTM Model
bilstm_path = os.path.join(MODEL_DIR, "bi_lstm_model.keras")
bilstm_model = keras.models.load_model(bilstm_path)
print(f"✅ BiLSTM Model loaded from: {bilstm_path}")

# Load XGBoost Model
xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_path)
print(f"✅ XGBoost Model loaded from: {xgb_path}")

# Load Scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"✅ Scaler loaded from: {scaler_path}")

# Load Metadata
metadata_path = os.path.join(MODEL_DIR, "metadata.txt")
with open(metadata_path, 'r') as f:
    print(f"\n📋 Model Metadata:\n{f.read()}")

# ========================
# 2. LOAD & PREPARE DATA
# ========================
print("\n📊 Loading Historical Data...")

def load_all_companies():
    """Load daily data for all companies"""
    all_data = []
    
    # Check both time periods
    for period in ["Historical data 2020-2025", "Historical data 2015-2020"]:
        period_path = os.path.join(DATA_DIR, period)
        if not os.path.exists(period_path):
            continue
            
        for company_dir in os.listdir(period_path):
            company_path = os.path.join(period_path, company_dir, "Daily.csv")
            if os.path.exists(company_path):
                try:
                    df = pd.read_csv(company_path)
                    # Extract company code from folder name
                    company_code = company_dir.split(' - ')[0].strip()
                    df['Company'] = company_code
                    all_data.append(df)
                except Exception as e:
                    print(f"  ⚠️ Error loading {company_path}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return None

df_raw = load_all_companies()

if df_raw is None or len(df_raw) == 0:
    print("❌ No data loaded!")
    exit(1)

print(f"✅ Loaded {len(df_raw)} rows from {df_raw['Company'].nunique()} companies")
print(f"   Companies: {df_raw['Company'].unique()}")

# ========================
# 3. FEATURE ENGINEERING
# ========================
print("\n🔧 Engineering Features...")

def create_features(df):
    """Create features matching the training script"""
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Handle Date
    if 'Trade Date' in df.columns:
        df.rename(columns={'Trade Date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Handle Close price
    if 'Close (Rs.)' in df.columns:
        df.rename(columns={'Close (Rs.)': 'Close'}, inplace=True)
    
    # Clean numeric columns
    for col in ['Close', 'Open (Rs.)', 'High (Rs.)', 'Low (Rs.)', 'ShareVolume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing Close
    df = df.dropna(subset=['Close'])
    
    # Sort by company and date
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    
    # Create features per company
    result_dfs = []
    for company in df['Company'].unique():
        company_df = df[df['Company'] == company].copy()
        
        # Basic features
        company_df['Last'] = company_df['Close']
        
        # Rolling statistics (matching training: Window 20 days)
        company_df['Mean'] = company_df['Close'].rolling(window=20, min_periods=1).mean()
        company_df['Std'] = company_df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # Lag features (matching metadata: Lags 1, 3, 5)
        company_df['Lag_1'] = company_df['Close'].shift(1)
        company_df['Lag_3'] = company_df['Close'].shift(3)
        company_df['Lag_5'] = company_df['Close'].shift(5)
        
        # Target: Price went UP next day
        company_df['Target'] = (company_df['Close'].shift(-1) > company_df['Close']).astype(int)
        
        # Drop NaN rows
        company_df = company_df.dropna()
        
        result_dfs.append(company_df)
    
    return pd.concat(result_dfs, ignore_index=True)

df_processed = create_features(df_raw)
print(f"✅ Processed data shape: {df_processed.shape}")

# ========================
# 4. CREATE SEQUENCES
# ========================
print("\n🔄 Creating Sequences...")

FEATURE_COLS = ['Last', 'Mean', 'Std', 'Lag_1', 'Lag_3', 'Lag_5']

def create_sequences(data, seq_len):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len-1, -1])  # Target is last column
    return np.array(X), np.array(y)

# Prepare data
X_all, y_all = [], []

for company in df_processed['Company'].unique():
    company_df = df_processed[df_processed['Company'] == company]
    
    # Extract features and target
    features = company_df[FEATURE_COLS].values
    targets = company_df['Target'].values
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Add target column for sequence creation
    data_with_target = np.column_stack([features_scaled, targets])
    
    # Create sequences
    if len(data_with_target) > SEQ_LENGTH:
        X_seq, y_seq = create_sequences(data_with_target, SEQ_LENGTH)
        X_all.append(X_seq[:, :, :-1])  # Remove target from features
        y_all.append(y_seq)

if not X_all:
    print("❌ No sequences created!")
    exit(1)

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)

print(f"✅ Created {len(X)} sequences with shape: {X.shape}")
print(f"   Target distribution: UP={np.sum(y==1)}, DOWN={np.sum(y==0)}")

# ========================
# 5. PREDICTIONS
# ========================
print("\n🎯 Making Predictions...")

# BiLSTM Predictions
print("  Running BiLSTM predictions...")
bilstm_pred_prob = bilstm_model.predict(X, verbose=0).flatten()

# XGBoost Predictions (needs smart features)
print("  Running XGBoost predictions...")

def get_smart_features(X_seq):
    """Extract smart features from sequences for XGBoost"""
    # Features from last timestep
    last_step = X_seq[:, -1, :]
    
    # Mean of sequence
    seq_mean = np.mean(X_seq, axis=1)
    
    # Std of sequence
    seq_std = np.std(X_seq, axis=1)
    
    # Trend (last - first)
    seq_trend = X_seq[:, -1, :] - X_seq[:, 0, :]
    
    return np.concatenate([last_step, seq_mean, seq_std, seq_trend], axis=1)

X_smart = get_smart_features(X)
dmatrix = xgb.DMatrix(X_smart)
xgb_pred_prob = xgb_model.predict(dmatrix)

# Ensemble (matching training: 40% BiLSTM, 60% XGBoost)
print("  Combining ensemble predictions...")
ensemble_prob = 0.4 * bilstm_pred_prob + 0.6 * xgb_pred_prob

# Apply best threshold from metadata
BEST_THRESHOLD = 0.3
ensemble_pred = (ensemble_prob > BEST_THRESHOLD).astype(int)

# ========================
# 6. EVALUATION
# ========================
print("\n" + "=" * 60)
print("📊 VERIFICATION RESULTS")
print("=" * 60)

# Individual model accuracies
bilstm_pred = (bilstm_pred_prob > BEST_THRESHOLD).astype(int)
xgb_pred = (xgb_pred_prob > BEST_THRESHOLD).astype(int)

print(f"\n🔹 BiLSTM Accuracy: {accuracy_score(y, bilstm_pred):.4f}")
print(f"🔹 XGBoost Accuracy: {accuracy_score(y, xgb_pred):.4f}")
print(f"🔹 Ensemble Accuracy: {accuracy_score(y, ensemble_pred):.4f}")

print("\n📋 Ensemble Classification Report:")
print(classification_report(y, ensemble_pred, target_names=['DOWN', 'UP']))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y, ensemble_pred)
print(f"           Predicted")
print(f"           DOWN   UP")
print(f"Actual DOWN  {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"       UP    {cm[1][0]:4d}  {cm[1][1]:4d}")

# Calculate directional accuracy
total = len(y)
correct = np.sum(ensemble_pred == y)
print(f"\n✅ Directional Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)
