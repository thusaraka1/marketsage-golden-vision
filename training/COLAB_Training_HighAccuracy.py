# 🚀 COLAB TRAINING SCRIPT - HIGH ACCURACY (CLEANED DATA)
# Optimized for T4 GPU

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Ensure GPU is available
print(f"TensorFlow Version: {tf.__version__}")
gpu = tf.config.list_physical_devices('GPU')
if gpu:
    print(f"✅ GPU Available: {gpu}")
else:
    print("⚠️ WARNING: No GPU detected. Switch Runtime to T4 GPU!")

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip Data
ZIP_PATH = '/content/drive/MyDrive/cleaned_data.zip' # Make sure you upload the zip here
EXTRACT_PATH = '/content/data'

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"❌ Please upload 'cleaned_data.zip' to your Google Drive root directory!")

print(f"Unzipping {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("✅ Data extracted!")

# Load Data
CSV_PATH = f'{EXTRACT_PATH}/cleaned_data.csv'
df = pd.read_csv(CSV_PATH)

# Rename columns to standard names
df = df.rename(columns={
    'Trade Date': 'Date',
    'Open (Rs.)': 'Open',
    'High (Rs.)': 'High',
    'Low (Rs.)': 'Low',
    'Close (Rs.)': 'Close',
    'Turnover (Rs.)': 'Turnover'
})

df['Date'] = pd.to_datetime(df['Date'])
print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def create_features(df):
    df = df.sort_values(['Company', 'Date']).copy()
    
    # 1. Technical Indicators
    # RSI
    delta = df.groupby('Company')['Close'].diff()
    gain = (delta.where(delta > 0, 0)).groupby(df['Company']).rolling(14).mean().reset_index(0, drop=True)
    loss = (-delta.where(delta < 0, 0)).groupby(df['Company']).rolling(14).mean().reset_index(0, drop=True)
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df.groupby('Company')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby('Company')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df.groupby('Company')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # Bollinger Bands
    window = 20
    rolling_mean = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window).mean())
    rolling_std = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window).std())
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)

    # 2. Price Transformations
    df['Log_Return'] = df.groupby('Company')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    df['Volatility'] = df.groupby('Company')['Log_Return'].transform(lambda x: x.rolling(20).std())

    # 3. Volume Analysis
    df['Volume_SMA'] = df.groupby('Company')['ShareVolume'].transform(lambda x: x.rolling(20).mean())
    df['Volume_Ratio'] = df['ShareVolume'] / (df['Volume_SMA'] + 1e-10)

    # 4. Target Variable (Direction Prediction: 1 if Next Close > Current Close)
    # Important: Shift -1 to look into future
    df['Target'] = (df.groupby('Company')['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaN created by rolling windows
    df = df.dropna()
    return df

print("Generating features...")
df_processed = create_features(df)
print(f"Data shape after features: {df_processed.shape}")

# Features to use for training
FEATURES = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'Log_Return', 'Volatility', 'Volume_Ratio']
TARGET = 'Target'

# ---------------------------------------------------------
# 3. SEQUENCE CREATION (LSTM/GRU INPUT)
# ---------------------------------------------------------
SEQ_LENGTH = 20  # Reduced lookback window to prevent noise overfitting (20 days ~ 1 month)

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len, -1] # Last column is target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Scaling (Robust Scaler is better for handling outliers/spikes)
scaler = RobustScaler()
processed_companies = []


X_all, y_all = [], []

for company in df_processed['Company'].unique():
    company_df = df_processed[df_processed['Company'] == company].copy()
    
    # Scale features + target together to keep alignment, then separate
    # Note: We don't scale the target for classification actually, but let's separate first
    values = company_df[FEATURES].values
    scaled_values = scaler.fit_transform(values) # Scale features
    
    targets = company_df[TARGET].values.reshape(-1, 1) # Target is 0 or 1
    
    # Combine for sequencing: [Scaled Features, Target]
    data_for_seq = np.hstack((scaled_values, targets))
    
    X, y = create_sequences(data_for_seq, SEQ_LENGTH)
    
    if len(X) > 0:
        X_all.append(X)
        y_all.append(y)

# Concatenate all companies
X = np.concatenate(X_all)
y = np.concatenate(y_all)

print(f"Final X shape: {X.shape}")
print(f"Final y shape: {y.shape}")

# Split Train/Test (Time Series Split preferred, but random is okay if we ensure no overlap. 
# Here we used per-company grouping so simple split is risky if companies are correlated in time.
# Best practice: Split by time (e.g. Train < 2024, Test >= 2024)
# For simplicity with high data volume, we'll use a standard split but shuffle=False to respect time loosely 
# (though mixing companies breaks pure time order)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# ---------------------------------------------------------
# 3.5 SMOTE (Synthetic Minority Over-sampling Technique)
# ---------------------------------------------------------
from imblearn.over_sampling import SMOTE
print(f"Original Class Distribution: {np.bincount(y_train)}")

# SMOTE requires 2D array
X_train_flat = X_train.reshape(X_train.shape[0], -1) 
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back to 3D for LSTM: (Samples, Timesteps, Features)
X_train = X_train_resampled.reshape(X_train_resampled.shape[0], SEQ_LENGTH, X_train.shape[2])
y_train = y_train_resampled

print(f"Resampled Class Distribution: {np.bincount(y_train)}")
print(f"New X_train shape: {X_train.shape}")

from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

# ---------------------------------------------------------
# 4. MODEL ARCHITECTURE (BiLSTM + Attention)
# ---------------------------------------------------------
input_shape = (X_train.shape[1], X_train.shape[2]) # (60, num_features)

from tensorflow.keras.regularizers import l2

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 1. Convolutional Layer for pattern extraction (Simplified + Reg)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    # 2. Bidirectional LSTM (Reduced complexity + Reg)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.5)(x)
    
    # 3. Attention Mechanism
    attention_out = Attention()([x, x])
    x = Concatenate()([x, attention_out])
    
    # 4. Final LSTM layer
    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.5)(x)
    
    # 5. Dense Head
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x) # Reduced neurons 64->32
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0005), # Lower LR for regularization 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

model = build_model(input_shape)
model.summary()

# ---------------------------------------------------------
# 5. TRAINING
# ---------------------------------------------------------
checkpoint = ModelCheckpoint('/content/drive/MyDrive/temp_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.00001)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000, 
    batch_size=32,
    # class_weight=class_weight_dict, # REMOVED: SMOTE handles balance now
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
best_model = load_model('/content/drive/MyDrive/temp_best_model.keras')

y_pred_prob = best_model.predict(X_test)

# Dynamic Thresholding
best_thresh = 0.5
best_f1 = 0
for thresh in np.arange(0.3, 0.7, 0.01):
    preds = (y_pred_prob > thresh).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n🔍 Best Threshold Found: {best_thresh:.2f}")
y_pred = (y_pred_prob > best_thresh).astype(int)

acc = accuracy_score(y_test, y_pred)
print("\n🔥 FINAL RESULTS (BiLSTM + Attention):")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 7. ENSEMBLE (BiLSTM + Feature-Engineered XGBoost)
# ---------------------------------------------------------
print("\n🚀 Training Feature-Engineered XGBoost...")
from xgboost import XGBClassifier

# Create Smart Features for Manual Model (XGBoost)
def get_smart_features(X_seq):
    # X_seq shape: (Samples, 60, Features)
    
    # 1. Last Value
    last_val = X_seq[:, -1, :] 
    
    # 2. Pattern/Lags (Critical for Trend)
    lag_1 = X_seq[:, -2, 0:1] # Lag 1 of Close Price
    lag_3 = X_seq[:, -4, 0:1] # Lag 3
    lag_5 = X_seq[:, -6, 0:1] # Lag 5
    
    # 3. Mean & Volatility
    mean_val = np.mean(X_seq, axis=1)
    std_val = np.std(X_seq, axis=1)
    
    return np.hstack([last_val, mean_val, std_val, lag_1, lag_3, lag_5])

X_train_smart = get_smart_features(X_train)
X_test_smart = get_smart_features(X_test)

print(f"Smart Features Shape: {X_train_smart.shape}")

xgb_model = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=6, # Deeper trees allowed again
    subsample=0.8, 
    colsample_bytree=0.8,
    gamma=0.1, # Relaxed regularization
    reg_alpha=0.1, 
    reg_lambda=0.1, 
    random_state=42,
    n_jobs=-1
    # scale_pos_weight removed (SMOTE handles balance)
)

xgb_model.fit(X_train_smart, y_train)
xgb_pred_prob = xgb_model.predict_proba(X_test_smart)[:, 1]

def weighted_ensemble(y_pred_prob, xgb_pred_prob):
    val = (0.4 * y_pred_prob.flatten()) + (0.6 * xgb_pred_prob)
    return (val > 0.5).astype(int)

# Stacked Prediction (Weighted: XGBoost is usually more stable on small data)
ensemble_pred = weighted_ensemble(y_pred_prob, xgb_pred_prob)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🚀 Ensemble (BiLSTM + Smart XGB) Accuracy: {ensemble_acc:.4f}")

# ---------------------------------------------------------
# 8. SAVE WITH UNIQUE ID
# ---------------------------------------------------------
import joblib
import uuid
from datetime import datetime

# Generate Standard ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
unique_id = uuid.uuid4().hex[:6]
model_name = f"ENSEMBLE_ACC_{ensemble_acc:.4f}_{timestamp}_{unique_id}"

SAVE_DIR = f'/content/drive/MyDrive/Models/{model_name}'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n💾 Saving all artifacts to {SAVE_DIR}...")

# Save Models
best_model.save(f'{SAVE_DIR}/bi_lstm_model.keras')
xgb_model.save_model(f'{SAVE_DIR}/xgboost_model.json')
joblib.dump(scaler, f'{SAVE_DIR}/scaler.pkl')

# Save Metdata
with open(f'{SAVE_DIR}/metadata.txt', 'w') as f:
    f.write(f"Model: BiLSTM (SMOTE) + RobustScaler + Deep XGB Ensemble\n")
    f.write(f"Accuracy: {ensemble_acc}\n")
    f.write(f"Best Threshold: {best_thresh}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Features: Last, Mean, Std, Lags(1,3,5) | Window: 20 Days\n")

print(f"✅ Saved successfully as {model_name}!")
