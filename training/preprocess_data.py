import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\nethumi final research\Historical_Data_Reextracted"
OUTPUT_FILE = r"d:\nethumi final research\cleaned_data.csv"

def load_data(root_dir):
    all_data = []
    print(f"Loading data from {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        if 'Daily.csv' in files:
            file_path = os.path.join(root, 'Daily.csv')
            try:
                company_name = os.path.basename(root).split(' - ')[0]
                df = pd.read_csv(file_path)
                df['Company'] = company_name
                if 'Date' in df.columns:
                     df['Date'] = pd.to_datetime(df['Date'])
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.concat(all_data, ignore_index=True)

def preprocess():
    # 1. Load Data
    df = load_data(DATA_DIR)
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")

    # 2. Key Step: Remove Anomalies (High == 0)
    anomalies = df[df['High (Rs.)'] == 0]
    num_anomalies = len(anomalies)
    print(f"\nFound {num_anomalies} rows with High (Rs.) == 0. Removing them...")
    df_clean = df[df['High (Rs.)'] != 0].copy()

    # 3. Handle Missing Values
    # Dropping rows with any NaNs (very small % of data)
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.dropna()
    missing_removed = initial_count - num_anomalies - len(df_clean)
    
    print(f"Removed {missing_removed} rows containing missing values.")

    # 4. Save
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")
    print(f"Final row count: {len(df_clean)}")
    print(f"Total rows removed: {initial_count - len(df_clean)}")

if __name__ == "__main__":
    preprocess()
