import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\nethumi final research\Historical_Data_Reextracted"

def load_data(root_dir):
    all_data = []
    print(f"Searching for Daily.csv files in {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        if 'Daily.csv' in files:
            file_path = os.path.join(root, 'Daily.csv')
            try:
                # Extract company name from path (parent folder name)
                # Structure seems to be: ...\Company Name\Daily.csv
                company_name = os.path.basename(root).split(' - ')[0] 
                
                df = pd.read_csv(file_path)
                df['Company'] = company_name
                
                # Ensure Date is parsed
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No 'Daily.csv' files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def analyze_distribution(df):
    print("\n--- Distribution Analysis ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    
    stats = df[numeric_cols].describe()
    print(stats)
    
    # Save stats to file for review
    with open("analysis_results.txt", "w") as f:
        f.write("--- Distribution Statistics ---\n")
        f.write(stats.to_string())
        f.write("\n\n")

    return numeric_cols

def analyze_correlation(df, numeric_cols):
    print("\n--- Correlation Analysis ---")
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    with open("analysis_results.txt", "a") as f:
        f.write("--- Correlation Matrix ---\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")

def detect_anomalies(df, numeric_cols):
    print("\n--- Anomaly Detection (IQR Method) ---")
    anomalies_report = ""
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        pct_outliers = (len(outliers) / len(df)) * 100
        
        report = f"{col}: {len(outliers)} outliers ({pct_outliers:.2f}%)\n"
        report += f"    Range: {df[col].min()} - {df[col].max()}\n"
        report += f"    Bounds: {lower_bound:.2f} - {upper_bound:.2f}\n"
        print(report)
        anomalies_report += report
        
    with open("analysis_results.txt", "a") as f:
        f.write("--- Anomaly Detection ---\n")
        f.write(anomalies_report)
        f.write("\n")

def check_missing_values(df):
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    with open("analysis_results.txt", "a") as f:
        f.write("--- Missing Values ---\n")
        f.write(missing[missing > 0].to_string())
        f.write("\n\n")

def main():
    try:
        print("Loading data...")
        df = load_data(DATA_DIR)
        print(f"Loaded {len(df)} rows from {df['Company'].nunique()} companies.")
        
        check_missing_values(df)
        numeric_cols = analyze_distribution(df)
        if numeric_cols:
            analyze_correlation(df, numeric_cols)
            detect_anomalies(df, numeric_cols)
            
        print("\nAnalysis complete. Results saved to analysis_results.txt")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
