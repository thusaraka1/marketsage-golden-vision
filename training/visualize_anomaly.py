import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = r"d:\nethumi final research\Historical_Data_Reextracted"

def load_data(root_dir):
    all_data = []
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
            except:
                pass
    return pd.concat(all_data, ignore_index=True)

df = load_data(DATA_DIR)
# Filter for the specific company with the issue
ctc_data = df[df['Company'] == 'CTC.N0000'].reset_index(drop=True)

print(df.columns)
# Find the anomaly indices
anomalies = ctc_data[ctc_data['High (Rs.)'] == 0]
print("Anomaly Rows:")
cols_to_print = ['High (Rs.)', 'Close (Rs.)']
if 'Date' in anomalies.columns:
    cols_to_print.insert(0, 'Date')
print(anomalies[cols_to_print])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ctc_data.index, ctc_data['High (Rs.)'], label='High Price', color='blue')
plt.scatter(anomalies.index, anomalies['High (Rs.)'], color='red', s=100, label='Anomaly (Value=0)', zorder=5)
# Also plot Close price to show it was actually high
plt.plot(ctc_data.index, ctc_data['Close (Rs.)'], label='Close Price', color='green', alpha=0.5, linestyle='--')

plt.title('Anomaly Detection: High Price Dropping to 0')
plt.xlabel('Index')
plt.ylabel('Price (Rs.)')
plt.legend()
plt.grid(True)
plt.savefig('anomaly_chart.png')
print("Chart saved to anomaly_chart.png")
