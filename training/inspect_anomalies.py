import os
import pandas as pd

DATA_DIR = r"d:\nethumi final research\Historical_Data_Extracted"

def load_data(root_dir):
    all_data = []
    for root, dirs, files in os.walk(root_dir):
        if 'Daily.csv' in files:
            file_path = os.path.join(root, 'Daily.csv')
            try:
                company_name = os.path.basename(root).split(' - ')[0]
                df = pd.read_csv(file_path)
                df['Company'] = company_name
                all_data.append(df)
            except:
                pass
    return pd.concat(all_data, ignore_index=True)

df = load_data(DATA_DIR)
print(df.columns)
print("Rows with High (Rs.) == 0:")
cols_to_show = ['Company', 'High (Rs.)', 'Close (Rs.)']
if 'Date' in df.columns:
    cols_to_show.insert(0, 'Date')
print(df[df['High (Rs.)'] == 0][cols_to_show])
