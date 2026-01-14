import zipfile
import os

zip_path = 'Historical Data.zip'
extract_to = 'Historical_Data_Reextracted'

os.makedirs(extract_to, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Successfully extracted {zip_path} to {extract_to}")
except Exception as e:
    print(f"Failed to extract: {e}")
