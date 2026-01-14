import zipfile
import os

file_to_zip = "cleaned_data.csv"
zip_file_name = "cleaned_data.zip"
cwd = r"d:\nethumi final research"

os.chdir(cwd)

try:
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_to_zip)
    print(f"Successfully created {zip_file_name} containing {file_to_zip}")
except Exception as e:
    print(f"Failed to zip file: {e}")
