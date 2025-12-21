import zipfile
import os
import pandas as pd

# Define paths
base_dir = r"c:\Users\DELL\Downloads\anti\credit_score_predictor"
data_dir = os.path.join(base_dir, "data")
zip_path = os.path.join(data_dir, "archive.zip")

def extract_and_inspect():
    # 1. Extract
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            extracted_files = zip_ref.namelist()
            print(f"Extracted files: {extracted_files}")
    except Exception as e:
        print(f"Error extracting: {e}")
        return

    # 2. Inspect CSVs
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the zip.")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"\n--- Inspecting {csv_file} ---")
        try:
            df = pd.read_csv(csv_path, nrows=5) # Read only first 5 rows for inspection
            print("Columns:", list(df.columns))
            print("dtypes:\n", df.dtypes)
            print("Head:\n", df.head())
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

if __name__ == "__main__":
    extract_and_inspect()
