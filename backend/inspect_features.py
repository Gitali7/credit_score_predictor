import pandas as pd
import os

data_dir = r"c:\Users\DELL\Downloads\anti\credit_score_predictor\data"
csv_path = os.path.join(data_dir, "train.csv")

def inspect_features():
    cols_to_check = [
        'Loan Amount', 
        'Debit to Income', 
        'Delinquency - two years', 
        'Revolving Balance', 
        'Loan Status'
    ]
    
    try:
        df = pd.read_csv(csv_path, usecols=cols_to_check)
        print("--- Feature Stats ---")
        print(df.describe())
        print("\n--- Missing Values ---")
        print(df.isnull().sum())
        print("\n--- Target Distribution ---")
        print(df['Loan Status'].value_counts(normalize=True))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_features()
