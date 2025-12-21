import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = r"c:\Users\DELL\Downloads\anti\credit_score_predictor\backend"
DATA_PATH = os.path.join(BASE_DIR, "../data/train.csv")

def analyze_correlations():
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Select numerical columns only
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlation with Target (Loan Status)
        correlations = numeric_df.corr()['Loan Status'].sort_values(ascending=False)
        
        print("\n--- Top Positive Correlations ---")
        print(correlations.head(10))
        
        print("\n--- Top Negative Correlations ---")
        print(correlations.tail(10))
        
        # Check non-numeric promising features
        print("\n--- Categorical Columns ---")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(cat_cols)
        
        # Let's check 'Grade' and 'Employment Duration' specifically
        if 'Grade' in df.columns:
             print("\nGrade Value Counts:")
             print(df['Grade'].value_counts())
        
        if 'Employment Duration' in df.columns:
             print("\nEmployment Duration Value Counts:")
             print(df['Employment Duration'].value_counts())
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_correlations()
