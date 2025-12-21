import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_samples=5000):
    # User Inputs (Features)
    income = np.random.normal(50000, 15000, n_samples)
    income = np.maximum(income, 10000)  # Min income 10k

    savings = np.random.normal(10000, 5000, n_samples)
    savings = np.maximum(savings, 0)
    
    # 0 = No prev loan, 1 = Good history, 2 = Bad history
    prev_loan_status = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
    
    missed_installments = np.zeros(n_samples)
    # If bad history, add missed installments
    mask_bad = prev_loan_status == 2
    missed_installments[mask_bad] = np.random.randint(1, 10, size=np.sum(mask_bad))
    
    loan_amount = np.random.normal(income * 0.5, 10000, n_samples)
    loan_amount = np.maximum(loan_amount, 1000)

    # DataFrame
    df = pd.DataFrame({
        'income': income,
        'savings': savings,
        'prev_loan_status': prev_loan_status,
        'missed_installments': missed_installments,
        'loan_amount': loan_amount
    })

    # --- Feature Engineering (Backend Logic) ---
    # These are the same transformations we will do in the live backend
    
    # Feature 1: Debt Burden Ratio (Loan / Income)
    df['debt_burden'] = df['loan_amount'] / df['income']
    
    # Feature 2: Savings Ratio (Savings / Loan)
    df['savings_ratio'] = df['savings'] / df['loan_amount']
    
    # Feature 3: Credit Discipline (Penalty for missed installments)
    df['credit_discipline'] = 1 - (df['missed_installments'] * 0.1)
    df['credit_discipline'] = df['credit_discipline'].clip(0, 1)

    # --- Target Generation (Simulation) ---
    # Calculate a "True Risk Score" (hidden) to determine default
    # Higher Score = Higher Risk of Default
    
    risk_score = (
        (df['debt_burden'] * 3.0) +       # High debt is bad
        (df['prev_loan_status'] * 0.5) -  # Bad history (2) increases risk
        (df['savings_ratio'] * 1.5) -     # High savings reduce risk
        (df['credit_discipline'] * 2.0)   # Good discipline reduces risk
    )
    
    # Add some random noise
    risk_score += np.random.normal(0, 0.5, n_samples)
    
    # Convert score to probability (Sigmoid)
    probability = 1 / (1 + np.exp(-risk_score))
    
    # Define Default (1) or No Default (0)
    # If prob > 0.5, high chance of default
    df['default'] = (probability > 0.5).astype(int)
    
    print(f"Dataset Generated: {n_samples} samples")
    print(f"Default Rate: {df['default'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("loan_data.csv", index=False)
    print("Saved to loan_data.csv")
