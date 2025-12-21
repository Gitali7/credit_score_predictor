from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os

# Initialize App
app = FastAPI(title="CreditScore Predictor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
    print("---------------------------------------------------------")
    print("   🚀 CREDIT SCORE PREDICTOR SERVER STARTED   ")
    print("   Frontend is mounted at http://127.0.0.1:8000      ")
    print("---------------------------------------------------------")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Input Schema
class LoanApplication(BaseModel):
    monthly_income: float = Field(..., gt=0, description="Monthly Income")
    monthly_debt_payments: float = Field(..., ge=0, description="Monthly Debt Payments")
    loan_amount: float = Field(..., gt=1000, description="Requested Loan Amount")
    missed_installments: int = Field(..., ge=0, description="Missed installments last 2 yrs")
    credit_card_balance: float = Field(..., ge=0, description="Current Balance")
    
    total_open_accounts: int = Field(..., ge=0, description="Total number of credit lines")
    home_ownership: str = Field(..., pattern="^(RENT|MORTGAGE|OWN)$", description="Home Ownership Status")

@app.post("/predict")
def predict_risk(data: LoanApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    dti = 0.0
    if data.monthly_income > 0:
        dti = (data.monthly_debt_payments / data.monthly_income) * 100

    features = pd.DataFrame([{
        'Loan Amount': data.loan_amount,
        'Debit to Income': dti,
        'Delinquency - two years': data.missed_installments,
        'Revolving Balance': data.credit_card_balance,
        'Open Account': data.total_open_accounts,
        'Employment Duration': data.home_ownership
    }])
    
    try:
        probs = model.predict_proba(features)[0]
        default_prob = probs[1]
        
        # --- STRICT VALIDATION RULES (RED FLAGS) ---
        risk_reasons = []

        # 1. Base Penalty for Renting
        if data.home_ownership == 'RENT':
            default_prob += 0.05 # +5% Risk for Renting
        
        # 2. Missed Installments (Severe Penalty)
        if data.missed_installments > 0:
            if data.missed_installments == 1:
                default_prob = max(default_prob, 0.45) # At least Medium Risk
                default_prob += 0.15 
                risk_reasons.append("Recent missed installment detected.")
            else:
                default_prob = max(default_prob, 0.85) # Force High Risk
                risk_reasons.append(f"Critical: {data.missed_installments} missed installments.")

        # 3. Renting + Missed Installments = AUTOMATIC HIGH RISK
        if data.home_ownership == 'RENT' and data.missed_installments > 0:
            default_prob = max(default_prob, 0.80)
            risk_reasons.append("High Risk combination: Renting with delinquency.")

        # 4. Low Savings / Disposable Income (The 'Save 1 Rupee' Case)
        disposable_income = data.monthly_income - data.monthly_debt_payments
        if disposable_income < 1000: # Very tight budget
            default_prob = max(default_prob, 0.90) # Almost certain default
            risk_reasons.append("Critical: Extremely low disposable income.")
        elif disposable_income < 3000:
            default_prob += 0.10
            risk_reasons.append("Warning: Low disposable income.")

        # Clamp Probability
        default_prob = min(max(default_prob, 0.0), 0.99)

        risk_category = "Low Risk"
        color = "green"
        
        if default_prob > 0.60: # Stricter threshold (was 0.7)
            risk_category = "High Risk"
            color = "red"
        elif default_prob > 0.30: # Stricter threshold (was 0.4)
            risk_category = "Medium Risk"
            color = "#ffcc00"
            
        msg = f"Estimated default risk is {default_prob:.1%}"
        if risk_reasons:
            msg += f". Alerts: {'; '.join(risk_reasons)}"

        return {
            "default_probability": round(default_prob * 100, 2),
            "risk_category": risk_category,
            "color": color,
            "message": msg
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# Mount Frontend - MUST BE LAST to avoid overriding API routes
# We verify the path relative to this file
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Revert to port 8000 (Standard)
    uvicorn.run(app, host="127.0.0.1", port=8005)
