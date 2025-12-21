# System Architecture & Data Flow 🏗️

This document outlines the technical architecture of the **CreditScore Predictor**, explaining how data flows from the user to the AI model and back.

## 📐 System Overview

The system follows a **Client-Server Architecture**:
- **Client (Frontend)**: A browser-based interface for data collection and result visualization.
- **Server (Backend)**: A REST API that hosts the Machine Learning model and business logic.
- **Model Artifact**: A serialized Python object (`model.pkl`) that contains the learned mathematical rules.

```mermaid
graph TD
    User[User] -->|Inputs Financial Data| UI[Frontend (HTML/JS)]
    UI -->|Validates Data| UI
    UI -->|POST /predict JSON| API[FastAPI Backend]
    
    subgraph "Backend Logic"
        API -->|Extract Features| FE[Feature Engineer]
        FE -->|Scale & Transform| PP[Preprocessing]
        PP -->|Input Vector| ML[Logistic Regression Model]
        ML -->|Probability Score| Logic[Risk Categorizer]
    end
    
    Logic -->|Returns Result| UI
    UI -->|Displays Gauge & Color| User
```

## 🔌 Data Flow Steps

### 1. User Input (Frontend)
The user enters raw textual/numerical data:
- `Monthly Income`
- `Monthly Debt`
- `Loan Amount`
- `Missed Installments`
- `Credit Card Balance`

### 2. Pre-Validation (Frontend)
Before sending data, the browser checks for logical fallacies:
- *Rule*: `Debt` cannot exceed `Income`.
- *Rule*: Values cannot be negative.

### 3. API Request (HTTP/JSON)
The frontend sends a JSON payload to `http://localhost:8000/predict`:
```json
{
  "monthly_income": 5000,
  "monthly_debt_payments": 1000,
  "loan_amount": 20000,
  "missed_installments": 0,
  "credit_card_balance": 500
}
```

### 4. Feature Engineering (Backend)
The backend transforms raw inputs into the specific features the model understands. This mimics the training phase.
- **Debt-to-Income (DTI)**: `(Debt / Income) * 100`

*The model expects: `['Loan Amount', 'Debit to Income', 'Delinquency - two years', 'Revolving Balance']`*

### 5. Prediction (AI Model)
The `LogisticRegression` model applies its learned weights:
$$ P(Default) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + ... + b)}} $$
- It outputs a probability between 0.0 and 1.0 (e.g., 0.15).

### 6. Risk Categorization
The system maps the probability to a user-friendly category:
- **Low Risk**: < 40% (Green)
- **Medium Risk**: 40% - 70% (Yellow)
- **High Risk**: > 70% (Red)

### 7. Response
The API responds with the final decision:
```json
{
    "default_probability": 15.0,
    "risk_category": "Low Risk",
    "color": "green",
    "message": "Estimated default risk is 15.0%"
}
```

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Training Data**: Real-world `Loan Default Dataset` with 67,000+ records.
- **Handling Imbalance**: Uses `class_weight='balanced'` to correctly punish false negatives, ensuring it catches potential defaulters even if they are rare.
- **Pipeline**:
    1. `StandardScaler`: Normalizes inputs so large values (Loan Amount) don't dominate small values (DTI).
    2. `LogisticRegression`: The classifier.

## 📂 Folder Structure
```
credit_score_predictor/
├── backend/
│   ├── main.py             # API Server
│   ├── train_model.py      # Model Training Script
│   ├── model.pkl           # Saved AI Model (Artifact)
│   └── requirements.txt    # Python Dependencies
├── frontend/
│   ├── index.html          # User Interface
│   ├── style.css           # Styling
│   └── script.js           # Client Logic
├── data/
│   └── train.csv           # Historical Data
├── README.md               # Quick Start
└── ARCHITECTURE.md         # Technical Docs (This File)
```
