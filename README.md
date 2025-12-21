# CreditScore Predictor 🚀

A premium, full-stack AI application that predicts loan default risk using Logistic Regression. This system simulates a real-world fintech product, offering instant, explainable risk assessments with a modern user interface.

## 🌟 Key Features

- **AI-Powered Risk Analysis**: Uses a Logistic Regression model trained on real historical loan data.
- **Instant Feedback**: Real-time probability calculation and risk categorization (Low, Medium, High).
- **Premium UI**: Glassmorphism design, smooth animations, and responsive layout.
- **Bank-Grade Validation**: Logical consistency checks for income, debt, and loan amounts.

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Scikit-Learn
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript
- **Model**: Logistic Regression (Scikit-Learn) with StandardScaler

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Setup Backend
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model
The system uses the `Loan Default Dataset` to learn risk patterns.
```bash
python train_model.py
```
*This will generate `model.pkl`, the trained brain of the system.*

### 3. Run the API Server
```bash
python main.py
```
The server will start at `http://127.0.0.1:8000`.

### 4. Launch the Frontend
Simply open `frontend/index.html` in your web browser. 
No extra server needed for the frontend!

## 📊 How It Works
1. **User Input**: You provide financial details (Income, Debt, Loan Amount, History).
2. **Validation**: The frontend ensures data is logical (e.g., Debt < Income).
3. **API Computation**: The backend calculates derived features like *Debt-to-Income Ratio*.
4. **AI Prediction**: The trained model computes the probability of default.
5. **Result**: You get a color-coded risk assessment instantly.

---
*Built with ❤️ for advanced fintech prediction.*
