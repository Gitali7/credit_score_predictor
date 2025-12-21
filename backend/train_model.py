import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Select Features
    # Added 'Open Account' (Total Accounts) and 'Employment Duration' (Home Status)
    numeric_features = ['Loan Amount', 'Debit to Income', 'Delinquency - two years', 'Revolving Balance', 'Open Account']
    categorical_features = ['Employment Duration'] # Contains MORTGAGE, RENT, OWN
    
    target = 'Loan Status'
    
    # Keep useful columns
    df = df[numeric_features + categorical_features + [target]].dropna()
    
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    print(f"Training on {len(df)} records with features: {X.columns.tolist()}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define Preprocessing Pipeline
    # We use ColumnTransformer to apply different logic to Numeric vs Categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 4. Define Full Pipeline with XGBoost
    # scale_pos_weight is helpful for imbalanced datasets (approx ratio neg/pos ~ 10)
    xgb = XGBClassifier(
        n_estimators=200,          # More trees
        learning_rate=0.05,        # Slower learning for robustness
        max_depth=6,               # Deeper trees to capture complexity
        scale_pos_weight=10,       # Handle imbalance (90:10 ratio)
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])
    
    # 5. Train
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # 7. Save
    print("Saving pipeline...")
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
