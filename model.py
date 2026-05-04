import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from data_loader import load_data
from feature_engineering import engineer_features

def train_model():
    print("Loading data...")
    df = load_data("data.csv")
    
    print("Engineering features...")
    X, y, valid_smiles = engineer_features(df)
    
    print(f"Dataset shape after feature engineering: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, valid_smiles, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    
    print(f"Training XGBoost model (scale_pos_weight={scale_pos_weight:.2f})...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test AUC: {auc:.3f}")
    print(f"Test Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and reference data
    print("Saving model and reference data...")
    joblib.dump(model, "model.joblib")
    
    # Save X_train and smiles_train for confidence/similarity calculations
    reference_data = {
        'X_train': X_train,
        'smiles_train': smiles_train,
        'y_train': y_train
    }
    joblib.dump(reference_data, "reference_data.joblib")
    print("Training complete.")

if __name__ == "__main__":
    train_model()
