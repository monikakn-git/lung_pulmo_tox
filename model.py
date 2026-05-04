import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from data_loader import load_data
from feature_engineering import engineer_features

def train_model():
    print("Loading data...")
    df = load_data("data.csv")
    
    print("Engineering features...")
    X, y, valid_smiles = engineer_features(df)
    
    # Scale features for ANN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Dataset shape after feature engineering: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, valid_smiles, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    
    # Define suite of models
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
        ),
        "ArtificialNeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True
        )
    }
    
    best_model = None
    best_auc = -1
    best_name = ""
    
    print("\n--- Evaluating Multiple Models ---")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"[{name}] Test AUC: {auc:.3f} | Test Accuracy: {acc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print("-" * 32)
    print(f"🏆 Best Model: {best_name} (AUC: {best_auc:.3f})")
    
    # Save the best model and reference data
    print("\nSaving best model and artifacts...")
    joblib.dump(best_model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    
    reference_data = {
        'X_train': X_train,
        'smiles_train': smiles_train,
        'y_train': y_train,
        'model_name': best_name
    }
    joblib.dump(reference_data, "reference_data.joblib")
    print("Training complete.")

if __name__ == "__main__":
    train_model()
