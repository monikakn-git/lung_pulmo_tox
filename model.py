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
import matplotlib.pyplot as plt
import os

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
    
    # Store metrics for plotting
    model_names = []
    auc_scores = []
    acc_scores = []
    
    print("\n--- Evaluating Multiple Models ---")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        model_names.append(name)
        auc_scores.append(auc)
        acc_scores.append(acc)
        
        print(f"[{name}] Test AUC: {auc:.3f} | Test Accuracy: {acc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    # Generate Performance Plot
    print("Generating model comparison plot...")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#030d12')
    ax.set_facecolor('#030d12')
    
    rects1 = ax.bar(x - width/2, auc_scores, width, label='ROC AUC', color='#00ffa2')
    rects2 = ax.bar(x + width/2, acc_scores, width, label='Accuracy', color='#3b82f6')

    ax.set_ylabel('Scores', color='#e6f7f2')
    ax.set_title('Model Performance Comparison', color='#e6f7f2', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color='#e6f7f2')
    ax.tick_params(axis='y', colors='#e6f7f2')
    
    # Grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#e6f7f2')
    
    # Spine colors
    for spine in ax.spines.values():
        spine.set_color((0, 255/255, 162/255, 0.3))

    ax.legend(facecolor='#030d12', edgecolor=(0, 255/255, 162/255, 0.3), labelcolor='#e6f7f2')

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='#e6f7f2', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    # Ensure frontend directory exists
    os.makedirs('frontend', exist_ok=True)
    plt.savefig('frontend/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='#030d12')
    plt.close('all')

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
