import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from feature_engineering import get_features_for_smiles

import os

# Global variables for caching
MODEL = None
ALL_MODELS = None
SCALER = None
REFERENCE_DATA = None

def load_artifacts():
    global MODEL, ALL_MODELS, SCALER, REFERENCE_DATA
    if MODEL is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            MODEL = joblib.load(os.path.join(base_dir, "model.joblib"))
            ALL_MODELS = joblib.load(os.path.join(base_dir, "all_models.joblib"))
            SCALER = joblib.load(os.path.join(base_dir, "scaler.joblib"))
            REFERENCE_DATA = joblib.load(os.path.join(base_dir, "reference_data.joblib"))
        except FileNotFoundError:
            return False
    return True

def compute_similarity(smiles):
    """
    Compute Tanimoto similarity to the closest molecule in the training set.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    max_sim = 0.0
    for train_smiles in REFERENCE_DATA['smiles_train']:
        train_mol = Chem.MolFromSmiles(train_smiles)
        if train_mol:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            if sim > max_sim:
                max_sim = sim
                
    return max_sim

def predict_toxicity(smiles):
    """
    Predict toxicity for a given SMILES string.
    """
    if not load_artifacts():
        raise RuntimeError("Model artifacts not found. Please train the model first.")
        
    features = get_features_for_smiles(smiles)
    if features is None:
        raise ValueError("Invalid SMILES string")
        
    # Convert to DataFrame
    X_input_raw = pd.DataFrame([features])
    
    # Scale features
    X_scaled = SCALER.transform(X_input_raw)
    X_input = pd.DataFrame(X_scaled, columns=X_input_raw.columns)
    
    # Predict
    prob = float(MODEL.predict_proba(X_input)[0, 1])
    
    # Determine risk category
    if prob >= 0.7:
        risk = "HIGH RISK"
    elif prob >= 0.4:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"
        
    similarity = compute_similarity(smiles)
    confidence_score = similarity * 100
    
    return {
        "probability": prob,
        "risk_category": risk,
        "confidence_score": confidence_score,
        "max_similarity": similarity,
        "features": X_input,
        "model_name": REFERENCE_DATA.get("model_name", "Unknown")
    }

def predict_all_models(smiles):
    """
    Get predictions from all trained models for comparison.
    """
    if not load_artifacts():
        return {}
        
    features = get_features_for_smiles(smiles)
    if features is None:
        return {}
        
    X_input_raw = pd.DataFrame([features])
    X_scaled = SCALER.transform(X_input_raw)
    X_input = pd.DataFrame(X_scaled, columns=X_input_raw.columns)
    
    results = {}
    for name, model in ALL_MODELS.items():
        prob = float(model.predict_proba(X_input)[0, 1])
        results[name] = prob
    
    # NEW: Complexity Boost - Tanimoto Similarity Search
    # Find nearest neighbor in training set
    from rdkit import DataStructs
    from feature_engineering import get_morgan_fingerprint
    
    query_fp = get_morgan_fingerprint(smiles)
    train_smiles = REFERENCE_DATA['smiles_train']
    train_labels = REFERENCE_DATA['y_train']
    
    max_sim = 0
    nearest_idx = -1
    
    for i, ts in enumerate(train_smiles):
        tfp = get_morgan_fingerprint(ts)
        if tfp:
            sim = DataStructs.TanimotoSimilarity(query_fp, tfp)
            if sim > max_sim:
                max_sim = sim
                nearest_idx = i
    
    nearest_drug = "Unknown"
    nearest_label = "Unknown"
    if nearest_idx != -1:
        # Cross reference with data.csv to get the name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(base_dir, "data.csv"))
        match = df[df['smiles'] == train_smiles[nearest_idx]]
        if not match.empty:
            nearest_drug = match.iloc[0]['drug_name']
            nearest_label = "TOXIC" if train_labels[nearest_idx] == 1 else "SAFE"

    return {
        "model_predictions": results,
        "nearest_neighbor": {
            "name": nearest_drug,
            "similarity": float(max_sim),
            "label": nearest_label
        }
    }
