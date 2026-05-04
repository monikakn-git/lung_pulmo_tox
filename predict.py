import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from feature_engineering import get_features_for_smiles

# Global variables for caching
MODEL = None
SCALER = None
REFERENCE_DATA = None

def load_artifacts():
    global MODEL, SCALER, REFERENCE_DATA
    if MODEL is None:
        try:
            MODEL = joblib.load("model.joblib")
            SCALER = joblib.load("scaler.joblib")
            REFERENCE_DATA = joblib.load("reference_data.joblib")
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
