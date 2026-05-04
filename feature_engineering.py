import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def get_features_for_smiles(smiles, radius=2, nBits=2048):
    """
    Convert a SMILES string to a dictionary of features:
    Morgan fingerprints (2048 bits) + MW, LogP, Num rings.
    Returns None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_array = np.array(list(fp.ToBitString())).astype(int)
        
        # Additional Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        
        # Combine
        features = {}
        for i, bit in enumerate(fp_array):
            features[f'bit_{i}'] = bit
            
        features['MolWt'] = mw
        features['LogP'] = logp
        features['NumRings'] = num_rings
        
        return features
    except Exception as e:
        print(f"Feature engineering error: {e}")
        return None

def engineer_features(df):
    """
    Apply feature engineering to a dataframe containing a 'smiles' column.
    Returns features dataframe and corresponding labels.
    Invalid SMILES are dropped.
    """
    features_list = []
    labels = []
    valid_smiles = []
    
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row.get('label', None)
        
        feats = get_features_for_smiles(smiles)
        if feats is not None:
            features_list.append(feats)
            labels.append(label)
            valid_smiles.append(smiles)
            
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    return X, y, valid_smiles
