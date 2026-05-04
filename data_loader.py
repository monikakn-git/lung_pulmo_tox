import pandas as pd
import os

def load_data(filepath="data.csv"):
    """
    Load dataset from a CSV file.
    Expected columns: drug_name, smiles, label
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {filepath} not found.")
    
    df = pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in ["drug_name", "smiles", "label"]):
            raise ValueError("Dataset must contain 'drug_name', 'smiles', and 'label' columns.")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return df
