import pandas as pd
import pubchempy as pcp
import os
import time

def get_smiles(drug_name):
    try:
        print(f"Fetching SMILES for {drug_name}...")
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"Error fetching SMILES for {drug_name}: {e}")
    return None

def process_data():
    filepath = "data.csv"
    
    # Read the file line by line to handle different sections
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First section: original data (lines 1 to 51)
    original_data = []
    for line in lines[1:51]:
        original_data.append(line.strip().split(','))
    
    df_original = pd.DataFrame(original_data, columns=["drug_name", "smiles", "label"])
    df_original['label'] = pd.to_numeric(df_original['label'])
    
    # Second section: new data (lines 52 to end)
    # The header is at line 52 (index 51)
    header_new = lines[51].strip().split(',')
    new_data = []
    for line in lines[52:]:
        if line.strip():
            new_data.append(line.strip().split(','))
    
    df_new_raw = pd.DataFrame(new_data, columns=header_new)
    
    # Clean up df_new
    # drugname is the 5th column (index 4), risk_label is the 7th (index 6)
    # Mapping: safe -> 0, risky -> 1
    processed_new = []
    unique_drugs = df_new_raw['prod_ai'].unique() # prod_ai seems more descriptive (e.g. rosuvastatin calcium)
    
    # To save time and API calls, cache SMILES
    smiles_cache = {}
    
    for drug in unique_drugs:
        smiles = get_smiles(drug)
        if smiles:
            smiles_cache[drug] = smiles
        time.sleep(0.2) # Rate limiting
    
    for idx, row in df_new_raw.iterrows():
        drug = row['prod_ai']
        label = 1 if row['risk_label'].lower() == 'risky' else 0
        smiles = smiles_cache.get(drug)
        if smiles:
            processed_new.append([drug, smiles, label])
            
    df_processed_new = pd.DataFrame(processed_new, columns=["drug_name", "smiles", "label"])
    
    # Combine
    df_final = pd.concat([df_original, df_processed_new], ignore_index=True)
    
    # Drop duplicates
    df_final = df_final.drop_duplicates(subset=['drug_name', 'smiles', 'label'])
    
    print(f"Final dataset size: {len(df_final)}")
    df_final.to_csv("data_unified.csv", index=False)
    
    # Replace the old data.csv
    # os.replace("data_unified.csv", "data.csv")
    print("Data unification complete. Saved to data_unified.csv")

if __name__ == "__main__":
    process_data()
