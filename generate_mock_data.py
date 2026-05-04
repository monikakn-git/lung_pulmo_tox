import pandas as pd
import random

# A mix of valid SMILES with real chemical/drug names
safe_compounds = [
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
    {"name": "Acetaminophen", "smiles": "CC1=C(C=C(C=C1)NC(=O)C)O"},
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"name": "Phenylglycine", "smiles": "C1=CC=C(C=C1)C(C(=O)O)N"},
    {"name": "Benzoic Acid", "smiles": "C1=CC=C(C=C1)C(=O)O"},
    {"name": "Albuterol", "smiles": "CC(C)(C)NCC(O)C1=CC(=C(C=C1)O)CO"},
    {"name": "Cyclohexanecarboxylic Acid", "smiles": "C1CCC(CC1)C(=O)O"},
    {"name": "Amoxicillin", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"},
    {"name": "Omeprazole", "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC"},
]

toxic_compounds = [
    {"name": "Aniline", "smiles": "C1=CC=C(C=C1)N"},
    {"name": "Dinitrobenzene", "smiles": "C1=CC(=CC=C1[N+](=O)[O-])[N+](=O)[O-]"},
    {"name": "Benzyl Chloride", "smiles": "C1=CC=C(C=C1)CCl"},
    {"name": "Phenacyl Chloride", "smiles": "C1=CC=C(C=C1)C(=O)CCl"},
    {"name": "Triphenylphosphine", "smiles": "C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=CC=CC=C3"},
    {"name": "Phenyl Isocyanate", "smiles": "O=C=NC1=CC=CC=C1"},
    {"name": "Bleomycin", "smiles": "CC1=C(C(=NC(=N1)C(C(C(=O)N)N)NC(=O)C(C(C)O)NC(=O)C(C(C)(C)O)NC(=O)C(C)C(=O)N2CCC(C2)C(=O)NC(CC3=CN=CN3)C(=O)NCC4=C(C=C(C=N4)C(=O)N)O)C)O"}, # Bleomycin causes pulmonary toxicity
    {"name": "Amiodarone", "smiles": "CCCCC1=C(C2=C(O1)C=CC(=C2)I)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I"}, # Amiodarone causes pulmonary toxicity
    {"name": "Methotrexate", "smiles": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"}, # Can cause lung toxicity
]

def generate_dataset(num_samples=150):
    data = []
    
    # First, make sure every template is included at least once
    for comp in safe_compounds:
        data.append({"drug_name": comp["name"], "smiles": comp["smiles"], "label": 0})
    for comp in toxic_compounds:
        data.append({"drug_name": comp["name"], "smiles": comp["smiles"], "label": 1})
        
    # Then fill the rest with slight variations
    remaining = num_samples - len(data)
    for i in range(remaining):
        is_toxic = random.choice([0, 1])
        if is_toxic == 1:
            base_comp = random.choice(toxic_compounds)
            chain = "C" * random.randint(1, 3)
            smiles = base_comp["smiles"] + chain
            name = f'{base_comp["name"]} Derivative {i}'
        else:
            base_comp = random.choice(safe_compounds)
            chain = "C" * random.randint(1, 3)
            smiles = base_comp["smiles"] + chain
            name = f'{base_comp["name"]} Derivative {i}'
        
        data.append({"drug_name": name, "smiles": smiles, "label": is_toxic})
    
    # Shuffle
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    print(f"Generated {len(data)} samples in data.csv")

if __name__ == "__main__":
    generate_dataset()
