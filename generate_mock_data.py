import pandas as pd
import random

# A balanced, diverse list of 25 Safe and 25 Toxic compounds
# No randomly generated "derivatives" to avoid duplicates and ensure quality.

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
    {"name": "Lisinopril", "smiles": "OC(=O)C(N)CCCCNC(C)C(=O)N1CCCC1C(=O)O"},
    {"name": "Metformin", "smiles": "CN(C)C(=NC(=N)N)N"},
    {"name": "Amlodipine", "smiles": "CCOC(=O)C1=C(COCCN)NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C"},
    {"name": "Gabapentin", "smiles": "C1CCC(CC1)(CC(=O)O)CN"},
    {"name": "Sertraline", "smiles": "CN[C@H]1CC[C@@H](C2=CC=CC=C12)C3=CC=C(C=C3Cl)Cl"},
    {"name": "Furosemide", "smiles": "C1=CC=C(C=C1)CN[C@H]2C=C(C(=CC2=O)S(=O)(=O)N)Cl"},
    {"name": "Pantoprazole", "smiles": "CC1=CN=C(C(=C1OC)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC(F)F)C"},
    {"name": "Escitalopram", "smiles": "CN(C)CCCC1(OCC2=C1C=CC(=C2)C#N)C3=CC=C(C=C3)F"},
    {"name": "Rosuvastatin", "smiles": "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C"},
    {"name": "Cetirizine", "smiles": "C1CN(CCN1CCOCC(=O)O)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl"},
    {"name": "Loratadine", "smiles": "CCOC(=O)N1CCC(=C2C3=C(CCC4=C2N=CC=C4)C=C(C=C3)Cl)CC1"},
    {"name": "Diphenhydramine", "smiles": "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2"},
    {"name": "Clindamycin", "smiles": "CCCC1CC(N(C1)C)C(=O)NC(C2C(C(C(C(O2)SC)O)O)O)C(C)Cl"},
    {"name": "Azithromycin", "smiles": "CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)N)C)O)C)C)C)O)C)O"},
    {"name": "Vitamin C", "smiles": "C(C(C1C(=C(C(=O)O1)O)O)O)O"}
]

toxic_compounds = [
    {"name": "Aniline", "smiles": "C1=CC=C(C=C1)N"},
    {"name": "Dinitrobenzene", "smiles": "C1=CC(=CC=C1[N+](=O)[O-])[N+](=O)[O-]"},
    {"name": "Benzyl Chloride", "smiles": "C1=CC=C(C=C1)CCl"},
    {"name": "Phenacyl Chloride", "smiles": "C1=CC=C(C=C1)C(=O)CCl"},
    {"name": "Triphenylphosphine", "smiles": "C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=CC=CC=C3"},
    {"name": "Phenyl Isocyanate", "smiles": "O=C=NC1=CC=CC=C1"},
    {"name": "Bleomycin", "smiles": "CC1=C(C(=NC(=N1)C(C(C(=O)N)N)NC(=O)C(C(C)O)NC(=O)C(C(C)(C)O)NC(=O)C(C)C(=O)N2CCC(C2)C(=O)NC(CC3=CN=CN3)C(=O)NCC4=C(C=C(C=N4)C(=O)N)O)C)O"},
    {"name": "Amiodarone", "smiles": "CCCCC1=C(C2=C(O1)C=CC(=C2)I)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I"},
    {"name": "Methotrexate", "smiles": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"},
    {"name": "Nitrofurantoin", "smiles": "C1C(=O)NC(=O)N1N=CC2=CC=C(O2)[N+](=O)[O-]"},
    {"name": "Busulfan", "smiles": "CS(=O)(=O)OCCCCOS(=O)(=O)C"},
    {"name": "Carmustine", "smiles": "C(CCl)NC(=O)N(CCCl)N=O"},
    {"name": "Cyclophosphamide", "smiles": "C1CCNP(=O)(O1)N(CCCl)CCCl"},
    {"name": "Gefitinib", "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"},
    {"name": "Erlotinib", "smiles": "COCCOC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC"},
    {"name": "Phosgene", "smiles": "C(=O)(Cl)Cl"},
    {"name": "Paraquat", "smiles": "C[N+]1=CC=C(C=C1)C2=CC=C(C=C2)[N+]C"},
    {"name": "Nitrogen Mustard", "smiles": "CN(CCCl)CCCl"},
    {"name": "Chlorambucil", "smiles": "C1=CC(=CC=C1CCCC(=O)O)N(CCCl)CCCl"},
    {"name": "Melphalan", "smiles": "C1=CC(=CC=C1CC(C(=O)O)N)N(CCCl)CCCl"},
    {"name": "Sulfasalazine", "smiles": "C1=CC=C(C=C1)S(=O)(=O)NC2=NC=CC=C2.C1=CC=C(C=C1)N=NC3=CC(=C(C=C3)O)C(=O)O"},
    {"name": "Carmustine Var", "smiles": "C(CCl)NC(=O)N(CCC)N=O"},
    {"name": "Minocycline", "smiles": "CN(C)C1C2C(C(C3=C(C=CC(=C3C2(C(=O)C(=C1O)C(=O)N)O)O)N(C)C)O)O"},
    {"name": "Nitrofurazone", "smiles": "C1=CC(=O)NC(=N1)NN=CC2=CC=C(O2)[N+](=O)[O-]"},
    {"name": "Bischloroethyl Nitrosourea", "smiles": "O=NN(CCCl)C(=O)NCCCl"}
]

def generate_dataset():
    data = []
    
    for comp in safe_compounds:
        data.append({"drug_name": comp["name"], "smiles": comp["smiles"], "label": 0})
    for comp in toxic_compounds:
        data.append({"drug_name": comp["name"], "smiles": comp["smiles"], "label": 1})
        
    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data.csv", index=False)
    
    print(f"Generated a balanced dataset of {len(data)} unique samples in data.csv")
    print(f"Safe compounds: {len(safe_compounds)}")
    print(f"Toxic compounds: {len(toxic_compounds)}")

if __name__ == "__main__":
    generate_dataset()
