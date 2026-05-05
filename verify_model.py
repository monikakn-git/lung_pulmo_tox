import json
from predict import predict_toxicity

def run_verification():
    print("="*50)
    print("MODEL VERIFICATION TESTS")
    print("="*50)

    # Test 1: Known Safe Compound (Aspirin)
    safe_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print("\n[TEST 1] Testing Known Safe Compound (Aspirin)")
    print(f"SMILES: {safe_smiles}")
    try:
        res = predict_toxicity(safe_smiles)
        print(f"Risk Category: {res['risk_category']}")
        print(f"Probability of Toxicity: {res['probability']:.2%}")
        print(f"Confidence Score: {res['confidence_score']:.1f}%")
        if res['risk_category'] == "LOW RISK":
            print("[PASS] TEST 1: Model correctly identified compound as safe.")
        else:
            print("[FAIL] TEST 1: Model misclassified safe compound.")
    except Exception as e:
        print(f"[ERROR] TEST 1: {e}")

    # Test 2: Known Toxic Compound (Aniline)
    toxic_smiles = "C1=CC=C(C=C1)N"
    print("\n[TEST 2] Testing Known Toxic Compound (Aniline)")
    print(f"SMILES: {toxic_smiles}")
    try:
        res = predict_toxicity(toxic_smiles)
        print(f"Risk Category: {res['risk_category']}")
        print(f"Probability of Toxicity: {res['probability']:.2%}")
        print(f"Confidence Score: {res['confidence_score']:.1f}%")
        if res['risk_category'] in ["HIGH RISK", "MEDIUM RISK"]:
            print("[PASS] TEST 2: Model correctly identified compound as toxic.")
        else:
            print("[FAIL] TEST 2: Model misclassified toxic compound.")
    except Exception as e:
        print(f"[ERROR] TEST 2: {e}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    run_verification()
