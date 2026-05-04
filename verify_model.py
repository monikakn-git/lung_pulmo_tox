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
            print("✅ TEST 1 PASSED: Model correctly identified compound as safe.")
        else:
            print("❌ TEST 1 FAILED: Model misclassified safe compound.")
    except Exception as e:
        print(f"❌ TEST 1 ERROR: {e}")

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
            print("✅ TEST 2 PASSED: Model correctly identified compound as toxic.")
        else:
            print("❌ TEST 2 FAILED: Model misclassified toxic compound.")
    except Exception as e:
        print(f"❌ TEST 2 ERROR: {e}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    run_verification()
