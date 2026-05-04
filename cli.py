import argparse
import sys
from predict import predict_toxicity
from explain import explain_prediction
import json

def main():
    parser = argparse.ArgumentParser(description="Pulmonary Toxicity Predictor CLI")
    parser.add_argument("smiles", type=str, help="SMILES string to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    try:
        result = predict_toxicity(args.smiles)
        explanation_text, _ = explain_prediction(result['features'], result['risk_category'])
        
        if args.json:
            output = {
                "smiles": args.smiles,
                "risk_category": result['risk_category'],
                "probability": float(result['probability']),
                "confidence_score": float(result['confidence_score']),
                "explanation": explanation_text
            }
            print(json.dumps(output, indent=2))
        else:
            print("\n" + "="*50)
            print("🫁  PULMONARY TOXICITY PREDICTION")
            print("="*50)
            print(f"Input SMILES     : {args.smiles}")
            print(f"Risk Category    : {result['risk_category']}")
            print(f"Probability      : {result['probability']:.1%}")
            print(f"Confidence Score : {result['confidence_score']:.1f}%")
            print("-" * 50)
            print("Explanation:")
            print(explanation_text)
            print("="*50 + "\n")
            
    except ValueError as e:
        print(f"Error: {e}. Please provide a valid SMILES string.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
