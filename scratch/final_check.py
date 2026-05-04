import sys
import os
sys.path.append("/Users/apple/Documents/hackathons/lung_pulmo_tox")

from predict import predict_all_models, predict_toxicity
from explain import explain_prediction

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
print(f"Testing SMILES: {smiles}")

# 1. Test Toxicity Prediction
res = predict_toxicity(smiles)
print(f"\n[Best Model Prediction]")
print(f"Risk: {res['risk_category']}, Prob: {res['probability']:.3f}, Confidence: {res['confidence_score']:.1f}%")

# 2. Test Multi-Model & Similarity
multi = predict_all_models(smiles)
print(f"\n[Multi-Model Benchmark]")
for name, prob in multi['model_predictions'].items():
    print(f"{name}: {prob:.3f}")

print(f"\n[Nearest Neighbor]")
print(f"Name: {multi['nearest_neighbor']['name']}")
print(f"Similarity: {multi['nearest_neighbor']['similarity']:.3f}")
print(f"Label: {multi['nearest_neighbor']['label']}")

# 3. Test SHAP
explanation, top_features = explain_prediction(res['features'], res['risk_category'])
print(f"\n[SHAP Explanation]")
print(explanation)
print(f"\n[Top 3 Features]")
for feat in top_features[:3]:
    print(f"{feat['feature']}: {feat['value']:.3f}")
