import sys
import os

# Add the parent directory to sys.path so we can import our existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pubchempy as pcp

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from predict import predict_toxicity, predict_all_models
from explain import explain_prediction
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

app = FastAPI(title="Pulmonary Toxicity Predictor API")

# Configure CORS to allow our Vercel frontend (and localhost testing) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    drug_name: str

class PredictResponse(BaseModel):
    drug_name: str
    smiles: str
    risk_category: str
    probability: float
    confidence_score: float
    explanation: str
    image_base64: str
    all_models: dict
    shap_values: list
    nearest_neighbor: dict

@app.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.csv")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=500, detail="Database file not found.")
            
        df = pd.read_csv(data_path)
        
        # Case insensitive search
        drug_match = df[df['drug_name'].str.lower() == request.drug_name.lower()]
        
        if not drug_match.empty:
            smiles = drug_match.iloc[0]['smiles']
            actual_drug_name = drug_match.iloc[0]['drug_name']
        else:
            # Try PubChem as fallback
            try:
                compounds = pcp.get_compounds(request.drug_name, 'name')
                if compounds:
                    smiles = compounds[0].canonical_smiles
                    actual_drug_name = request.drug_name
                else:
                    raise HTTPException(status_code=404, detail=f"Drug '{request.drug_name}' not found in database or PubChem.")
            except Exception as e:
                if isinstance(e, HTTPException): raise e
                raise HTTPException(status_code=404, detail=f"Drug '{request.drug_name}' not found.")

        # Get the prediction and confidence score
        result = predict_toxicity(smiles)
        
        # Get the human-readable SHAP explanation
        explanation_text, _ = explain_prediction(result['features'], result['risk_category'])
        
        # Generate Image
        mol = Chem.MolFromSmiles(smiles)
        img_base64 = ""
        if mol:
            buffered = BytesIO()
            img = Draw.MolToImage(mol, size=(500, 500))
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
        # Get all model predictions & Similarity Search
        multi_data = predict_all_models(smiles)
        
        # Get SHAP top features
        _, top_features = explain_prediction(result['features'], result['risk_category'])
            
        return PredictResponse(
            drug_name=actual_drug_name,
            smiles=smiles,
            risk_category=result['risk_category'],
            probability=float(result['probability']),
            confidence_score=float(result['confidence_score']),
            explanation=explanation_text,
            image_base64=img_base64,
            all_models=multi_data['model_predictions'],
            shap_values=top_features,
            nearest_neighbor=multi_data['nearest_neighbor']
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health")
def health_check():
    health = {"status": "healthy", "checks": {}}
    try:
        import rdkit
        health["checks"]["rdkit"] = "ok"
    except Exception as e:
        health["checks"]["rdkit"] = f"failed: {str(e)}"
        health["status"] = "unhealthy"
        
    try:
        from predict import load_artifacts
        if load_artifacts():
            health["checks"]["artifacts"] = "ok"
        else:
            health["checks"]["artifacts"] = "not_found"
            health["status"] = "unhealthy"
    except Exception as e:
        health["checks"]["artifacts"] = f"error: {str(e)}"
        health["status"] = "unhealthy"
        
    return health
