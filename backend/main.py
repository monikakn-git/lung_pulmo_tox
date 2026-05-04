import sys
import os

# Add the parent directory to sys.path so we can import our existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_toxicity
from explain import explain_prediction

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
    smiles: str

class PredictResponse(BaseModel):
    smiles: str
    risk_category: str
    probability: float
    confidence_score: float
    explanation: str

@app.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        # Get the prediction and confidence score
        result = predict_toxicity(request.smiles)
        
        # Get the human-readable SHAP explanation
        explanation_text, _ = explain_prediction(result['features'], result['risk_category'])
        
        return PredictResponse(
            smiles=request.smiles,
            risk_category=result['risk_category'],
            probability=float(result['probability']),
            confidence_score=float(result['confidence_score']),
            explanation=explanation_text
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error while processing SMILES.")

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
