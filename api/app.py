from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import joblib

app = FastAPI(title="SESTRAV 2.0 Immunogenicity Scoring API", version="2.0.0")

class PeptideRequest(BaseModel):
    sequence: str
    alleles: List[str]

class PeptideResponse(BaseModel):
    sequence: str
    sestrav_score: float
    percentile_rank: Optional[float] = None
    model_version: str

# Lazy load model (mock path for scaffolding)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/rf_30feature_integrated.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/score", response_model=PeptideResponse)
def score_peptide(request: PeptideRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")
    
    # Validation logic (length check)
    if len(request.sequence) < 8 or len(request.sequence) > 11:
        raise HTTPException(status_code=400, detail="Sequence length must be 8-11 amino acids.")
    
    # -------------------------------------------------------------------------
    # PLACEHOLDER: 
    # 1. Compute 20 physicochemical features via src.features.compute_features
    # 2. Invoke MHCflurry backend for the specified alleles to get 10 context features
    # 3. Concatenate to 30-feature vector and run model.predict_proba()
    # -------------------------------------------------------------------------
    
    dummy_score = 0.85 # Placeholder
    
    return PeptideResponse(
        sequence=request.sequence,
        sestrav_score=dummy_score,
        model_version="rf_30feature_integrated"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
