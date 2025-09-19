from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from app.predictor import predict_stock

app = FastAPI(title="Fingrow Hybrid")

# IMPORTANT: add CORS middleware BEFORE your routes so preflight (OPTIONS) is handled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow all origins for now (change to your domain for production)
    allow_credentials=True,
    allow_methods=["*"],            # allow all methods including OPTIONS
    allow_headers=["*"],            # allow headers like Content-Type, Authorization
)

class PredictRequest(BaseModel):
    ticker: str
    fmp_api_key: Optional[str] = None
    horizon: str

@app.post("/predict")
async def predict(req: PredictRequest):
    key = req.fmp_api_key or os.environ.get("FMP_API_KEY")
    try:
        out = predict_stock(req.ticker.strip().upper(), req.horizon, key)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
