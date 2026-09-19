from typing import List
import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time MLOps inference service for financial fraud detection.",
    version="1.0.0",
)

try:
    model = joblib.load("model.joblib")
except Exception:
    model = None


class TransactionData(BaseModel):
    features: List[float]


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model artifact not found. Please train model first."
        )

    features_array = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
    }
