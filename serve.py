#!/usr/bin/env python3
"""Serve the registered churn model over FastAPI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
MODEL_NAME = "churn-predictor"
DEFAULT_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()


class CustomerFeatures(BaseModel):
    customerID: Optional[str] = Field(
        default=None,
        description="Optional customer identifier to echo in the response",
        example="7590-VHVEG",
    )
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: float = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=1234.56)


def _load_model(stage: Optional[str] = DEFAULT_STAGE):
    """Load the Production model using mlflow.pyfunc (the magic line!)."""
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except MlflowException:
        # Fallback: if Production stage not set, use latest version
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise
        latest = max(versions, key=lambda mv: int(mv.version))
        fallback_uri = f"models:/{MODEL_NAME}/{latest.version}"
        return mlflow.pyfunc.load_model(fallback_uri)


MODEL = _load_model()
app = FastAPI(title="Churn-as-a-Service", version="1.0.0")


def _to_frame(payload: CustomerFeatures):
    import pandas as pd

    data: Dict[str, Any] = payload.model_dump()
    data.pop("customerID", None)
    return pd.DataFrame([data])


@app.get("/health")
def health_check():
    return {"status": "ok", "model_name": MODEL_NAME}


@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    """
    Predict customer churn.
    
    Returns a clean JSON response like:
    {"churn_prediction": "No", "probability": 0.15}
    """
    features = _to_frame(payload)
    
    try:
        # Get the underlying sklearn model from pyfunc wrapper
        sklearn_model = MODEL._model_impl.python_model
        
        # Use predict_proba to get probability of churn (class 1)
        proba = float(sklearn_model.predict_proba(features)[0][1])
        
        prediction = "Yes" if proba >= 0.5 else "No"
        
        return {
            "churn_prediction": prediction,
            "probability": round(proba, 2)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
