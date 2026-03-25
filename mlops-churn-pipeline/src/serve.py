"""
serve.py — FastAPI model serving.

This is Step 4 of the MLOps pipeline: deploying the trained model
as a REST API. It loads the best model from MLflow and exposes:

    POST /predict     → Get churn prediction for a customer
    GET  /health      → Health check (is the server alive?)
    GET  /model-info  → What model is currently loaded?
    GET  /docs        → Auto-generated Swagger API docs

Run with: python src/serve.py
  or:     uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    API_HOST, API_PORT, MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME, MODEL_STAGE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models for Request/Response Validation ---

class CustomerData(BaseModel):
    """Input schema — validated before reaching the model."""
    tenure: int = Field(..., ge=0, le=100, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, le=500, description="Monthly bill ($)")
    total_charges: float = Field(..., ge=0, description="Total charges to date ($)")
    contract_type: str = Field(..., description="month-to-month, one-year, or two-year")
    internet_service: str = Field(..., description="fiber_optic, dsl, or none")
    payment_method: str = Field(..., description="electronic_check, mailed_check, bank_transfer, or credit_card")
    num_services: int = Field(default=3, ge=0, le=10, description="Number of subscribed services")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.50,
                "total_charges": 846.00,
                "contract_type": "month-to-month",
                "internet_service": "fiber_optic",
                "payment_method": "electronic_check",
                "num_services": 4,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    churn_prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_probability: float = Field(..., description="Probability of churning (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High")
    recommendation: str = Field(..., description="Suggested action")
    timestamp: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    features_used: list[str]
    load_method: str


# --- Global State ---
model_state = {
    "model": None,
    "version": "unknown",
    "stage": "unknown",
    "load_method": "none",
    "encoders": None,
}

# Encoding maps (matching what data_pipeline.py produces)
ENCODING_MAPS = {
    "contract_type": {"month-to-month": 0, "one-year": 1, "two-year": 2},
    "internet_service": {"dsl": 0, "fiber_optic": 1, "none": 2},
    "payment_method": {"bank_transfer": 0, "credit_card": 1,
                       "electronic_check": 2, "mailed_check": 3},
}


def load_model():
    """
    Load the production model.

    Tries MLflow Model Registry first, falls back to local file.
    """
    # Try MLflow first
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        model_state["model"] = mlflow.sklearn.load_model(model_uri)
        model_state["version"] = MODEL_STAGE
        model_state["stage"] = MODEL_STAGE
        model_state["load_method"] = "mlflow_registry"
        logger.info(f"Model loaded from MLflow: {model_uri}")
        return
    except Exception as e:
        logger.warning(f"Could not load from MLflow registry: {e}")

    # Try loading latest MLflow run
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name("churn-prediction")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.f1_score DESC"],
                max_results=1,
            )
            if len(runs) > 0:
                best_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{best_run_id}/model"
                model_state["model"] = mlflow.sklearn.load_model(model_uri)
                model_state["version"] = best_run_id[:8]
                model_state["stage"] = "latest_run"
                model_state["load_method"] = "mlflow_run"
                logger.info(f"Model loaded from MLflow run: {best_run_id}")
                return
    except Exception as e:
        logger.warning(f"Could not load from MLflow runs: {e}")

    # Fallback: load from local file
    local_path = MODELS_DIR / "model.joblib"
    if local_path.exists():
        model_state["model"] = joblib.load(local_path)
        model_state["version"] = "local"
        model_state["stage"] = "local"
        model_state["load_method"] = "local_file"
        logger.info(f"Model loaded from local file: {local_path}")
        return

    logger.error("No model found! Run train.py first.")


def prepare_features(customer: CustomerData) -> pd.DataFrame:
    """
    Transform raw customer data into model-ready features.

    This must match exactly what the training pipeline does.
    """
    # Encode categoricals
    contract_encoded = ENCODING_MAPS["contract_type"].get(
        customer.contract_type, 0)
    internet_encoded = ENCODING_MAPS["internet_service"].get(
        customer.internet_service, 0)
    payment_encoded = ENCODING_MAPS["payment_method"].get(
        customer.payment_method, 0)

    # Engineer features (same as data_pipeline.py)
    avg_charge = customer.total_charges / max(customer.tenure, 1)
    charge_ratio = customer.monthly_charges / max(avg_charge, 1)

    features = pd.DataFrame([{
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "num_services": customer.num_services,
        "avg_charge_per_month": round(avg_charge, 2),
        "charge_tenure_ratio": round(charge_ratio, 4),
        "contract_type": contract_encoded,
        "internet_service": internet_encoded,
        "payment_method": payment_encoded,
    }])

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting up — loading model...")
    load_model()
    if model_state["model"] is None:
        logger.warning("⚠ No model loaded. Predictions will fail until a model is trained.")
    else:
        logger.info(f"Model ready: v{model_state['version']} ({model_state['load_method']})")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using a trained ML model. "
                "Part of the End-to-End MLOps Pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_state["model"] is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Information about the currently loaded model."""
    return ModelInfoResponse(
        model_name=REGISTERED_MODEL_NAME,
        model_version=model_state["version"],
        model_stage=model_state["stage"],
        features_used=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        load_method=model_state["load_method"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """
    Predict whether a customer will churn.

    Accepts customer data, runs it through the ML model,
    and returns a prediction with risk level and recommendation.
    """
    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first.",
        )

    # Prepare features
    features = prepare_features(customer)

    # Predict
    model = model_state["model"]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    # Determine risk level and recommendation
    if probability < 0.3:
        risk_level = "Low"
        recommendation = "Customer is stable. Maintain current engagement."
    elif probability < 0.6:
        risk_level = "Medium"
        recommendation = ("Consider offering a loyalty discount or "
                         "contract upgrade to retain this customer.")
    else:
        risk_level = "High"
        recommendation = ("Urgent: High churn risk. Recommend immediate "
                         "outreach with retention offer or contract review.")

    # Log prediction (in production, this would go to a monitoring system)
    logger.info(
        f"Prediction: churn={prediction}, prob={probability:.3f}, "
        f"risk={risk_level}, tenure={customer.tenure}, "
        f"contract={customer.contract_type}"
    )

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk_level,
        recommendation=recommendation,
        timestamp=datetime.now().isoformat(),
        model_version=model_state["version"],
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers at once."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for customer in customers:
        features = prepare_features(customer)
        model = model_state["model"]
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        results.append({
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "tenure": customer.tenure,
            "contract_type": customer.contract_type,
        })

    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
