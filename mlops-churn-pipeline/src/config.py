"""
config.py — Centralized configuration for the MLOps pipeline.

Single source of truth for paths, parameters, and settings.
Change things here, not scattered across files.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_PATH = DATA_DIR / "churn_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "churn_processed.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# --- MLflow ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-classifier"

# --- Data Pipeline ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target column
TARGET = "churn"

# Features to use
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_services",
    "avg_charge_per_month",
    "charge_tenure_ratio",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

# --- Training ---
# Models to try during training
MODELS_TO_TRY = ["logistic_regression", "random_forest", "xgboost"]

# Optuna hyperparameter optimization
OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT = 300  # seconds

# --- Serving ---
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_STAGE = "Production"  # Which MLflow stage to serve
