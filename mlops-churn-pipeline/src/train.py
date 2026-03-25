"""
train.py — Model training pipeline with MLflow experiment tracking.

This is the heart of the MLOps pipeline. It:
1. Loads processed data
2. Trains multiple model types (Logistic Regression, Random Forest, XGBoost)
3. Runs hyperparameter optimization with Optuna
4. Logs EVERYTHING to MLflow (params, metrics, artifacts, models)
5. Registers the best model in MLflow Model Registry
6. Promotes the best model to "Production" stage

After running this, open http://localhost:5000 to see your experiments.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME, TARGET, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, OPTUNA_N_TRIALS, RANDOM_STATE,
)
from data_pipeline import run_pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load train and test data, creating it if needed."""
    if not TRAIN_DATA_PATH.exists():
        print("[train] Data not found, running data pipeline first...")
        run_pipeline()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


def create_model(model_type: str, params: dict = None):
    """
    Create a model pipeline with preprocessing + classifier.

    Using sklearn Pipeline ensures preprocessing is part of the model artifact —
    critical for reproducibility in production.
    """
    params = params or {}

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            **params,
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            **params,
        )
    elif model_type == "xgboost":
        classifier = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            **params,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap in a Pipeline with scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    return pipeline


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a model and return all metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return {k: round(v, 4) for k, v in metrics.items()}


def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """Create and save a confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names, save_path: str):
    """Plot feature importance for tree-based models."""
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return  # Model doesn't support feature importance

    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(importances)),
        importances[indices],
        color="#c8903e",
        edgecolor="none",
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def get_optuna_params(trial, model_type: str) -> dict:
    """Define the hyperparameter search space for each model type."""
    if model_type == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",
        }
    elif model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }


def train_with_tracking(model_type: str, X_train, y_train, X_test, y_test,
                         params: dict = None, run_name: str = None) -> tuple:
    """
    Train a single model and log everything to MLflow.

    This is the core MLOps pattern:
    1. Start an MLflow run
    2. Log parameters (what config was used)
    3. Train the model
    4. Log metrics (how well it did)
    5. Log artifacts (plots, reports)
    6. Log the model itself (for later deployment)
    """
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    with mlflow.start_run(run_name=run_name or f"{model_type}_{datetime.now():%H%M%S}"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        if params:
            mlflow.log_params(params)

        # Train
        model = create_model(model_type, params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)
        print(f"  [{model_type}] F1: {metrics['f1_score']:.4f} | "
              f"AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")

        # Log artifacts (plots)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        cm_path = str(MODELS_DIR / "confusion_matrix.png")
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = str(MODELS_DIR / "feature_importance.png")
        plot_feature_importance(model, features, fi_path)
        if Path(fi_path).exists():
            mlflow.log_artifact(fi_path)

        # Log classification report as text
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        report_path = str(MODELS_DIR / "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log the model
        if model_type == "xgboost":
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    return model, metrics, run_id


def run_optuna_optimization(model_type: str, X_train, y_train,
                            X_test, y_test) -> dict:
    """
    Run Optuna hyperparameter optimization with MLflow tracking.

    Each Optuna trial is logged as a separate MLflow run,
    so you can compare them all in the MLflow UI.
    """
    print(f"\n  Running Optuna optimization for {model_type} "
          f"({OPTUNA_N_TRIALS} trials)...")

    def objective(trial):
        params = get_optuna_params(trial, model_type)
        model = create_model(model_type, params)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        return metrics["f1_score"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    print(f"  Best F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def run_training_pipeline():
    """
    Execute the full training pipeline.

    1. Load data
    2. For each model type:
       a. Run baseline training
       b. Run hyperparameter optimization
       c. Train final model with best params
    3. Compare all models
    4. Register the best one in MLflow Model Registry
    """
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\n[MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[MLflow] Experiment: {MLFLOW_EXPERIMENT_NAME}")

    # Load data
    print("\n[1/4] Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  → Train: {X_train.shape}, Test: {X_test.shape}")

    # Train each model type
    results = {}
    models_to_try = ["logistic_regression", "random_forest", "xgboost"]

    print("\n[2/4] Training baseline models...")
    for model_type in models_to_try:
        model, metrics, run_id = train_with_tracking(
            model_type, X_train, y_train, X_test, y_test,
            run_name=f"{model_type}_baseline",
        )
        results[f"{model_type}_baseline"] = {
            "model": model, "metrics": metrics, "run_id": run_id,
        }

    # Hyperparameter optimization
    print("\n[3/4] Hyperparameter optimization with Optuna...")
    for model_type in models_to_try:
        best_params = run_optuna_optimization(
            model_type, X_train, y_train, X_test, y_test,
        )

        # Train final model with best params
        model, metrics, run_id = train_with_tracking(
            model_type, X_train, y_train, X_test, y_test,
            params=best_params,
            run_name=f"{model_type}_optimized",
        )
        results[f"{model_type}_optimized"] = {
            "model": model, "metrics": metrics, "run_id": run_id,
            "params": best_params,
        }

    # Find the best model
    print("\n[4/4] Selecting and registering best model...")
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_result = results[best_name]

    print(f"\n  🏆 Best model: {best_name}")
    print(f"     F1 Score:  {best_result['metrics']['f1_score']:.4f}")
    print(f"     ROC AUC:   {best_result['metrics']['roc_auc']:.4f}")
    print(f"     Accuracy:  {best_result['metrics']['accuracy']:.4f}")
    print(f"     Run ID:    {best_result['run_id']}")

    # Register in MLflow Model Registry
    try:
        model_uri = f"runs:/{best_result['run_id']}/model"
        registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        print(f"\n  ✓ Model registered: {REGISTERED_MODEL_NAME} v{registered.version}")

        # Promote to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=registered.version,
            stage="Production",
        )
        print(f"  ✓ Model promoted to Production stage")
    except Exception as e:
        print(f"\n  ⚠ Could not register model (MLflow server may not be running): {e}")
        print(f"  → Model saved locally. Start MLflow with: mlflow ui --port 5000")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'F1':>8} {'AUC':>8} {'Acc':>8}")
    print("-" * 60)
    for name, res in sorted(results.items(),
                            key=lambda x: x[1]["metrics"]["f1_score"],
                            reverse=True):
        m = res["metrics"]
        marker = " ← BEST" if name == best_name else ""
        print(f"{name:<35} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f} "
              f"{m['accuracy']:>8.4f}{marker}")

    print(f"\n{'='*60}")
    print("TRAINING PIPELINE COMPLETE ✓")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. View experiments: mlflow ui --port 5000")
    print(f"  2. Deploy model:     python src/serve.py")
    print(f"  3. Test API:         http://localhost:8000/docs")

    return results


if __name__ == "__main__":
    run_training_pipeline()
