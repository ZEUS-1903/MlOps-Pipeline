# mlops-churn-pipeline

There's a famous paper from Google that says the actual ML model is maybe 5% of a production ML system. The other 95% is everything around it — data pipelines, experiment tracking, deployment, monitoring, testing. Most tutorials skip all of that.

This project doesn't skip it.

It's an end-to-end MLOps pipeline that predicts customer churn. But the point isn't the churn model — it's the *system*. Training 3 model types, logging 96 experiments to MLflow, optimizing hyperparameters with Optuna, registering the winner, and deploying it as a FastAPI endpoint. The kind of thing you'd actually build at a job.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![MLflow](https://img.shields.io/badge/mlflow-2.9+-orange) ![FastAPI](https://img.shields.io/badge/fastapi-0.104+-green)

---

## the pipeline

```
                    ┌─────────────┐
                    │  Raw Data   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Clean &   │
                    │  Engineer   │
                    │  Features   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
   │  Logistic   │ │   Random    │ │   XGBoost   │
   │ Regression  │ │   Forest    │ │             │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │                │                │
          │         Optuna × 30 trials each │
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │   MLflow    │
                    │  Registry   │  ← every run logged
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │   /predict  │  ← production model
                    └─────────────┘
```

## results

After training all 3 model types (baseline + Optuna-optimized), here's how they compared:

| Run | Model | F1 | AUC | Status |
|-----|-------|----|-----|--------|
| xgboost_optimized | XGBoost | **0.8247** | **0.8891** | Production ★ |
| random_forest_opt | Random Forest | 0.8102 | 0.8756 | Staging |
| xgboost_baseline | XGBoost | 0.7983 | 0.8644 | Archived |
| random_forest_base | Random Forest | 0.7841 | 0.8521 | Archived |
| logreg_optimized | Logistic Reg | 0.7654 | 0.8390 | Archived |
| logreg_baseline | Logistic Reg | 0.7512 | 0.8210 | Archived |

Optuna improved XGBoost's F1 by +3.3% over baseline. The optimized model is automatically registered and promoted to Production in MLflow.

## quick start

```bash
git clone https://github.com/YOUR_USERNAME/mlops-churn-pipeline.git
cd mlops-churn-pipeline

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Run the whole thing:**

```bash
# terminal 1 — start mlflow
mlflow ui --port 5000

# terminal 2 — generate data + train
python src/data_pipeline.py
python src/train.py

# terminal 3 — deploy
python src/serve.py
```

Then open:
- `http://localhost:5000` — MLflow experiment dashboard
- `http://localhost:8000/docs` — FastAPI Swagger UI

**Test a prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5,
    "monthly_charges": 92,
    "total_charges": 460,
    "contract_type": "month-to-month",
    "internet_service": "fiber_optic",
    "payment_method": "electronic_check"
  }'
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7823,
  "risk_level": "High",
  "recommendation": "Urgent: High churn risk. Recommend immediate outreach with retention offer."
}
```

**Run tests:**
```bash
pytest tests/ -v
```

## what each file does

```
mlops-churn-pipeline/
├── src/
│   ├── config.py          # all settings in one place
│   ├── data_pipeline.py   # data generation, cleaning, feature engineering
│   ├── train.py           # trains 3 models, optuna tuning, mlflow logging
│   └── serve.py           # fastapi with /predict, /health, /model-info
├── tests/
│   └── test_pipeline.py   # 14 tests covering data, training, and serving
├── data/                  # generated csvs + sqlite (gitignored)
├── models/                # saved artifacts (gitignored)
└── mlruns/                # mlflow experiment data (gitignored)
```

**config.py** — I hate magic numbers scattered across files. Everything lives here: feature names, MLflow URIs, Optuna trial counts, API ports.

**data_pipeline.py** — Generates 5,000 synthetic telecom customers with realistic churn patterns. Feature engineering adds `avg_charge_per_month` and `charge_tenure_ratio`. Validates for nulls/infinities before splitting.

**train.py** — The main event. For each of the 3 model types:
1. Trains a baseline
2. Runs 30 Optuna trials to find optimal hyperparameters
3. Trains a final model with the best params
4. Logs everything to MLflow — params, metrics, confusion matrix, feature importance plot, classification report
5. Registers the best overall model and promotes it to Production

**serve.py** — Loads the Production model from MLflow, validates input with Pydantic, returns predictions with risk level and recommended action. Includes health check and batch prediction endpoints.

**test_pipeline.py** — 14 tests. Data quality checks (no nulls, binary target, realistic churn rate). Model checks (trains without error, binary predictions, probabilities between 0 and 1). Encoding checks (maps cover all values).

## what I'd add next

- Docker compose to run everything with one command
- Data drift detection with Evidently AI
- GitHub Actions CI that retrains when data changes
- A/B testing between model versions
- Prometheus + Grafana monitoring

I didn't add these yet because I wanted the core pipeline to be clean and understandable first. Complexity for the sake of complexity helps nobody.

## tech stack

| What | Why |
|------|-----|
| scikit-learn + XGBoost | solid, interpretable models for tabular data |
| MLflow | experiment tracking + model registry. the industry standard |
| Optuna | hyperparameter optimization. cleaner API than GridSearch |
| FastAPI | modern, fast, auto-generates API docs |
| Pydantic | input validation at the API layer |
| pytest | because untested ML pipelines are scary |
| pandas + numpy | the usual suspects |

## things I learned building this

- MLflow's autologging is nice but I prefer explicit logging — you control exactly what gets tracked
- Optuna's pruning feature saves a lot of time on bad trials
- The biggest F1 improvement came from feature engineering, not hyperparameter tuning
- FastAPI's auto-generated docs at `/docs` are incredibly useful for testing
- Writing tests for ML pipelines is different — you're testing distributions and ranges, not exact values
