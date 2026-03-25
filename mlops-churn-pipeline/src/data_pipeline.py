"""
data_pipeline.py — Data loading, cleaning, feature engineering, and splitting.

This is Step 1 of the MLOps pipeline. It takes raw data, cleans it,
engineers useful features, and produces train/test splits ready for modeling.

In a real project, this would connect to a database or data warehouse.
Here we generate realistic synthetic data for demonstration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    TEST_SIZE, RANDOM_STATE, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate realistic synthetic customer churn data.

    In production, you'd load from a database. This simulates
    a telecom customer dataset similar to the classic Telco Churn dataset.
    """
    np.random.seed(RANDOM_STATE)

    # Customer tenure (months)
    tenure = np.random.exponential(scale=30, size=n_samples).astype(int)
    tenure = np.clip(tenure, 1, 72)

    # Contract type affects churn heavily
    contracts = np.random.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_samples,
        p=[0.5, 0.3, 0.2],
    )

    # Internet service
    internet = np.random.choice(
        ["fiber_optic", "dsl", "none"],
        size=n_samples,
        p=[0.45, 0.35, 0.2],
    )

    # Payment method
    payment = np.random.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        size=n_samples,
        p=[0.35, 0.2, 0.25, 0.2],
    )

    # Monthly charges (depends on internet type)
    base_charges = np.where(
        internet == "fiber_optic", np.random.normal(85, 15, n_samples),
        np.where(internet == "dsl", np.random.normal(55, 12, n_samples),
                 np.random.normal(25, 8, n_samples))
    )
    monthly_charges = np.clip(base_charges, 18, 120).round(2)

    # Total charges = monthly * tenure (with some noise)
    total_charges = (monthly_charges * tenure * np.random.uniform(0.9, 1.1, n_samples)).round(2)

    # Number of services (phone, streaming, backup, etc.)
    num_services = np.random.poisson(lam=3, size=n_samples)
    num_services = np.clip(num_services, 0, 8)

    # Generate churn based on realistic factors
    churn_prob = np.zeros(n_samples)

    # Month-to-month contracts churn more
    churn_prob += np.where(contracts == "month-to-month", 0.25, 0)
    churn_prob += np.where(contracts == "one-year", 0.05, 0)

    # Short tenure = higher churn
    churn_prob += np.where(tenure < 6, 0.2, 0)
    churn_prob += np.where(tenure < 12, 0.1, 0)
    churn_prob -= np.where(tenure > 48, 0.15, 0)

    # Fiber optic has more churn (higher cost, more issues)
    churn_prob += np.where(internet == "fiber_optic", 0.1, 0)

    # Electronic check = higher churn (less committed)
    churn_prob += np.where(payment == "electronic_check", 0.1, 0)

    # High monthly charges = more churn
    churn_prob += np.where(monthly_charges > 80, 0.1, 0)

    # Few services = less engaged
    churn_prob += np.where(num_services < 2, 0.1, 0)

    # Add noise
    churn_prob += np.random.normal(0, 0.05, n_samples)
    churn_prob = np.clip(churn_prob, 0.02, 0.95)

    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contracts,
        "internet_service": internet,
        "payment_method": payment,
        "num_services": num_services,
        "churn": churn,
    })

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that help the model.

    Feature engineering is where domain knowledge meets data science.
    """
    df = df.copy()

    # Average charge per month of tenure
    df["avg_charge_per_month"] = (
        df["total_charges"] / df["tenure"].clip(lower=1)
    ).round(2)

    # Ratio of current charge to what they've been paying
    df["charge_tenure_ratio"] = (
        df["monthly_charges"] / df["avg_charge_per_month"].clip(lower=1)
    ).round(4)

    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables as integers."""
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def run_pipeline(n_samples: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data pipeline.

    Returns:
        Tuple of (train_df, test_df)
    """
    print("=" * 60)
    print("DATA PIPELINE")
    print("=" * 60)

    # Step 1: Generate / Load data
    print("\n[1/5] Generating synthetic data...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(n_samples)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  → {len(df)} samples generated")
    print(f"  → Churn rate: {df[TARGET].mean():.1%}")
    print(f"  → Saved to: {RAW_DATA_PATH}")

    # Step 2: Feature engineering
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    print(f"  → Added: avg_charge_per_month, charge_tenure_ratio")
    print(f"  → Total features: {len(df.columns) - 1}")

    # Step 3: Encode categoricals
    print("\n[3/5] Encoding categorical variables...")
    df, encoders = encode_categoricals(df)
    for col, enc in encoders.items():
        print(f"  → {col}: {list(enc.classes_)}")

    # Step 4: Validate data quality
    print("\n[4/5] Validating data quality...")
    null_counts = df.isnull().sum().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  → Null values: {null_counts}")
    print(f"  → Infinite values: {inf_counts}")
    assert null_counts == 0, "Data contains null values!"
    assert inf_counts == 0, "Data contains infinite values!"
    print(f"  → Data quality checks passed ✓")

    # Step 5: Train/test split
    print("\n[5/5] Splitting into train/test...")
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train_df, test_df = train_test_split(
        df[features + [TARGET]],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"  → Train: {len(train_df)} samples")
    print(f"  → Test:  {len(test_df)} samples")
    print(f"  → Train churn rate: {train_df[TARGET].mean():.1%}")
    print(f"  → Test churn rate:  {test_df[TARGET].mean():.1%}")

    print(f"\n{'='*60}")
    print("DATA PIPELINE COMPLETE ✓")
    print(f"{'='*60}\n")

    return train_df, test_df


if __name__ == "__main__":
    run_pipeline()
