"""
test_pipeline.py — Tests for the MLOps pipeline.

Testing ML pipelines is different from testing regular software.
We test:
1. Data pipeline produces valid output
2. Model can train without errors
3. Predictions are in expected range
4. API endpoints respond correctly
5. Feature engineering is deterministic

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline import generate_synthetic_data, engineer_features
from config import TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class TestDataPipeline:
    """Tests for data generation and feature engineering."""

    def test_generate_data_shape(self):
        """Data should have expected number of rows and columns."""
        df = generate_synthetic_data(n_samples=100)
        assert len(df) == 100
        assert TARGET in df.columns

    def test_generate_data_no_nulls(self):
        """Generated data should have no null values."""
        df = generate_synthetic_data(n_samples=500)
        assert df.isnull().sum().sum() == 0

    def test_churn_is_binary(self):
        """Target variable should be 0 or 1."""
        df = generate_synthetic_data(n_samples=500)
        assert set(df[TARGET].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self):
        """Churn rate should be between 10% and 60% (realistic)."""
        df = generate_synthetic_data(n_samples=2000)
        churn_rate = df[TARGET].mean()
        assert 0.1 < churn_rate < 0.6, f"Churn rate {churn_rate:.2%} is unrealistic"

    def test_tenure_range(self):
        """Tenure should be between 1 and 72 months."""
        df = generate_synthetic_data(n_samples=1000)
        assert df["tenure"].min() >= 1
        assert df["tenure"].max() <= 72

    def test_monthly_charges_positive(self):
        """Monthly charges should be positive."""
        df = generate_synthetic_data(n_samples=1000)
        assert (df["monthly_charges"] > 0).all()

    def test_feature_engineering(self):
        """Feature engineering should add expected columns."""
        df = generate_synthetic_data(n_samples=100)
        df = engineer_features(df)
        assert "avg_charge_per_month" in df.columns
        assert "charge_tenure_ratio" in df.columns

    def test_feature_engineering_no_infinities(self):
        """Engineered features should not contain infinities."""
        df = generate_synthetic_data(n_samples=500)
        df = engineer_features(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df[numeric_cols]).any().any()

    def test_feature_engineering_deterministic(self):
        """Same input should produce same output."""
        df = generate_synthetic_data(n_samples=100)
        result1 = engineer_features(df)
        result2 = engineer_features(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_contract_types_valid(self):
        """Contract types should be from expected set."""
        df = generate_synthetic_data(n_samples=1000)
        valid = {"month-to-month", "one-year", "two-year"}
        assert set(df["contract_type"].unique()).issubset(valid)

    def test_internet_service_valid(self):
        """Internet service should be from expected set."""
        df = generate_synthetic_data(n_samples=1000)
        valid = {"fiber_optic", "dsl", "none"}
        assert set(df["internet_service"].unique()).issubset(valid)


class TestModelTraining:
    """Tests for model creation and training."""

    def test_model_trains_without_error(self):
        """Model should train without crashing."""
        from train import create_model
        from sklearn.preprocessing import LabelEncoder

        df = generate_synthetic_data(n_samples=200)
        df = engineer_features(df)

        for col in CATEGORICAL_FEATURES:
            df[col] = LabelEncoder().fit_transform(df[col])

        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[features]
        y = df[TARGET]

        for model_type in ["logistic_regression", "random_forest", "xgboost"]:
            model = create_model(model_type)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(X)

    def test_predictions_are_binary(self):
        """Model predictions should be 0 or 1."""
        from train import create_model
        from sklearn.preprocessing import LabelEncoder

        df = generate_synthetic_data(n_samples=200)
        df = engineer_features(df)

        for col in CATEGORICAL_FEATURES:
            df[col] = LabelEncoder().fit_transform(df[col])

        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[features]
        y = df[TARGET]

        model = create_model("random_forest")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_in_range(self):
        """Prediction probabilities should be between 0 and 1."""
        from train import create_model
        from sklearn.preprocessing import LabelEncoder

        df = generate_synthetic_data(n_samples=200)
        df = engineer_features(df)

        for col in CATEGORICAL_FEATURES:
            df[col] = LabelEncoder().fit_transform(df[col])

        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[features]
        y = df[TARGET]

        model = create_model("xgboost")
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestFeaturePreparation:
    """Tests for the serving layer feature preparation."""

    def test_encoding_maps_complete(self):
        """Encoding maps should cover all possible values."""
        from serve import ENCODING_MAPS

        df = generate_synthetic_data(n_samples=1000)

        for col in ["contract_type", "internet_service", "payment_method"]:
            unique_values = set(df[col].unique())
            encoded_values = set(ENCODING_MAPS[col].keys())
            assert unique_values.issubset(encoded_values), \
                f"Missing encodings for {col}: {unique_values - encoded_values}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
