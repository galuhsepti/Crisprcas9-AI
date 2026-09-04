"""
Unit tests for XGBoost model.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.xgboost_model import XGBoostModel


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def create_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
        return X, y

    def test_init(self):
        """Test model initialization."""
        model = XGBoostModel(n_estimators=20, random_state=42)
        assert model.n_estimators == 20
        assert not model.is_fitted

    def test_fit(self):
        """Test model training."""
        X, y = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        history = model.fit(X, y)
        assert model.is_fitted
        assert 'training_time' in history

    def test_predict(self):
        """Test prediction."""
        X, y = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X[:10])
        assert y_pred.shape == (10,)
        assert not np.isnan(y_pred).any()

    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        X, _ = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_cross_validate(self):
        """Test cross-validation."""
        X, y = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        cv_results = model.cross_validate(X, y, cv=3)
        assert 'mean_rmse' in cv_results
        assert len(cv_results['cv_scores']) == 3

    def test_get_feature_importance(self):
        """Test feature importance."""
        X, y = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        model.fit(X, y, feature_names=[f'feature_{i}' for i in range(10)])
        importance_df = model.get_feature_importance(top_k=5)
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_get_params(self):
        """Test parameter extraction."""
        model = XGBoostModel(n_estimators=50, max_depth=4, learning_rate=0.05)
        params = model.get_params()
        assert params['n_estimators'] == 50
        assert params['max_depth'] == 4
        assert params['learning_rate'] == 0.05

    def test_save_load(self, tmp_path):
        """Test save/load."""
        X, y = self.create_sample_data()
        model = XGBoostModel(n_estimators=20, random_state=42)
        model.fit(X, y)
        save_path = tmp_path / "test_xgb.pkl"
        model.save_model(str(save_path))
        loaded = XGBoostModel.load_model(str(save_path))
        assert loaded.is_fitted
        np.testing.assert_array_almost_equal(model.predict(X[:5]), loaded.predict(X[:5]))


class TestXGBoostEarlyStopping:
    """Test early stopping behavior."""

    def create_sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(80, 10)
        y = X @ np.random.randn(10)
        return X, y

    def test_early_stopping(self):
        """Test early stopping with eval set."""
        X, y = self.create_sample_data()
        X_val = X[60:]
        y_val = y[60:]
        X_tr = X[:60]
        y_tr = y[:60]
        model = XGBoostModel(n_estimators=500, random_state=42)
        history = model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=10)
        assert 'best_iteration' in history
        assert model.best_iteration is not None and model.best_iteration < 500
