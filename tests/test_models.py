"""
Unit tests for Random Forest model.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.random_forest import RandomForestModel


class TestRandomForestModel:
    """Tests for RandomForestModel."""
    
    def create_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
        return X, y
    
    def test_init(self):
        """Test model initialization."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        assert model.n_estimators == 10
        assert not model.is_fitted
    
    def test_fit(self):
        """Test model training."""
        X, y = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        history = model.fit(X, y)
        
        assert model.is_fitted
        assert 'training_time' in history
        assert 'n_samples' in history
    
    def test_predict(self):
        """Test prediction."""
        X, y = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X[:10])
        
        assert y_pred.shape == (10,)
        assert not np.isnan(y_pred).any()
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        X, _ = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with pytest.raises(ValueError):
            model.predict(X)
    
    def test_cross_validate(self):
        """Test cross-validation."""
        X, y = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        cv_results = model.cross_validate(X, y, cv=3)
        
        assert 'mean_rmse' in cv_results
        assert 'std_rmse' in cv_results
        assert len(cv_results['cv_scores']) == 3
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        X, y = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        model.fit(X, y, feature_names=[f'feature_{i}' for i in range(10)])
        
        importance_df = model.get_feature_importance(top_k=5)
        
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_get_params(self):
        """Test parameter extraction."""
        model = RandomForestModel(n_estimators=50, max_depth=5, random_state=42)
        
        params = model.get_params()
        
        assert params['n_estimators'] == 50
        assert params['max_depth'] == 5
        assert params['random_state'] == 42
    
    def test_save_load(self, tmp_path):
        """Test model saving and loading."""
        X, y = self.create_sample_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save
        save_path = tmp_path / "test_model.pkl"
        model.save_model(str(save_path))
        
        # Load
        loaded_model = RandomForestModel.load_model(str(save_path))
        
        assert loaded_model.is_fitted
        assert loaded_model.n_estimators == 10
        
        # Compare predictions
        y_pred_original = model.predict(X[:5])
        y_pred_loaded = loaded_model.predict(X[:5])
        np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)


class TestHyperparameterTuning:
    """Tests for hyperparameter tuning."""
    
    def create_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.random.randn(5) + np.random.randn(50) * 0.1
        return X, y
    
    def test_grid_search(self):
        """Test grid search tuning."""
        X, y = self.create_sample_data()
        model = RandomForestModel(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        results = model.hyperparameter_tuning(
            X, y,
            param_grid=param_grid,
            cv=2,
            method='grid'
        )
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert model.is_fitted
    
    def test_random_search(self):
        """Test random search tuning."""
        X, y = self.create_sample_data()
        model = RandomForestModel(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': [5, 10, None]
        }
        
        results = model.hyperparameter_tuning(
            X, y,
            param_grid=param_grid,
            cv=2,
            method='random',
            n_iter=4
        )
        
        assert 'best_params' in results
        assert 'best_score' in results
