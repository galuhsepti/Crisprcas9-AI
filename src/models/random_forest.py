"""
Random Forest model for CRISPR-Cas9 sgRNA activity prediction.

This module implements a Random Forest regressor for predicting
sgRNA activity based on bioinformatics features.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import json
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model for sgRNA activity prediction.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Training history
        self.training_history = {}
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional list of feature names
            sample_weight: Optional per-sample non-negative weights (e.g.
                domain-adaptation reweighting). Defaults to None -> canonical
                unweighted fit.

        Returns:
            Dictionary with training results
        """
        logger.info("Training Random Forest model...")
        start_time = time.time()

        # Store feature names
        self.feature_names = feature_names

        # Fit model
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.size != X.shape[0]:
                raise ValueError("sample_weight length must equal X rows")
            if np.any(sample_weight < 0) or not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight must be finite and non-negative")
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        # Calculate training time
        training_time = time.time() - start_time

        # Get feature importances
        importances = self.model.feature_importances_

        # Store training history
        self.training_history = {
            'training_time': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_importances': importances.tolist() if feature_names is None else 
                dict(zip(feature_names, importances.tolist()))
        }
        if sample_weight is not None:
            self.training_history['sample_weight'] = {
                'mean': float(np.mean(sample_weight)),
                'min': float(np.min(sample_weight)),
                'max': float(np.max(sample_weight)),
            }

        self.is_fitted = True

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Number of trees: {self.n_estimators}")

        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs
        )
        
        # Convert negative MSE to RMSE
        rmse_scores = np.sqrt(-cv_scores)
        
        results = {
            'cv_scores': rmse_scores.tolist(),
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'min_rmse': rmse_scores.min(),
            'max_rmse': rmse_scores.max(),
            'cv_folds': cv
        }
        
        logger.info(f"CV Results: RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        
        return results
    
    def hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        method: str = 'grid',
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid to search
            cv: Number of folds
            method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary with tuning results
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        logger.info(f"Starting hyperparameter tuning ({method} search)...")
        start_time = time.time()
        
        if method == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        search.fit(X, y)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.is_fitted = True
        
        tuning_time = time.time() - start_time
        
        results = {
            'best_params': search.best_params_,
            'best_score': np.sqrt(-search.best_score_),  # Convert to RMSE
            'cv_results': {
                'mean_test_score': np.sqrt(-search.cv_results_['mean_test_score']).tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            },
            'tuning_time': tuning_time
        }
        
        logger.info(f"Best RMSE: {results['best_score']:.4f}")
        logger.info(f"Best params: {results['best_params']}")
        logger.info(f"Tuning completed in {tuning_time:.2f} seconds")
        
        return results
    
    def get_feature_importance(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        model_data = {
            'model': self.model,
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestModel':
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded RandomForestModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance with saved hyperparameters
        instance = cls(**model_data['hyperparameters'])
        instance.model = model_data['model']
        instance.training_history = model_data['training_history']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state
        }
