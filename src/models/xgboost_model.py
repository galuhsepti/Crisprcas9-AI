"""
XGBoost model for CRISPR-Cas9 sgRNA activity prediction.

This module implements an XGBoost regressor for predicting sgRNA activity
based on bioinformatics features. It mirrors the interface of
RandomForestModel for a consistent comparison.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import logging
import time

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost model for sgRNA activity prediction.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = 'reg:squarederror',
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_child_weight: Minimum sum of instance weight in a child
            subsample: Fraction of samples used per tree
            colsample_bytree: Fraction of features used per tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            objective: Learning objective
            n_jobs: Number of CPU cores to use
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
            early_stopping_rounds=None
        )
        
        # Training history
        self.training_history = {}
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: int = 20,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional list of feature names
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Stop if score doesn't improve for N rounds
            sample_weight: Optional per-sample non-negative weights (e.g.
                domain-adaptation reweighting). Defaults to None -> canonical
                unweighted fit.

        Returns:
            Dictionary with training results
        """
        logger.info("Training XGBoost model...")
        start_time = time.time()

        self.feature_names = feature_names

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.size != X.shape[0]:
                raise ValueError("sample_weight length must equal X rows")
            if np.any(sample_weight < 0) or not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight must be finite and non-negative")

        if eval_set is not None and early_stopping_rounds:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective=self.objective,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=0,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='rmse'
            )
            X_val, y_val = eval_set
            fit_kwargs = dict(eval_set=[(X_val, y_val)], verbose=False)
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            self.model.fit(X, y, **fit_kwargs)
            self.best_iteration = self.model.best_iteration
            self.best_score = self.model.best_score
        else:
            fit_kwargs = dict(verbose=False)
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            self.model.fit(X, y, **fit_kwargs)
            self.best_iteration = None
            self.best_score = None
        
        training_time = time.time() - start_time
        
        # Get feature importances
        self.training_history = {
            'training_time': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'best_iteration': self.best_iteration,
            'best_score': self.best_score
        }
        
        self.is_fitted = True
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best iteration: {self.best_iteration}")
        
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
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning (grid search).
        
        This must be done on the training split ONLY; the held-out validation
        set must never enter tuning.
        
        Args:
            X: Feature matrix (training split only)
            y: Target values (training split only)
            param_grid: Parameter grid to search
            cv: Number of folds (internal, within the training split)
            
        Returns:
            Dictionary with tuning results
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1]
            }
        
        logger.info("Starting hyperparameter tuning (grid search)...")
        start_time = time.time()
        
        search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        search.fit(X, y)
        
        self.model = search.best_estimator_
        self.is_fitted = True
        
        tuning_time = time.time() - start_time
        
        results = {
            'best_params': search.best_params_,
            'best_score': np.sqrt(-search.best_score_),
            'tuning_time': tuning_time
        }
        
        logger.info(f"Best RMSE: {results['best_score']:.4f}")
        logger.info(f"Best params: {results['best_params']}")
        
        return results
    
    def get_feature_importance(
        self,
        top_k: Optional[int] = None,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_k: Number of top features to return
            importance_type: 'weight', 'gain', or 'cover'
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        if not importance_dict:
            # Fallback: use feature_importances_ attribute
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
        else:
            if self.feature_names is not None:
                feature_map = {f'f{i}': name for i, name in enumerate(self.feature_names)}
                importance_df = pd.DataFrame({
                    'feature': [feature_map.get(k, k) for k in importance_dict.keys()],
                    'importance': list(importance_dict.values())
                })
            else:
                importance_df = pd.DataFrame({
                    'feature': list(importance_dict.keys()),
                    'importance': list(importance_dict.values())
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
            'hyperparameters': self.get_params()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostModel':
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded XGBoostModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
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
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': self.objective,
            'random_state': self.random_state
        }
