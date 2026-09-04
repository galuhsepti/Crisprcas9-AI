#!/usr/bin/env python3
"""
Training script for XGBoost baseline model.

Methodology (identical to Phase 3 RF for fair comparison):
  - DeepSpCas9 split into train (85%) / validation (15%), fixed seed.
  - Model fitted ONLY on X_train/y_train (with early stopping on validation).
  - Validation is truly unseen (used only for early stopping/model selection).
  - Moreno-Mateos is a held-out independent test set, used ONLY for final
    evaluation, never for tuning or feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import json
import logging
from datetime import datetime

from src.bioinformatics import SequenceFeatureExtractor
from src.data.validation import validate_sequence
from src.models import XGBoostModel
from src.evaluation import (
    calculate_all_metrics,
    format_metrics_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    valid_mask = []
    for seq in df['sequence_30mer']:
        is_valid, _ = validate_sequence(seq, check_pam=False)
        valid_mask.append(is_valid)
    df_valid = df[valid_mask].copy()
    logger.info(f"Loaded {len(df_valid)}/{len(df)} valid sequences from {data_path}")
    return df_valid


def extract_features(df: pd.DataFrame, feature_extractor: SequenceFeatureExtractor) -> np.ndarray:
    features = feature_extractor.extract_features_batch(df['sequence_30mer'].tolist())
    return features.values


def main():
    print("=" * 60)
    print("XGBoost Baseline Training (Phase 4)")
    print("=" * 60)

    config = load_config()
    random_seed = config['project']['random_seed']

    context_length = config['data']['context_length']
    guide_length = config['data']['guide_length']
    guide_start = config['data']['guide_start']

    xgb_cfg = config['models']['xgboost']

    results_dir = Path(config['experiments']['output_dir'])
    models_dir = Path(config['experiments']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and validating data...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")

    print("\n2. Initializing feature extractor...")
    feature_extractor = SequenceFeatureExtractor(
        context_length=context_length,
        guide_length=guide_length,
        guide_start=guide_start,
        k_values=[2, 3],
        include_one_hot=False,
        include_gc=True,
        include_composition=True,
        include_kmer=True,
        include_positional=True
    )
    feature_names = feature_extractor.get_feature_names()
    print(f"   Features: {len(feature_names)}")

    print("\n3. Extracting features...")
    X_full = extract_features(df_train, feature_extractor)
    y_full = df_train['activity'].values
    X_test = extract_features(df_test, feature_extractor)
    y_test = df_test['activity'].values
    print(f"   DeepSpCas9: {X_full.shape}")
    print(f"   Moreno-Mateos (held-out): {X_test.shape}")

    print(f"\n4. Splitting DeepSpCas9 into train/validation (seed={random_seed})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    print(f"   Train: {X_train.shape}, Validation: {X_val.shape}")

    print("\n5. Fitting XGBoost on X_train/y_train (no early stopping, fixed n_estimators)...")
    model = XGBoostModel(
        n_estimators=xgb_cfg['n_estimators'],
        max_depth=xgb_cfg['max_depth'],
        learning_rate=xgb_cfg['learning_rate'],
        random_state=xgb_cfg['random_seed']
    )
    # Baseline uses a fixed number of boosting rounds (n_estimators=100).
    # Validation is NOT used for early stopping or model selection, matching
    # the Random Forest protocol. Fit uses train split only.
    history = model.fit(
        X_train, y_train,
        feature_names=feature_names
    )
    print(f"   n_estimators used: {model.n_estimators} (no early stopping)")

    print("\n6. Evaluating on truly unseen validation set...")
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_all_metrics(y_val, y_val_pred)
    print(format_metrics_report(val_metrics))

    print("\n7. Evaluating on held-out test set (Moreno-Mateos)...")
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)
    print(format_metrics_report(test_metrics))

    print("\n8. Feature importance (top 20):")
    importance_df = model.get_feature_importance(top_k=20)
    print(importance_df.to_string(index=False))

    # ---------- Comparison vs Random Forest ----------
    print("\n9. Comparison vs Random Forest (same data/split methodology)...")
    from src.models import RandomForestModel
    rf_model = RandomForestModel(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth'],
        min_samples_split=config['models']['random_forest']['min_samples_split'],
        min_samples_leaf=config['models']['random_forest']['min_samples_leaf'],
        max_features=config['models']['random_forest']['max_features'],
        random_state=config['models']['random_forest']['random_seed']
    )
    rf_history = rf_model.fit(X_train, y_train, feature_names=feature_names)
    rf_val_pred = rf_model.predict(X_val)
    rf_val_metrics = calculate_all_metrics(y_val, rf_val_pred)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_metrics = calculate_all_metrics(y_test, rf_test_pred)

    print("\n   Validation comparison:")
    print(f"     {'Metric':<14} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr']:
        print(f"     {m:<14} {val_metrics[m]:<12.4f} {rf_val_metrics[m]:<12.4f}")
    print("\n   Test comparison (Moreno-Mateos):")
    print(f"     {'Metric':<14} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr']:
        print(f"     {m:<14} {test_metrics[m]:<12.4f} {rf_test_metrics[m]:<12.4f}")

    print("\n10. Saving results...")
    experiment_name = f"xgboost_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    comparison = {
        'xgboost': {
            'validation': val_metrics,
            'test': test_metrics,
            'early_stopping': False,
            'n_estimators': model.n_estimators
        },
        'random_forest': {
            'validation': rf_val_metrics,
            'test': rf_test_metrics
        }
    }

    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'train_split': 'DeepSpCas9 85%',
            'validation_split': 'DeepSpCas9 15% (unseen, used only for evaluation, not model selection)',
            'test_set': 'Moreno-Mateos (held-out, never used for tuning)',
            'guide_slice': '[4:24]',
            'pam_slice': '[24:27]',
            'early_stopping': False
        },
        'hyperparameters': model.get_params(),
        'training_time': history['training_time'],
        'n_features': len(feature_names),
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(X_test.shape[0]),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': importance_df.to_dict('records'),
        'comparison': comparison
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    model_path = models_dir / f"{experiment_name}.pkl"
    model.save_model(str(model_path))

    print(f"\n   Results: {results_path}")
    print(f"   Model:   {model_path}")

    print("\n" + "=" * 60)
    print("Phase 4 complete - XGBoost baseline")
    print("=" * 60)
    print(f"Validation MAE:  {val_metrics['mae']:.4f}")
    print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"Validation R²:   {val_metrics['r2']:.4f}")
    print(f"Validation r:    {val_metrics['pearson_corr']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print(f"Test r:    {test_metrics['pearson_corr']:.4f}")

    return results


if __name__ == "__main__":
    main()