#!/usr/bin/env python3
"""
Training script for Random Forest baseline model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import yaml
import json
import logging
import time
from datetime import datetime

from src.bioinformatics import SequenceFeatureExtractor
from src.data.validation import validate_sequence
from src.models import RandomForestModel
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
    print("Random Forest Baseline Training")
    print("=" * 60)

    config = load_config()
    random_seed = config['project']['random_seed']

    results_dir = Path('results/experiments')
    models_dir = Path('models')
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and validating data...")
    df_train = load_and_validate_data('data/raw/DeepSpCas9.csv')
    df_test = load_and_validate_data('data/raw/Moreno-Mateos.csv')

    print("\n2. Initializing feature extractor...")
    feature_extractor = SequenceFeatureExtractor(
        context_length=30, guide_length=20, k_values=[2, 3],
        include_one_hot=False, include_gc=True,
        include_composition=True, include_kmer=True,
        include_positional=True
    )
    feature_names = feature_extractor.get_feature_names()
    print(f"   Features: {len(feature_names)}")

    print("\n3. Extracting features...")
    X_train_full = extract_features(df_train, feature_extractor)
    y_train_full = df_train['activity'].values
    X_test = extract_features(df_test, feature_extractor)
    y_test = df_test['activity'].values
    print(f"   Training set: {X_train_full.shape}")
    print(f"   Test set: {X_test.shape}")

    print("\n4. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=random_seed
    )
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

    print("\n5. Training Random Forest (n=200, depth=20)...")
    model = RandomForestModel(
        n_estimators=200, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', random_state=random_seed
    )
    history = model.fit(X_train_full, y_train_full, feature_names=feature_names)

    print("\n6. Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_all_metrics(y_val, y_val_pred)
    print(format_metrics_report(val_metrics))

    print("\n7. Evaluating on test set (Moreno-Mateos)...")
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)
    print(format_metrics_report(test_metrics))

    print("\n8. Feature importance (top 20):")
    importance_df = model.get_feature_importance(top_k=20)
    print(importance_df.to_string(index=False))

    print("\n9. Saving results...")
    experiment_name = f"rf_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': model.get_params(),
        'training_time': history['training_time'],
        'n_features': len(feature_names),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': importance_df.to_dict('records')
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    model_path = models_dir / f"{experiment_name}.pkl"
    model.save_model(str(model_path))

    print(f"\n   Results: {results_path}")
    print(f"   Model:   {model_path}")

    print("\n" + "=" * 60)
    print("Phase 3 Complete - Random Forest Baseline")
    print("=" * 60)
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print(f"Pearson r: {test_metrics['pearson_corr']:.4f}")
    print(f"Spearman ρ:{test_metrics['spearman_corr']:.4f}")

    return results


if __name__ == "__main__":
    main()
