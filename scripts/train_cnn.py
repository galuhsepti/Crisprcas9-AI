#!/usr/bin/env python3
"""
Training script for CNN model (Phase 5).

Methodology:
  - DeepSpCas9 split into train (85%) / validation (15%), fixed seed 42.
  - Input: one-hot encoded 30-mer (n, 30, 4) exactly as extracted by
    extract_one_hot_for_cnn (same sequence geometry guide [4:24], PAM [24:27]).
  - CNN is the main model of the thesis; the validation set is used for early
    stopping (best model by validation loss retained) — standard for deep
    learning.
  - The validation set is a held-out validation set: it is NOT used for
    gradient updates, but it IS used for early stopping / model selection.
  - Moreno-Mateos held-out test set: used ONLY for final evaluation,
    never for training, tuning, early stopping, or model selection.
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

from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.data.validation import validate_sequence
from src.models import CNNModel, RandomForestModel, XGBoostModel
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


def main():
    print("=" * 60)
    print("CNN Training (Phase 5)")
    print("=" * 60)

    config = load_config()

    context_length = config['data']['context_length']
    guide_length = config['data']['guide_length']
    guide_start = config['data']['guide_start']

    cnn_cfg = config['models']['cnn']
    split_cfg = config['split']

    results_dir = Path(config['experiments']['output_dir'])
    models_dir = Path(config['experiments']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and validating data...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")

    print("\n2. One-hot encoding (CNN input)...")
    seq_train = df_train['sequence_30mer'].tolist()
    seq_test = df_test['sequence_30mer'].tolist()

    X_train_full = extract_one_hot_for_cnn(seq_train, context_length)
    X_test = extract_one_hot_for_cnn(seq_test, context_length)
    y_train_full = df_train['activity'].values
    y_test = df_test['activity'].values
    print(f"   DeepSpCas9 one-hot: {X_train_full.shape}")
    print(f"   Moreno-Mateos one-hot: {X_test.shape}")

    print("\n3. Splitting DeepSpCas9 into train/validation (seed=42)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=split_cfg['val_ratio'],
        random_state=split_cfg['random_seed']
    )
    print(f"   Train: {X_train.shape}, Validation: {X_val.shape}")

    print("\n4. Training CNN (early stopping on validation, main model)...")
    model = CNNModel(
        input_length=cnn_cfg['input_length'],
        n_nucleotides=cnn_cfg['n_nucleotides'],
        conv_n_filters=cnn_cfg['conv_n_filters'],
        conv_kernel_sizes=cnn_cfg['conv_kernel_sizes'],
        dense_units=cnn_cfg['dense_units'],
        dropout_rate=cnn_cfg['dropout_rate'],
        learning_rate=cnn_cfg['learning_rate'],
        batch_size=cnn_cfg['batch_size'],
        epochs=cnn_cfg['epochs'],
        patience=cnn_cfg['patience'],
        optimizer=cnn_cfg['optimizer'],
        loss=cnn_cfg['loss'],
        random_state=config['project']['random_seed']
    )
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        verbose=True
    )
    print(f"   Best validation loss: {history['best_val_loss']:.6f}")

    print("\n5. Evaluating on held-out validation set (used for early stopping, not gradient updates)...")
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_all_metrics(y_val, y_val_pred)
    print(format_metrics_report(val_metrics))

    print("\n6. Evaluating on held-out test set (Moreno-Mateos)...")
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)
    print(format_metrics_report(test_metrics))

    print("\n7. Comparison vs baselines (same data/split methodology)...")
    # --- Random Forest baseline on 197 tabular features ---
    feature_extractor = SequenceFeatureExtractor(
        context_length=context_length, guide_length=guide_length,
        guide_start=guide_start, k_values=[2, 3],
        include_one_hot=False, include_gc=True, include_composition=True,
        include_kmer=True, include_positional=True
    )
    feature_names = feature_extractor.get_feature_names()
    X_tab_full = feature_extractor.extract_features_batch(seq_train).values
    X_tab_test = feature_extractor.extract_features_batch(seq_test).values
    X_tab_tr, X_tab_val, y_tab_tr, y_tab_val = train_test_split(
        X_tab_full, y_train_full,
        test_size=split_cfg['val_ratio'], random_state=split_cfg['random_seed']
    )

    rf = RandomForestModel(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth'],
        min_samples_split=config['models']['random_forest']['min_samples_split'],
        min_samples_leaf=config['models']['random_forest']['min_samples_leaf'],
        max_features=config['models']['random_forest']['max_features'],
        random_state=config['models']['random_forest']['random_seed']
    )
    rf.fit(X_tab_tr, y_tab_tr, feature_names=feature_names)
    rf_val_metrics = calculate_all_metrics(y_tab_val, rf.predict(X_tab_val))
    rf_test_metrics = calculate_all_metrics(y_test, rf.predict(X_tab_test))

    # --- XGBoost baseline on 197 tabular features ---
    xgb = XGBoostModel(
        n_estimators=config['models']['xgboost']['n_estimators'],
        max_depth=config['models']['xgboost']['max_depth'],
        learning_rate=config['models']['xgboost']['learning_rate'],
        random_state=config['models']['xgboost']['random_seed']
    )
    xgb.fit(X_tab_tr, y_tab_tr, feature_names=feature_names)
    xgb_val_metrics = calculate_all_metrics(y_tab_val, xgb.predict(X_tab_val))
    xgb_test_metrics = calculate_all_metrics(y_test, xgb.predict(X_tab_test))

    print("\n   Validation comparison:")
    print(f"     {'Metric':<14} {'CNN':<12} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr']:
        print(f"     {m:<14} {val_metrics[m]:<12.4f} {xgb_val_metrics[m]:<12.4f} {rf_val_metrics[m]:<12.4f}")
    print("\n   Test comparison (Moreno-Mateos):")
    print(f"     {'Metric':<14} {'CNN':<12} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr']:
        print(f"     {m:<14} {test_metrics[m]:<12.4f} {xgb_test_metrics[m]:<12.4f} {rf_test_metrics[m]:<12.4f}")

    print("\n8. Saving results...")
    experiment_name = f"cnn_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    comparison = {
        'cnn': {'validation': val_metrics, 'test': test_metrics,
                'best_val_loss': history['best_val_loss'],
                'best_epoch': history['best_epoch']},
        'xgboost': {'validation': xgb_val_metrics, 'test': xgb_test_metrics},
        'random_forest': {'validation': rf_val_metrics, 'test': rf_test_metrics}
    }

    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'model': 'CNN (primary)',
            'train_split': 'DeepSpCas9 85%',
            'validation_split': 'DeepSpCas9 15% (held-out validation, not used for gradient updates but used for early stopping/model selection)',
            'test_set': 'Moreno-Mateos (held-out, never used for training/tuning)',
            'input': 'one-hot (n, 30, 4)',
            'guide_slice': '[4:24]',
            'pam_slice': '[24:27]',
            'early_stopping': True,
            'patience': cnn_cfg['patience']
        },
        'hyperparameters': model.get_params(),
        'training_time': history['training_time'],
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(X_test.shape[0]),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_history': {
            k: (v[-20:] if isinstance(v, list) else v)
            for k, v in history.items() if k != 'val_loss'
        },
        'val_loss_curve': history.get('val_loss', []),
        'comparison': comparison
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    model_path = models_dir / f"{experiment_name}.pt"
    model.save_model(str(model_path))

    print(f"\n   Results: {results_path}")
    print(f"   Model:   {model_path}")

    print("\n" + "=" * 60)
    print("Phase 5 complete - CNN")
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