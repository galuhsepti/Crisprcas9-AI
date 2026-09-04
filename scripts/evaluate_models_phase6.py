#!/usr/bin/env python3
"""
Phase 6 - Cross-Model Evaluation and Comparison.

Aggregates and statistically evaluates the three trained models
(Random Forest, XGBoost baselines and the CNN primary model) on:
  - DeepSpCas9 validation split (15%, held-out; see methodology notes)
  - Moreno-Mateos independent external test set (fair comparison basis)

Methodology:
  - Same validated sequences, same feature pipelines and the SAME fixed
    split (seed 42) used during training. No retraining, no tuning.
  - The saved model artifacts are the canonical ones used for the thesis:
      * models/rf_baseline_fixed_20260905_001107.pkl
      * models/xgboost_baseline_20260905_002839.pkl
      * models/cnn_baseline_20260905_011720.pt
  - Fair significance comparisons are made on the Moreno-Mateos test set
    (untouched by all models). The validation split is reported with the
    D-006 caveat: the CNN used it for early stopping/model selection.
  - MAPE is never used (see src/evaluation/metrics.py warning).
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

from src.bioinformatics import (
    SequenceFeatureExtractor,
    extract_one_hot_for_cnn
)
from src.data.validation import validate_sequence
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.evaluation import (
    calculate_all_metrics,
    calculate_metrics_by_activity_bin,
    calculate_ranking_metrics,
    paired_error_tests,
    bootstrap_metric_ci
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
    return df[valid_mask].copy()


def extract_features(df: pd.DataFrame, feature_extractor: SequenceFeatureExtractor) -> np.ndarray:
    return feature_extractor.extract_features_batch(df['sequence_30mer'].tolist()).values


def main():
    print("=" * 70)
    print("Phase 6 - Cross-Model Evaluation and Comparison")
    print("=" * 70)

    config = load_config()
    context_length = config['data']['context_length']
    guide_length = config['data']['guide_length']
    guide_start = config['data']['guide_start']

    results_dir = Path(config['experiments']['output_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and validating data (same pipeline as training)...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid sequences")
    print(f"   Moreno-Mateos: {len(df_test)} valid sequences")

    print("\n2. Feature extraction (tabulated for RF/XGB, one-hot for CNN)...")
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

    X_tab_full = extract_features(df_train, feature_extractor)
    X_tab_test = extract_features(df_test, feature_extractor)
    X_oh_full = extract_one_hot_for_cnn(df_train['sequence_30mer'].tolist(), context_length)
    X_oh_test = extract_one_hot_for_cnn(df_test['sequence_30mer'].tolist(), context_length)
    y_train_full = df_train['activity'].values
    y_test = df_test['activity'].values
    print(f"   Tabular features: {X_tab_full.shape[1]} per sequence (train {X_tab_full.shape[0]}, test {X_tab_test.shape[0]})")
    print(f"   One-hot: {X_oh_full.shape} / {X_oh_test.shape}")

    print("\n3. Reconstructing the exact train/validation split (seed=42)...")
    X_tab_tr, X_tab_val, y_tr, y_val = train_test_split(
        X_tab_full, y_train_full,
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    _, X_oh_val, _, _ = train_test_split(
        X_oh_full, y_train_full,
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    print(f"   Train {X_tab_tr.shape[0]} / Validation {X_tab_val.shape[0]} / Test {len(y_test)}")

    print("\n4. Loading canonical saved models...")
    models = {
        'random_forest': RandomForestModel.load_model(
            'models/rf_baseline_fixed_20260905_001107.pkl'),
        'xgboost': XGBoostModel.load_model(
            'models/xgboost_baseline_20260905_002839.pkl'),
        'cnn': CNNModel.load_model(
            'models/cnn_baseline_20260905_011720.pt')
    }
    for name, model in models.items():
        logger.info(f"Loaded {name}: {model.get_params()}")

    print("\n5. Generating predictions (validation + test)...")
    preds = {}
    for name, model in models.items():
        if name == 'cnn':
            preds[name] = {
                'validation': model.predict(X_oh_val),
                'test': model.predict(X_oh_test)
            }
        else:
            preds[name] = {
                'validation': model.predict(X_tab_val),
                'test': model.predict(X_tab_test)
            }
        print(f"   {name}: validation {preds[name]['validation'].shape[0]}, "
              f"test {preds[name]['test'].shape[0]}")

    print("\n6. Scalars per model...")
    metrics = {}
    for name in models:
        metrics[name] = {
            'validation': calculate_all_metrics(y_val, preds[name]['validation']),
            'test': calculate_all_metrics(y_test, preds[name]['test'])
        }

    print("\n   Validation comparison (in-domain; CNN caveat D-006 applies):")
    print(f"     {'Metric':<14} {'CNN':<12} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr', 'kendall_corr']:
        print(f"     {m:<14} "
              f"{metrics['cnn']['validation'][m]:<12.4f} "
              f"{metrics['xgboost']['validation'][m]:<12.4f} "
              f"{metrics['random_forest']['validation'][m]:<12.4f}")
    print("\n   Test comparison (Moreno-Mateos, fair for all models):")
    print(f"     {'Metric':<14} {'CNN':<12} {'XGBoost':<12} {'RandomForest':<12}")
    for m in ['mae', 'rmse', 'r2', 'pearson_corr', 'spearman_corr', 'kendall_corr']:
        print(f"     {m:<14} "
              f"{metrics['cnn']['test'][m]:<12.4f} "
              f"{metrics['xgboost']['test'][m]:<12.4f} "
              f"{metrics['random_forest']['test'][m]:<12.4f}")

    print("\n7. Paired significance tests on the TEST set (equal footing)...")
    order = ['random_forest', 'xgboost', 'cnn']
    paired_tests = {}
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            key = f"{a}_vs_{b}"
            paired_tests[key] = {
                'squared_error': paired_error_tests(
                    y_test, preds[a]['test'], preds[b]['test'],
                    error_type='squared_error'
                ),
                'absolute_error': paired_error_tests(
                    y_test, preds[a]['test'], preds[b]['test'],
                    error_type='absolute_error'
                )
            }
            t = paired_tests[key]['squared_error']
            print(f"   {a} vs {b}: mean sq-err diff = {t['mean_diff']:.4e} "
                  f"(t p={t['t_p_value']:.3e}, wilcoxon p={t['wilcoxon_p_value']:.3e})")

    print("\n8. Bootstrap 95% CIs on the TEST set (n_boot=1000, seed=42)...")
    bootstrap = {}
    for name in models:
        bootstrap[name] = {
            m: bootstrap_metric_ci(y_test, preds[name]['test'], metric=m,
                                   n_boot=1000, random_state=42)
            for m in ['r2', 'pearson', 'spearman', 'mae', 'rmse']
        }
        print(f"   {name}: R2={bootstrap[name]['r2']['point']:.4f} "
              f"[{bootstrap[name]['r2']['ci_lower']:.4f}, {bootstrap[name]['r2']['ci_upper']:.4f}]")

    print("\n9. Per-activity-bin metrics (test set)...")
    bin_metrics = {}
    for name in models:
        bin_metrics[name] = calculate_metrics_by_activity_bin(
            y_test, preds[name]['test'], n_bins=5
        )

    print("\n10. Ranking quality (top-k precision, NDCG)...")
    ranking = {}
    for name in models:
        ranking[name] = {
            'validation': {
                **calculate_ranking_metrics(y_val, preds[name]['validation'], k=5),
                **calculate_ranking_metrics(y_val, preds[name]['validation'], k=10),
            },
            'test': {
                **calculate_ranking_metrics(y_test, preds[name]['test'], k=5),
                **calculate_ranking_metrics(y_test, preds[name]['test'], k=10),
            }
        }
        print(f"   {name}: test P@5={ranking[name]['test']['precision_at_5']:.3f} "
              f"P@10={ranking[name]['test']['precision_at_10']:.3f} "
              f"NDCG@10={ranking[name]['test']['ndcg_at_10']:.3f}")

    print("\n11. Saving consolidated results...")
    experiment_name = f"evaluation_phase6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = {
        'experiment_name': experiment_name,
        'phase': '6',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'models': 'RandomForest + XGBoost (baselines) + CNN (primary)',
            'artifacts': {
                'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl',
                'xgboost': 'models/xgboost_baseline_20260905_002839.pkl',
                'cnn': 'models/cnn_baseline_20260905_011720.pt'
            },
            'validation_split': ('DeepSpCas9 15%, seed 42. For the CNN used '
                                 'for early stopping/model selection (D-006); '
                                 'in-domain significance tests are therefore '
                                 'NOT a fair comparison'),
            'test_set': ('Moreno-Mateos, held out from all training/tuning; '
                         'the fair basis for cross-model significance tests'),
            'note': ('MAPE is computed by the generic utility but is explicitly '
                     'not used as a primary metric (instability near zero).'),
            'no_retraining': True,
            'no_tuning': True
        },
        'n_validation': int(len(y_val)),
        'n_test': int(len(y_test)),
        'scalars': metrics,
        'paired_tests_test': paired_tests,
        'bootstrap_test': bootstrap,
        'bin_metrics_test': bin_metrics,
        'ranking': ranking
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results: {results_path}")

    print("\n" + "=" * 70)
    print("Phase 6 complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()