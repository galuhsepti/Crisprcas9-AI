#!/usr/bin/env python3
"""
Phase 8 - Model Interpretability.

Explains what the fitted models rely on:

CNN (primary model):
  - Position saliency  : mean |d(prediction)/d(one-hot channel)| per position.
  - Integrated Gradients with a zero baseline (steps=50).
  - Attribution mass per sequence region (guide / PAM / 5' flank / 3' flank),
    cross-checked against the Phase 7 region ablation.
  - Internal consistency: Spearman correlation between the two attribution
    profiles.

Tabular baselines:
  - Random Forest impurity-based importances and XGBoost split-gain
    importances, aggregated into conceptual groups (GC/composition, kmer_2,
    kmer_3, per-position guide one-hot), plus the top features of each model.

Interpretation guardrails:
  - Attribution is descriptive of the fitted models, NOT evidence of
    biological mechanism.
  - Vendor importances (impurity decrease / gain) are not statistical tests.
  - Analysis uses the in-domain DeepSpCas9 validation split (the models'
    native regime); no test-set (Moreno-Mateos) data is used here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import yaml
import json
import logging
from datetime import datetime

from src.data.validation import validate_sequence
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.ablation import extract_region_one_hot
from src.interpretability import (
    position_saliency,
    integrated_gradients,
    attribution_by_region,
    importance_frame,
    aggregate_importance_by_group,
    top_features
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


def main():
    print("=" * 70)
    print("Phase 8 - Model Interpretability")
    print("=" * 70)

    config = load_config()
    context_length = config['data']['context_length']
    results_dir = Path(config['experiments']['output_dir'])

    print("\n1. Loading and validating data...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid sequences")

    print("\n2. Reconstructing the train/validation split (seed=42)...")
    _, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"   Validation subset for attribution: {len(df_val)}")

    print("\n3. Loading canonical models...")
    cnn = CNNModel.load_model('models/cnn_baseline_20260905_011720.pt')
    rf = RandomForestModel.load_model('models/rf_baseline_fixed_20260905_001107.pkl')
    xgb = XGBoostModel.load_model('models/xgboost_baseline_20260905_002839.pkl')

    print("\n4. CNN attribution on the validation set...")
    X_val_oh = extract_region_one_hot(df_val['sequence_30mer'].tolist(),
                                      'full', context_length)
    saliency = position_saliency(cnn.network, X_val_oh, batch_size=256)
    ig = integrated_gradients(cnn.network, X_val_oh, steps=50, batch_size=256)
    print(f"   Saliency profile (sum={saliency.sum():.4f})")
    print(f"   IG profile       (sum={ig.sum():.4f})")

    rho_sal_ig, p_sal_ig = spearmanr(saliency, ig)
    print(f"   Spearman(saliency, IG) = {rho_sal_ig:.4f} (p={p_sal_ig:.3e})")

    cnn_by_region_sal = attribution_by_region(saliency, context_length)
    cnn_by_region_ig = attribution_by_region(ig, context_length)
    print(f"   Region shares (saliency): { {k: round(v, 4) for k, v in cnn_by_region_sal.items()} }")
    print(f"   Region shares (IG):       { {k: round(v, 4) for k, v in cnn_by_region_ig.items()} }")

    print("\n5. Tabular feature importance...")
    feature_names = rf.feature_names
    print(f"   Using {len(feature_names)} tabular features")

    rf_frame = importance_frame(rf, feature_names)
    xgb_frame = importance_frame(xgb, feature_names, importance_type='gain')

    rf_groups = aggregate_importance_by_group(rf_frame)
    xgb_groups = aggregate_importance_by_group(xgb_frame)

    print("\n   Group shares (RF impurity / XGBoost gain):")
    print(f"   {'Group':<26}{'RF share':<12}{'XGB share':<12}")
    all_groups = sorted(set(rf_groups) | set(xgb_groups))
    for g in all_groups:
        print(f"   {g:<26}{rf_groups.get(g, {}).get('share', 0.0):<12.4f}"
              f"{xgb_groups.get(g, {}).get('share', 0.0):<12.4f}")

    rf_top = top_features(rf_frame, k=15)
    xgb_top = top_features(xgb_frame, k=15)
    print("\n   Top-15 RF features:")
    for _, row in rf_top.iterrows():
        print(f"     {row['feature']:<28}{row['importance']:.4f}")
    print("\n   Top-15 XGBoost (gain) features:")
    for _, row in xgb_top.iterrows():
        print(f"     {row['feature']:<28}{row['importance']:.4f}")

    overlap = set(rf_top['feature']) & set(xgb_top['feature'])
    print(f"\n   Feature overlap in top-15: {len(overlap)} / 15 "
          f"({sorted(overlap)[:8]}...)")

    print("\n6. Cross-method view: guide-centric attribution...")
    guide_fraction_cnn_sal = cnn_by_region_sal['guide']
    guide_fraction_cnn_ig = cnn_by_region_ig['guide']
    guide_fraction_rf = rf_groups['positional_guide_onehot']['share']
    guide_fraction_xgb = xgb_groups['positional_guide_onehot']['share']
    print(f"   CNN saliency in guide:   {guide_fraction_cnn_sal:.4f}")
    print(f"   CNN IG in guide:         {guide_fraction_cnn_ig:.4f}")
    print(f"   RF  guide one-hot group: {guide_fraction_rf:.4f}")
    print(f"   XGB guide one-hot group: {guide_fraction_xgb:.4f}")

    print("\n7. Saving consolidated results...")
    experiment_name = f"interpretability_phase8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = {
        'experiment_name': experiment_name,
        'phase': '8',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'data_used': ('DeepSpCas9 validation split (in-domain regime); '
                          'no Moreno-Mateos data used in this phase'),
            'n_validation': int(len(df_val)),
            'cnn': {
                'model': 'models/cnn_baseline_20260905_011720.pt',
                'methods': ['position_saliency',
                            'integrated_gradients (zero baseline, steps=50)']
            },
            'tabular': {
                'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl'
                                 ' (impurity decrease)',
                'xgboost': 'models/xgboost_baseline_20260905_002839.pkl'
                           ' (split-gain)'
            },
            'guardrails': [
                'Attribution describes the fitted models; it is not evidence '
                'of a biological mechanism.',
                'Vendor importances are not statistical tests.',
                'Variant/group shares are descriptive within each model.'
            ]
        },
        'cnn_attribution': {
            'saliency_profile': saliency.tolist(),
            'integrated_gradients_profile': ig.tolist(),
            'spearman_saliency_vs_ig': float(rho_sal_ig),
            'spearman_p': float(p_sal_ig),
            'region_shares': {
                'saliency': cnn_by_region_sal,
                'integrated_gradients': cnn_by_region_ig
            }
        },
        'tabular_importance': {
            'random_forest': {
                'groups': rf_groups,
                'top_features': rf_top.to_dict('records')
            },
            'xgboost': {
                'groups': xgb_groups,
                'top_features': xgb_top.to_dict('records')
            },
            'top15_overlap': sorted(overlap)
        },
        'cross_method': {
            'guide_fraction_cnn_saliency': guide_fraction_cnn_sal,
            'guide_fraction_cnn_ig': guide_fraction_cnn_ig,
            'guide_fraction_rf': guide_fraction_rf,
            'guide_fraction_xgb': guide_fraction_xgb
        }
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results: {results_path}")

    print("\n" + "=" * 70)
    print("Phase 8 complete")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()