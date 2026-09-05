#!/usr/bin/env python3
"""
Phase 7 - Sequence-Region Ablation Study (CNN).

Determines which parts of the 30-mer window drive CNN predictive performance
by training CNN variants on restricted sequence regions with an IDENTICAL
training recipe (kernels [5,7,9], 64 filters, dense 64, dropout 0.3,
lr 1e-3, batch 32, patience 10, seed 42):

    - full        : [0:30]         (canonical input)
    - guide       : [4:24]         (guide sequence only)
    - guide_pam   : [4:27]         (guide + NGG PAM)
    - context_pam : [0:4] + [24:30] (flanking context + PAM, NO guide)

Only new variant models are trained here; the canonical CNN and its
artifacts are never modified. The 'full' region reuses the canonical saved
model for a pipeline-consistency check and a uniform comparison basis.

Methodology:
  - Same validated sequences, same fixed split (seed 42), same test set
    (Moreno-Mateos). No enrichment on the test set.
  - Validation split was used for early stopping / model selection (D-006);
    primary comparison relies on the untouched Moreno-Mateos test set.
  - Percentile bootstrap 95% CIs (n_boot=1000, seed 42) on test-set metrics.
  - Results saved to results/experiments/ablation_region_phase7_*.json;
    variant models saved to models/ablation/ (gitignored).
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

from src.data.validation import validate_sequence
from src.models import CNNModel
from src.evaluation import (
    calculate_all_metrics,
    bootstrap_metric_ci
)
from src.ablation import extract_region_one_hot, region_sequence_length

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REGIONS = ['full', 'guide', 'guide_pam', 'context_pam']


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


def cnn_for_region(region: str) -> CNNModel:
    """CNNModel with the canonical recipe adapted to the region length."""
    return CNNModel(
        input_length=region_sequence_length(region),
        n_nucleotides=4,
        conv_n_filters=64,
        conv_kernel_sizes=[5, 7, 9],
        dense_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=10,
        optimizer='adam',
        loss='mse',
        random_state=42,
        use_cpu_threads=4
    )


def main():
    print("=" * 70)
    print("Phase 7 - Sequence-Region Ablation Study (CNN)")
    print("=" * 70)

    config = load_config()
    context_length = config['data']['context_length']

    results_dir = Path(config['experiments']['output_dir'])
    artifacts_dir = Path('models/ablation')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and validating data (same pipeline as training)...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid / "
          f"Moreno-Mateos: {len(df_test)} valid")

    print("\n2. Region one-hot encoding...")
    region_one_hot = {}
    for region in REGIONS:
        region_one_hot[region] = {
            'train_full': extract_region_one_hot(
                df_train['sequence_30mer'].tolist(), region, context_length),
            'test': extract_region_one_hot(
                df_test['sequence_30mer'].tolist(), region, context_length)
        }
        print(f"   {region:<12} train {region_one_hot[region]['train_full'].shape} "
              f"test {region_one_hot[region]['test'].shape}")

    print("\n3. Reconstructing the exact train/validation split (seed=42)...")
    y_train_full = df_train['activity'].values
    y_test = df_test['activity'].values
    split = {}
    _, val_idx_full = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    train_mask = np.ones(len(df_train), dtype=bool)
    train_mask[val_idx_full] = False
    n_train = int(train_mask.sum())
    n_val = int((~train_mask).sum())
    print(f"   Train {n_train} / Validation {n_val} / Test {len(y_test)}")
    split['train_mask'] = train_mask

    print("\n4. Training / loading region variants...")
    results = {}
    canonical_params = None
    for region in REGIONS:
        X_full = region_one_hot[region]['train_full']
        X_train_r = X_full[train_mask]
        y_train_r = y_train_full[train_mask]
        X_val_r = X_full[~train_mask]
        y_val_r = y_train_full[~train_mask]
        X_test_r = region_one_hot[region]['test']

        if region == 'full':
            model = CNNModel.load_model(
                'models/cnn_baseline_20260905_011720.pt')
            canonical_params = model.get_params()
            entry = {
                'source': 'canonical model (cnn_baseline_20260905_011720.pt)',
                'trained_in_phase5': True
            }
        else:
            model = cnn_for_region(region)
            logger.info(f"Training {region} CNN "
                        f"(input_length={model.input_length})...")
            history = model.fit(
                X_train_r, y_train_r,
                X_val=X_val_r, y_val=y_val_r,
                verbose=False
            )
            artifact = artifacts_dir / f"cnn_{region}.pt"
            model.save_model(str(artifact))
            entry = {
                'source': f'models/ablation/cnn_{region}.pt',
                'trained_in_phase5': False,
                'training': {
                    'best_epoch': history.get('best_epoch'),
                    'total_epochs_run': history.get('total_epochs_run'),
                    'best_val_loss': history.get('best_val_loss')
                }
            }
            print(f"   {region}: best_epoch={history.get('best_epoch')}, "
                  f"total_epochs_run={history.get('total_epochs_run')}, "
                  f"best_val_loss={history.get('best_val_loss'):.6f}")

        val_pred = model.predict(X_val_r)
        test_pred = model.predict(X_test_r)

        results[region] = {
            **entry,
            'input_length': region_sequence_length(region),
            'validation': calculate_all_metrics(y_val_r, val_pred),
            'test': calculate_all_metrics(y_test, test_pred),
            'bootstrap_test': {
                m: bootstrap_metric_ci(y_test, test_pred, metric=m,
                                       n_boot=1000, random_state=42)
                for m in ['r2', 'pearson', 'spearman', 'mae', 'rmse']
            }
        }

    print("\n5. Results (validation = in-domain, descriptive, D-006 caveat):")
    print(f"   {'Region':<14}{'len':<5}{'MAE':<9}{'RMSE':<9}{'R2':<9}"
          f"{'Pearson':<10}{'Spearman':<10}")
    for region in REGIONS:
        v = results[region]['validation']
        print(f"   {region:<14}{results[region]['input_length']:<5}"
              f"{v['mae']:<9.4f}{v['rmse']:<9.4f}{v['r2']:<9.4f}"
              f"{v['pearson_corr']:<10.4f}{v['spearman_corr']:<10.4f}")

    print("\n6. Results (test / Moreno-Mateos, fair comparison):")
    print(f"   {'Region':<14}{'len':<5}{'MAE':<9}{'RMSE':<9}{'R2':<9}"
          f"{'Pearson':<10}{'Spearman':<10}")
    for region in REGIONS:
        t = results[region]['test']
        print(f"   {region:<14}{results[region]['input_length']:<5}"
              f"{t['mae']:<9.4f}{t['rmse']:<9.4f}{t['r2']:<9.4f}"
              f"{t['pearson_corr']:<10.4f}{t['spearman_corr']:<10.4f}")

    print("\n7. Percentile bootstrap 95% CI on test R2 (n_boot=1000, seed 42):")
    for region in REGIONS:
        b = results[region]['bootstrap_test']['r2']
        print(f"   {region:<14} R2={b['point']:.4f} "
              f"[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")

    print("\n8. Saving consolidated results...")
    experiment_name = f"ablation_region_phase7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = {
        'experiment_name': experiment_name,
        'phase': '7',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'objective': ('Isolate which sequence regions of the 30-mer window '
                          'carry predictive signal by training CNN variants on '
                          'restricted regions (identical recipe)'),
            'geometry': {
                'guide': '[4:24]',
                'pam': '[24:27]',
                'guide_pam': '[4:27]',
                'context_pam': '[0:4] + [24:30]'
            },
            'regions_evaluated': REGIONS,
            'training_recipe': canonical_params,
            'validation_split': ('DeepSpCas9 15%, seed 42; used for early '
                                 'stopping/model selection (D-006); in-domain '
                                 'comparisons are descriptive only'),
            'test_set': ('Moreno-Mateos, untouched by all variants; fair '
                         'comparison basis'),
            'no_canonical_change': ('canonical CNN artifacts and results '
                                    'unchanged; full region reuses the '
                                    'canonical model'),
            'note': 'MAPE not used as a primary metric (D-007).'
        },
        'n_train': int(n_train),
        'n_validation': int(n_val),
        'n_test': int(len(y_test)),
        'results': {k: v for k, v in results.items() if not k.startswith('_')}
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n   Results: {results_path}")

    print("\n" + "=" * 70)
    print("Phase 7 complete")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()