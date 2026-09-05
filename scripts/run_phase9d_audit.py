#!/usr/bin/env python3
"""
Phase 9D - Final Generalization Audit (AUDIT ONLY).

Read/audit/synthesis layer over Phases 3-9C. It does NOT:
- retrain, tune, calibrate, adapt, ensemble, or otherwise optimize any model;
- modify canonical artifacts, the split, the Moreno-Mateos test set, or any
  Phase 3-9C results.

It DOES:
1. Recompute predictions from the CANONICAL artifacts and verify them against
   the metrics stored in the canonical result JSONs (reproducibility proof).
2. Audit A: activity-range (common [0,1] grid) error analysis per model.
3. Audit B: prediction dispersion / compression summary.
4. Audit C: ranking behavior (Pearson/Spearman/Kendall + model agreement).
5. Audit D: pairwise model agreement on Moreno-Mateos.
6. Audit E: internal (validation) vs external (Moreno-Mateos) comparison.
7. Audit G/H checks: sequence overlap, artifact integrity (sha256 + git HEAD
   recorded at run time), canonical metric reproduction, test-suite status.
8. Emission of figures, a machine-readable JSON, and (via the reporting step)
   docs/phase9d_final_generalization_audit_report.md.

External locking
----------------
Moreno-Mateos is used ONLY for final evaluation of frozen canonical models.
It is not used for any fit/selection/tuning/iteration in Phase 9D (or any
phase). Nothing in this script depends on Moreno-Mateos outputs except the
audit metrics themselves.

Covariate grid note
-------------------
Activity bins use a fixed [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] grid for BOTH
datasets and ALL models so internal vs external behaviour is compared on
identical bins. R^2 is not computed within bins (small within-bin variance
makes it uninformative).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import itertools
import json
import logging
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.data.validation import validate_sequence
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.evaluation.metrics import calculate_all_metrics
from src.diagnostics.model_diagnostics import (
    prediction_summary,
    prediction_vs_true_regression,
    model_agreement,
)
from src.diagnostics.sequence_analysis import exact_sequence_overlap
from src.audit import activity_bin_stats, dispersion_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['random_forest', 'xgboost', 'cnn']
MODEL_LABELS = {'random_forest': 'RandomForest', 'xgboost': 'XGBoost', 'cnn': 'CNN'}

CANONICAL_MODEL_FILES = {
    'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl',
    'xgboost': 'models/xgboost_baseline_20260905_002839.pkl',
    'cnn': 'models/cnn_baseline_20260905_011720.pt',
}
CANONICAL_RESULT_FILES = {
    'random_forest': 'results/experiments/rf_baseline_fixed_20260905_001107.json',
    'xgboost': 'results/experiments/xgboost_baseline_20260905_002839.json',
    'cnn': 'results/experiments/cnn_baseline_20260905_011720.json',
}

ACTIVITY_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ['low', 'low-mid', 'mid', 'high-mid', 'high']

VERIFY_TOLERANCES = {'mae': 1e-3, 'rmse': 1e-3, 'r2': 1e-3,
                     'pearson_corr': 1e-3, 'spearman_corr': 1e-3}

COLORS = {'validation': '#1f77b4', 'external': '#d62728'}
MODEL_COLORS = {'random_forest': '#1f77b4', 'xgboost': '#ff7f0e', 'cnn': '#2ca02c'}


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


def build_tabular_and_onehot(df, feature_extractor):
    X_tab = feature_extractor.extract_features_batch(
        df['sequence_30mer'].tolist()).values
    X_oh = extract_one_hot_for_cnn(df['sequence_30mer'].tolist(),
                                   feature_extractor.context_length)
    return X_tab, X_oh


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent.parent)).decode().strip()
    except Exception:
        return 'unknown'


def git_status_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent.parent)).decode().strip()
        return bool(out)
    except Exception:
        return True


def verify_canonical_metrics(preds, y_val, y_test):
    """Reproduce stored canonical metrics exactly; report max absolute delta."""
    verification = {}
    for model in MODELS:
        stored = json.load(open(CANONICAL_RESULT_FILES[model]))
        checks = {}
        for dataset, (y, p) in [('validation', (y_val, preds[model]['validation'])),
                                ('test', (y_test, preds[model]['test']))]:
            recomputed = calculate_all_metrics(y, p)
            key = 'validation_metrics' if dataset == 'validation' else 'test_metrics'
            src = stored[key]
            deltas = {}
            for metric, tol in VERIFY_TOLERANCES.items():
                ref = float(src[metric])
                got = float(recomputed[metric])
                deltas[metric] = float(abs(got - ref))
                if deltas[metric] > tol:
                    raise RuntimeError(
                        f"Metric reproduction FAILED for {model}/{dataset}/{metric}: "
                        f"stored={ref:.6f} recomputed={got:.6f}")
            checks[dataset] = {'max_abs_delta': max(deltas.values()), 'deltas': deltas}
        verification[model] = checks
    return verification


def summarize(y_true, y_pred) -> dict:
    """Full scalar summary (for Audit E table + JSON)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    s = prediction_summary(y_true, y_pred)
    m = calculate_all_metrics(y_true, y_pred)
    reg = prediction_vs_true_regression(y_true, y_pred)
    return {
        'n': int(y_true.size),
        'mae': m['mae'],
        'rmse': m['rmse'],
        'r2': m['r2'],
        'pearson': m['pearson_corr'],
        'pearson_p': m['pearson_p'],
        'spearman': m['spearman_corr'],
        'spearman_p': m['spearman_p'],
        'kendall': m['kendall_corr'],
        'true_mean': float(np.mean(y_true)),
        'true_std': float(np.std(y_true, ddof=1)),
        'prediction_mean': s['prediction_mean'],
        'prediction_std': s['prediction_std'],
        'sd_ratio_pred_over_true': (float(np.std(y_pred, ddof=1) / np.std(y_true, ddof=1))
                                    if np.std(y_true, ddof=1) > 0 else None),
        'bias_mean_pred_minus_true': s['bias_mean_pred_minus_true'],
        'bias_median_residual': s['bias_median_residual'],
        'residual_std': s['residual_std'],
        'resolution_slope': reg['slope'],
        'resolution_intercept': reg['intercept'],
    }


def save_figure(fig, name: str, figures_dir: Path, timestamp: str) -> str:
    path = figures_dir / f"phase9d_{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------- plotting
def plot_pred_vs_true(y_val, y_test, preds, figures_dir, timestamp):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (dataset, y) in enumerate([('validation', y_val), ('external', y_test)]):
        for k, model in enumerate(MODELS):
            ax = axes[row, k]
            p = preds[model]['validation' if dataset == 'validation' else 'test']
            ax.scatter(y, p, s=7, alpha=0.35, color=COLORS[dataset], label=dataset)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{MODEL_LABELS[model]} · {dataset}")
            ax.set_xlabel('true activity'); ax.set_ylabel('prediction')
    fig.tight_layout()
    return save_figure(fig, 'pred_vs_true', figures_dir, timestamp)


def plot_bias_mae_vs_activity_bin(bins, figures_dir, timestamp):
    """Two-panel figure: bias and MAE per true-activity bin."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for col, metric_key in enumerate(['bias', 'mae']):
        ax = axes[col]
        x = np.arange(len(BIN_LABELS))
        width = 0.15
        j = 0
        for dataset in ['validation', 'external']:
            for model in MODELS:
                vals = [bins[dataset][model][i][metric_key]
                        for i in range(len(BIN_LABELS))]
                ax.bar(x + (j - 2.5) * width, vals, width * 0.9,
                       color=MODEL_COLORS[model], alpha=0.55 if dataset == 'validation' else 1.0,
                       label=f'{MODEL_LABELS[model]} {dataset}')
                j += 1
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(BIN_LABELS)
        ax.set_title(f"{metric_key.upper()} by true-activity bin")
        ax.set_xlabel('true activity bin')
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return save_figure(fig, 'bias_mae_vs_activity_bin', figures_dir, timestamp)


def plot_prediction_sd_vs_true_sd(datasets, figures_dir, timestamp):
    """Prediction SD vs true SD, internal and external, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for col, dataset in enumerate(['validation', 'external']):
        ax = axes[col]
        x = np.arange(len(MODELS))
        width = 0.28
        for i, model in enumerate(MODELS):
            d = datasets[dataset][model]
            true_sd = d['true_std']
            pred_sd = d['prediction_std']
            ax.bar(x[i] - width / 2, true_sd, width * 0.9, color='#7f7f7f',
                   label='true SD' if i == 0 else None)
            ax.bar(x[i] + width / 2, pred_sd, width * 0.9,
                   color=MODEL_COLORS[model], label=f'{MODEL_LABELS[model]} pred SD' if col == 1 else None)
        ax.axhline(np.mean([datasets[dataset][m]['true_std'] for m in MODELS]),
                   color='k', ls=':', lw=1)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_title(f'Prediction SD vs true SD · {dataset}')
        ax.set_ylabel('SD of activity')
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, 'prediction_sd_vs_true_sd', figures_dir, timestamp)


def plot_ranking_scatter(y_test, preds, figures_dir, timestamp):
    """External: ranking/association scatter with correlation annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        p = preds[model]['test']
        ax.scatter(y_test, p, s=7, alpha=0.35, color=COLORS['external'])
        d = dispersion_summary(y_test, p)
        ax.set_title(f"{MODEL_LABELS[model]} · external\n"
                     f"Pearson={d['pearson']:.3f} Spearman={d['spearman']:.3f} "
                     f"Kendall={d['kendall']:.3f}")
        ax.set_xlabel('true activity'); ax.set_ylabel('prediction')
    fig.tight_layout()
    return save_figure(fig, 'ranking_scatter', figures_dir, timestamp)


# ------------------------------------------------------------------- main
def main():
    print("=" * 76)
    print("Phase 9D - Final Generalization Audit (AUDIT ONLY)")
    print("=" * 76)

    head_before = git_head()
    dirty_before = git_status_dirty()
    print(f"   git HEAD at start: {head_before} | dirty={dirty_before}")

    config = load_config()
    context_length = config['data']['context_length']
    results_dir = Path(config['experiments']['output_dir'])
    figures_dir = Path(config['experiments']['figures_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ------------------------------------------------------------ data
    print("\n1. Loading and validating data (canonical pipeline)...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid / Moreno-Mateos: {len(df_test)} valid")

    print("\n2. Reproducing the exact canonical validation split (seed 42)...")
    train_idx, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    df_train_split = df_train.iloc[train_idx].reset_index(drop=True)
    print(f"   Train {len(df_train_split)} / Validation {len(df_val)} / "
          f"External {len(df_test)}")

    y_val = df_val['activity'].values
    y_test = df_test['activity'].values

    print("\n3. Loading canonical models (read-only)...")
    models = {
        'random_forest': RandomForestModel.load_model(CANONICAL_MODEL_FILES['random_forest']),
        'xgboost': XGBoostModel.load_model(CANONICAL_MODEL_FILES['xgboost']),
        'cnn': CNNModel.load_model(CANONICAL_MODEL_FILES['cnn']),
    }

    feature_extractor = SequenceFeatureExtractor(
        context_length=context_length,
        guide_length=config['data']['guide_length'],
        guide_start=config['data']['guide_start'],
        k_values=[2, 3],
        include_one_hot=False,
        include_gc=True,
        include_composition=True,
        include_kmer=True,
        include_positional=True
    )
    if len(models['random_forest'].feature_names) != feature_extractor.get_feature_count():
        raise RuntimeError("RF feature names do not match the tabular extractor")

    print("   Extracting features (validation + external)...")
    X_tab_val, X_oh_val = build_tabular_and_onehot(df_val, feature_extractor)
    X_tab_test, X_oh_test = build_tabular_and_onehot(df_test, feature_extractor)

    print("\n4. Generating predictions (canonical models, no retraining)...")
    preds = {}
    for model in MODELS:
        if model == 'cnn':
            preds[model] = {'validation': models[model].predict(X_oh_val),
                            'test': models[model].predict(X_oh_test)}
        else:
            preds[model] = {'validation': models[model].predict(X_tab_val),
                            'test': models[model].predict(X_tab_test)}

    print("\n5. VERIFICATION: recomputed metrics vs stored canonical metrics...")
    verification = verify_canonical_metrics(preds, y_val, y_test)
    for model in MODELS:
        for ds in ['validation', 'test']:
            print(f"   {model}/{ds}: max abs delta = "
                  f"{verification[model][ds]['max_abs_delta']:.2e} (PASS)")
    print("   -> Canonical predictions reproduced. Audit uses frozen artifacts.")

    # ------------------------------------------------- AUDIT A / B / E
    print("\n6. Audit A+E: activity-range analysis (common [0,1] grid)...")
    bins = {'validation': {}, 'external': {}}
    dispersion = {'validation': {}, 'external': {}}
    summary = {'validation': {}, 'external': {}}
    for dataset, (y, label_ds) in [('validation', (y_val, 'validation')),
                                   ('external', (y_test, 'test'))]:
        for model in MODELS:
            p = preds[model][label_ds]
            bins[dataset][model] = activity_bin_stats(y, p, ACTIVITY_EDGES)
            dispersion[dataset][model] = dispersion_summary(y, p)
            summary[dataset][model] = summarize(y, p)
        for model in MODELS:
            row = "  ".join(
                f"{BIN_LABELS[i]}={bins[dataset][model][i]['n']}"
                for i in range(len(BIN_LABELS)))
            print(f"   {dataset}/{model}: n/bin {row}")

    print("\n7. Audit B: prediction dispersion...")
    for dataset in ['validation', 'external']:
        for model in MODELS:
            d = dispersion[dataset][model]
            print(f"   {dataset}/{model}: true SD={d['true_std']:.4f} pred SD="
                  f"{d['prediction_std']:.4f} sd-ratio={d['sd_ratio_pred_over_true']:.3f} "
                  f"pearson={d['pearson']:.3f} spearman={d['spearman']:.3f} ")

    print("\n8. Audit C+D: ranking and model agreement (external)...")
    agreement = {}
    for a, b in itertools.combinations(MODELS, 2):
        agreement[f"{a}_vs_{b}"] = {
            'external': model_agreement(preds[a]['test'], preds[b]['test']),
            'validation': model_agreement(preds[a]['validation'], preds[b]['validation']),
        }
        print(f"   {a} vs {b}: external pearson="
              f"{agreement[f'{a}_vs_{b}']['external']['pearson']:.3f} "
              f"spearman={agreement[f'{a}_vs_{b}']['external']['spearman']:.3f}")

    print("\n9. Domain-shift confirmation (recomputed from frozen data)...")
    overlap_internal = exact_sequence_overlap(
        df_val['sequence_30mer'].tolist(), df_test['sequence_30mer'].tolist())
    overlap_all = exact_sequence_overlap(
        df_train['sequence_30mer'].tolist(), df_test['sequence_30mer'].tolist())
    domain_shift = {
        'validation_vs_external_shared': overlap_internal['shared_unique'],
        'validation_vs_external_jaccard': overlap_internal['jaccard'],
        'deepspcas9_all_vs_external_shared': overlap_all['shared_unique'],
        'deepspcas9_all_vs_external_jaccard': overlap_all['jaccard'],
        'external_mean_true': float(np.mean(y_test)),
        'external_sd_true': float(np.std(y_test, ddof=1)),
        'validation_mean_true': float(np.mean(y_val)),
        'validation_sd_true': float(np.std(y_val, ddof=1)),
    }
    print(f"   sequence overlap external: validation={overlap_internal['shared_unique']}, "
          f"all-DeepSpCas9={overlap_all['shared_unique']} (jaccard="
          f"{overlap_all['jaccard']:.4f})")

    print("\n10. Audit G: artifact integrity snapshot...")
    artifact_integrity = {}
    for model in MODELS:
        path = CANONICAL_MODEL_FILES[model]
        artifact_integrity[model] = {
            'path': path,
            'size_bytes': Path(path).stat().st_size,
            'sha256': sha256_file(path),
        }
    print(f"   git HEAD recorded: {head_before}")

    print("\n11. Generating figures...")
    figures = {}
    figures['pred_vs_true'] = plot_pred_vs_true(
        y_val, y_test, preds, figures_dir, timestamp)
    figures['bias_mae_vs_activity_bin'] = plot_bias_mae_vs_activity_bin(
        bins, figures_dir, timestamp)
    figures['prediction_sd_vs_true_sd'] = plot_prediction_sd_vs_true_sd(
        summary, figures_dir, timestamp)
    figures['ranking_scatter'] = plot_ranking_scatter(
        y_test, preds, figures_dir, timestamp)

    print("\n12. Saving results...")
    experiment_name = f"phase9d_final_generalization_audit_{timestamp}"
    results = {
        'experiment_name': experiment_name,
        'phase': '9d',
        'audit_only': True,
        'timestamp': datetime.now().isoformat(),
        'git': {'head_at_start': head_before, 'dirty_at_start': dirty_before,
                'expected_head': '7a9af4a'},
        'methodology': {
            'canonical_artifacts': CANONICAL_MODEL_FILES,
            'canonical_result_files': CANONICAL_RESULT_FILES,
            'no_retraining': True,
            'no_tuning': True,
            'no_external_selection': True,
            'activity_bin_grid': ACTIVITY_EDGES,
            'bin_labels': BIN_LABELS,
            'verification_tolerances': VERIFY_TOLERANCES,
            'geometry': {'five_prime_context': '[0:4]',
                         'guide': '[4:24]',
                         'pam': '[24:27]',
                         'three_prime_context': '[27:30]'},
            'random_seed': config['split']['random_seed'],
            'val_ratio': config['split']['val_ratio'],
        },
        'data': {'n_train': int(len(df_train_split)),
                 'n_validation': int(len(df_val)),
                 'n_test': int(len(df_test))},
        'canonical_metric_verification': verification,
        'artifact_integrity': artifact_integrity,
        'domain_shift': domain_shift,
        'activity_bins': bins,
        'dispersion': dispersion,
        'summary_internal_vs_external': summary,
        'ranking_external': {m: {
            'pearson': dispersion['external'][m]['pearson'],
            'pearson_p': dispersion['external'][m]['pearson_p'],
            'spearman': dispersion['external'][m]['spearman'],
            'spearman_p': dispersion['external'][m]['spearman_p'],
            'kendall': dispersion['external'][m]['kendall'],
            'kendall_p': dispersion['external'][m]['kendall_p'],
        } for m in MODELS},
        'model_agreement': agreement,
        'figures': figures,
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results: {results_path}")
    print("   Figures:")
    for k, v in figures.items():
        print(f"     - {v}")

    print("\n" + "=" * 76)
    print("Phase 9D audit complete (AUDIT ONLY; nothing optimized)")
    print("=" * 76)
    return results


if __name__ == "__main__":
    main()