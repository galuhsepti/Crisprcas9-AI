#!/usr/bin/env python3
"""
Phase 10 - Multi-Dataset Training-Domain Diversification Experiment.

Objective
---------
Test whether adding a second real experimental training domain (DeepHF) to the
canonical DeepSpCas9 training corpus improves generalisation to the LOCKED
Moreno-Mateos external test, relative to the canonical single-domain recipe.

Pre-registered protocol (see src/multidataset/config.py for full detail)
-------------------------------------------------------------------------
  Arm B  : DeepSpCas9 canonical training split only (8,599)  [baseline]
  Arm D1 : DeepSpCas9 training split + DeepHF (48,295 labelled rows)
           [PRIMARY diversified; 56,894 total]
  - Validation split is the canonical DeepSpCas9 15% (1,518) for BOTH arms
    (CNN early stopping uses it; documented conservative bias).
  - DeepHF rows enter the training split only. Moreno-Mateos is NEVER used
    for selection/tuning.
  - Primary endpoint: d_external = mean(|e_B| - |e_D1|) on Moreno-Mateos for
    the CNN (positive = diversified improves MAE). Secondary metrics for all
    three families.
  - External evaluation happens exactly ONCE per arm/model after the whole
    protocol is frozen.
  - Determine the automatic verdict (SUPPORTED / PARTIAL / NOT_SUPPORTED /
    INSUFFICIENT_EVIDENCE) with the frozen rule in config.decide_verdict.

Frozen hyperparameters are read at runtime from the loaded canonical models.
Outputs
-------
- results/experiments/phase10_multidataset_<timestamp>.json
- results/figures/phase10_*.png
- results/experiments/pool_d1_<timestamp>.csv and *external_predictions_*.csv
- models/phase10_* (artifacts; gitignored)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
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

import scipy
import sklearn
import torch
import xgboost
import openpyxl

from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.data.validation import validate_sequence
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.evaluation import (
    calculate_all_metrics,
    calculate_mae,
    calculate_rmse,
    paired_error_tests,
    bootstrap_metric_ci,
)
from src.diagnostics.model_diagnostics import (
    prediction_summary,
    prediction_vs_true_regression,
)
from src.multidataset.config import (
    ARMS,
    ARM_DEFINITIONS,
    PRIMARY_ARM,
    BASELINE_ARM,
    PRIMARY_MODEL,
    PRIMARY_ENDPOINT,
    EFFECT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    CI_HALFWIDTH_GUARD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    VERDICT_RULE,
    decide_verdict,
)
from src.multidataset.datasets import (
    DATASET_INVENTORY,
    dataset_inventory_table,
    allowed_additional_datasets,
    load_deepspcas9,
    load_moreno_mateos,
    load_deephf,
    sha256_file,
)
from src.multidataset.audit import (
    overlap_report,
    contamination_report,
    reverse_complement,
)
from src.multidataset.pool import build_training_pool, pool_summary
from src.multidataset.stats import paired_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['random_forest', 'xgboost', 'cnn']
MODEL_LABELS = {'random_forest': 'RandomForest', 'xgboost': 'XGBoost', 'cnn': 'CNN'}
ARM_COLORS = {'B': '#7f7f7f', 'D1': '#d62728', 'true': '#000000'}

DECISION_GATE_MAP = {
    'SUPPORTED': 'A - adopt diversified training (multi-dataset recipe)',
    'PARTIAL_SUPPORT': 'C - document; retain canonical as production model '
                       'until a replication is available',
    'NOT_SUPPORTED': 'B - keep the canonical single-domain recipe',
    'INSUFFICIENT_EVIDENCE': 'D - repeat with more data / tighter protocol',
}


def load_config() -> dict:
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def summarize(y_true, y_pred) -> dict:
    """Scalar evaluation summary (same helpers as Phase 9C)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = calculate_all_metrics(y_true, y_pred)
    s = prediction_summary(y_true, y_pred)
    true_std = float(np.std(y_true, ddof=1))
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
        'prediction_mean': s['prediction_mean'],
        'prediction_std': s['prediction_std'],
        'true_mean': float(np.mean(y_true)),
        'true_std': true_std,
        'std_ratio': (s['prediction_std'] / true_std) if true_std > 0 else None,
        'bias_mean_pred_minus_true': s['bias_mean_pred_minus_true'],
        'bias_median_residual': s['bias_median_residual'],
        'residual_std': s['residual_std'],
        'mape': m['mape'],
    }


def tercile_bins(y_true, y_pred) -> dict:
    """Activity tercile analysis (low/mid/high) on true activity."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    t = np.quantile(y_true, [1 / 3, 2 / 3])
    masks = {
        'low': y_true <= t[0],
        'mid': (y_true > t[0]) & (y_true < t[1]),
        'high': y_true >= t[1],
    }
    out = {'thresholds': {'low_mid_boundary': float(t[0]),
                          'mid_high_boundary': float(t[1])}}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[name] = {
            'n': int(mask.sum()),
            'mae': float(calculate_mae(yt, yp)),
            'rmse': float(calculate_rmse(yt, yp)),
            'bias': float(np.mean(yp - yt)),
            'mean_true': float(np.mean(yt)),
            'mean_pred': float(np.mean(yp)),
            'prediction_std': float(np.std(yp, ddof=1)),
        }
    return out


def gc_quintile_bins(y_true, y_pred, gc) -> dict:
    """External error analysis by GC-content quintile (per-sample GC)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    gc = np.asarray(gc, dtype=float).ravel()
    q = np.quantile(gc, [0.2, 0.4, 0.6, 0.8])
    edges = np.concatenate([[-np.inf], q, [np.inf]])
    out = {'thresholds': [float(v) for v in q]}
    for j in range(5):
        mask = (gc > edges[j]) & (gc <= edges[j + 1])
        yt, yp = y_true[mask], y_pred[mask]
        out[f'q{j + 1}'] = {
            'n': int(mask.sum()),
            'gc_mean': float(np.mean(gc[mask])),
            'mae': float(calculate_mae(yt, yp)),
            'rmse': float(calculate_rmse(yt, yp)),
            'bias': float(np.mean(yp - yt)),
            'mean_true': float(np.mean(yt)),
            'mean_pred': float(np.mean(yp)),
        }
    return out


def git_head() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(['git', 'status', '--porcelain', 'src',
                              'scripts', 'tests', 'config.yaml'],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "(clean)"
    except Exception as exc:
        return f"unavailable ({exc})"


def software_versions() -> dict:
    return {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'openpyxl': openpyxl.__version__,
    }


def save_figure(fig, name: str, figures_dir: Path, timestamp: str) -> str:
    path = figures_dir / f"phase10_{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------- plotting
def plot_pool_composition(pool, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    per = pool.groupby('source_dataset')['activity_label']
    axes[0].bar(['DeepSpCas9', 'DeepHF'],
                [per.get_group('deepspcas9').size, per.get_group('deephf').size],
                color=['#1f77b4', '#2ca02c'])
    axes[0].set_title('Pool composition')
    axes[0].set_ylabel('n sequences')
    for src, grp in pool.groupby('source_dataset'):
        axes[1].hist(grp['activity_label'], bins=30, alpha=0.5,
                     label=('DeepSpCas9' if src == 'deepspcas9' else 'DeepHF'))
    axes[1].set_title('Label distributions (raw scale, harmonised)')
    axes[1].set_xlabel('activity'); axes[1].legend() 
    gc = pool['normalized_sequence'].map(
        lambda s: np.mean([c in 'GC' for c in s]))
    axes[2].hist(gc, bins=30, alpha=0.7, color='#7f7f7f')
    axes[2].set_title('Pool GC distribution'); axes[2].set_xlabel('GC')
    fig.suptitle('Phase 10 - diversified training pool (arm D1)', y=1.02)
    fig.tight_layout()
    return save_figure(fig, 'pool_composition', figures_dir, timestamp)


def plot_external_metrics(external, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(MODELS))
    width = 0.32
    for col, metric in enumerate(['mae', 'rmse']):
        ax = axes[col]
        for j, arm in enumerate(ARMS):
            vals = [external[m][arm][metric] for m in MODELS]
            ax.bar(x + (j - 1) * width, vals, width * 0.9,
                   color=ARM_COLORS[arm],
                   label=f'Arm {arm} ({"baseline" if arm == BASELINE_ARM else "diversified"})')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_title(f'External {metric.upper()} (Moreno-Mateos, locked)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, 'external_metrics', figures_dir, timestamp)


def plot_external_delta_mae(external, paired, figures_dir, timestamp):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MODELS))
    deltas = [external[m]['D1']['mae'] - external[m]['B']['mae']
              for m in MODELS]
    ax.bar(x, deltas, color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.85)
    ax.axhline(0, color='k', lw=1)
    ax.axhline(-EFFECT_THRESHOLD, color='k', ls='--', lw=0.7,
               label=f'effect threshold (-{EFFECT_THRESHOLD})')
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel('ΔMAE external (D1 - B)')
    ax.set_title('Primary endpoint: ΔMAE(CNN); negative = diversified improves')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'external_delta_mae', figures_dir, timestamp)


def plot_pred_vs_true(y_test, preds_by_arm, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        for arm in ARMS:
            ax.scatter(y_test, preds_by_arm[model][arm], s=7, alpha=0.35,
                       color=ARM_COLORS[arm],
                       label=f'Arm {arm}')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{MODEL_LABELS[model]} · external")
        ax.set_xlabel('true'); ax.set_ylabel('prediction'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'pred_vs_true', figures_dir, timestamp)


def plot_external_error_by_bin(y_test, preds_by_arm, gc_test, figures_dir, timestamp):
    """RF MAE by activity tercile and GC quintile, B vs D1 (supporting)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for col, (kind, bins_fn) in enumerate([
            ('activity tercile',
             lambda p: tercile_bins(y_test, p)),
            ('GC quintile',
             lambda p: gc_quintile_bins(y_test, p, gc_test))]):
        ax = axes[col]
        names = (['low', 'mid', 'high'] if kind.startswith('activity')
                 else ['q1', 'q2', 'q3', 'q4', 'q5'])
        x = np.arange(len(names))
        width = 0.32
        for j, arm in enumerate(ARMS):
            b = bins_fn(preds_by_arm['random_forest'][arm])
            vals = [b[n]['mae'] for n in names]
            ax.bar(x + (j - 1) * width, vals, width * 0.9,
                   color=ARM_COLORS[arm], label=f'Arm {arm}')
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title(f'RF MAE by {kind} · external')
        ax.set_ylabel('MAE'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'external_error_by_bin', figures_dir, timestamp)


# ------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("Phase 10 - Multi-Dataset Training-Domain Diversification Experiment")
    print("=" * 72)

    config = load_config()
    context_length = config['data']['context_length']
    guide_length = config['data']['guide_length']
    guide_start = config['data']['guide_start']
    results_dir = Path(config['experiments']['output_dir'])
    figures_dir = Path(config['experiments']['figures_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    environment = software_versions()
    head = git_head()

    print("\n[0] Reproducibility environment")
    print(f"    git HEAD: {head}")
    print(f"    python {environment['python']}, torch {environment['torch']} "
          f"(cuda={environment['cuda_available']})")

    # ------------------------------------------------------------ inventory
    print("\n[1] Dataset inventory + inclusion audit")
    inv = dataset_inventory_table()
    print(inv[['Dataset', 'Included', 'N sequences', 'Label range']].to_string(index=False))
    print("\n    Full reasons are in src/multidataset/datasets.py::DATASET_INVENTORY")
    included = allowed_additional_datasets()
    print(f"    Included as additional training source: {included}")

    # ------------------------------------------------------------ data
    print("\n[2] Loading + validating data (canonical pipeline)")
    df_dsp = load_deepspcas9()
    df_mm = load_moreno_mateos()
    df_hf = load_deephf()
    print(f"    DeepSpCas9 valid: {len(df_dsp)}")
    print(f"    Moreno-Mateos valid (LOCKED): {len(df_mm)}")
    print(f"    DeepHF mapped+valid: {len(df_hf)}"
          f" (dropped {df_hf.attrs.get('n_unlabeled_dropped', 0)}"
          f" NaN-label rows of {df_hf.attrs.get('n_mapped_valid', 'n/a')} mapped)")
    hashes = {
        'deepspcas9_csv': sha256_file('data/raw/DeepSpCas9.csv'),
        'moreno_mateos_csv': sha256_file('data/raw/Moreno-Mateos.csv'),
        'deephf_xlsx': sha256_file('data/external/Benchmarking-CRISPR-on-tools/'
                                   'Training/DeepHF_training.xlsx'),
    }
    for k, v in hashes.items():
        print(f"    sha256[{k}] = {v}")

    print("\n[3] Contamination audit vs Locked Moreno-Mateos (all candidates)")
    sets = {
        'deephf': set(df_hf['normalized_sequence']),
        'deepspcas9': set(df_dsp['normalized_sequence']),
        'moreno_mateos': set(df_mm['normalized_sequence']),
    }
    contam = contamination_report(sets['moreno_mateos'],
                                  {k: v for k, v in sets.items()
                                   if k != 'moreno_mateos'})
    for cand, r in contam['candidates'].items():
        print(f"    {cand}: n={r['n']} overlap_fwd={r['overlap_fwd']} "
              f"overlap_rc={r['overlap_rc']} eligible={r['eligible']}")
    pairwise = overlap_report(sets)
    for pair, r in pairwise.items():
        print(f"    {pair}: fwd={r['intersection_fwd']} "
              f"rc={r['intersection_rc']} jaccard={r['jaccard']:.4f}")

    print("\n[4] Canonical split reproduction (seed 42, 85/15)")
    train_idx, val_idx = train_test_split(
        np.arange(len(df_dsp)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed'])
    df_train_split = df_dsp.iloc[train_idx].reset_index(drop=True)
    df_val = df_dsp.iloc[val_idx].reset_index(drop=True)
    print(f"    Train {len(df_train_split)} / Validation {len(df_val)} "
          "(matches canonical 8,599 / 1,518)")

    y_train = df_train_split['activity_label'].values
    y_val = df_val['activity_label'].values
    y_test = df_mm['activity_label'].values

    print("\n[5] Feature extraction (tabular + one-hot)")
    feature_extractor = SequenceFeatureExtractor(
        context_length=context_length,
        guide_length=guide_length,
        guide_start=guide_start,
        k_values=[2, 3],
        include_one_hot=False,
        include_gc=True,
        include_composition=True,
        include_kmer=True,
        include_positional=True)
    X_tab_train = feature_extractor.extract_features_batch(
        df_train_split['normalized_sequence'].tolist()).values
    X_oh_train = extract_one_hot_for_cnn(
        df_train_split['normalized_sequence'].tolist(), context_length)
    X_tab_val, X_oh_val = (feature_extractor.extract_features_batch(
        df_val['normalized_sequence'].tolist()).values,
        extract_one_hot_for_cnn(df_val['normalized_sequence'].tolist(),
                                context_length))
    X_tab_test, X_oh_test = (feature_extractor.extract_features_batch(
        df_mm['normalized_sequence'].tolist()).values,
        extract_one_hot_for_cnn(df_mm['normalized_sequence'].tolist(),
                                context_length))
    feat_test = feature_extractor.extract_features_batch(
        df_mm['normalized_sequence'].tolist())
    gc_test = feat_test['gc_full'].values.astype(float)

    # DeepHF tabular features + one-hot (the ONLY added training rows)
    X_tab_hf = feature_extractor.extract_features_batch(
        df_hf['normalized_sequence'].tolist()).values
    X_oh_hf = extract_one_hot_for_cnn(
        df_hf['normalized_sequence'].tolist(), context_length)
    global_feat_count = int(feature_extractor.get_feature_count())
    if X_tab_hf.shape[1] != global_feat_count:
        raise RuntimeError("DeepHF feature width mismatch with canonical extractor")

    print("\n[6] Building diversified training pool (arm D1, provenance-table)")
    pool = build_training_pool(df_train_split, df_hf)
    pool_out = results_dir / f"pool_d1_{timestamp}.csv"
    pool.to_csv(pool_out, index=False)
    psum = pool_summary(pool)
    n_dsp_pool = psum['per_source']['deepspcas9']['n']
    n_hf_pool = psum['per_source']['deephf']['n']
    print(f"    Pool total: {psum['n_total']} = {n_dsp_pool} (DeepSpCas9) + "
          f"{n_hf_pool} (DeepHF)")
    print(f"    within-source duplicates: {psum['duplicate_scan']['within_source_duplicate_rows']}")
    print(f"    cross-source 30-mers duplicated: {psum['duplicate_scan']['cross_source_exact_30mer_rows']}")
    print(f"    shared 20-mer guides cross-source: {psum['duplicate_scan']['shared_guide_cross_source']}")
    print(f"    Pool CSV: {pool_out}")

    combined_X_tab = np.vstack([X_tab_train, X_tab_hf])
    combined_y = np.concatenate([y_train, df_hf['activity_label'].values])

    print("\n[7] Loading canonical models for frozen hyperparameters")
    from src.multidataset.config import CANONICAL_MODEL_FILES
    canonical = {
        'random_forest': RandomForestModel.load_model(
            CANONICAL_MODEL_FILES['random_forest']),
        'xgboost': XGBoostModel.load_model(
            CANONICAL_MODEL_FILES['xgboost']),
        'cnn': CNNModel.load_model(CANONICAL_MODEL_FILES['cnn']),
    }
    hyperparams = {m: canonical[m].get_params() for m in MODELS}
    if len(canonical['random_forest'].feature_names) != global_feat_count:
        raise RuntimeError("RF feature names do not match the tabular "
                           "extractor (pipeline mismatch)")
    print(f"    RF feature count: {global_feat_count}")

    print("\n[8] Training arms B and D1 (canonical hyperparameters, frozen)")
    models = {m: {} for m in MODELS}
    Xoh_full_train = np.concatenate([X_oh_train, X_oh_hf], axis=0)
    for arm in ARMS:
        print(f"    --- arm {arm} ---")
        Xt = combined_X_tab if arm == PRIMARY_ARM else X_tab_train
        yt = combined_y if arm == PRIMARY_ARM else y_train
        for model in MODELS:
            hp = hyperparams[model]
            if model == 'random_forest':
                mdl = RandomForestModel(**hp)
                mdl.fit(Xt, yt,
                        feature_names=canonical['random_forest'].feature_names)
            elif model == 'xgboost':
                mdl = XGBoostModel(**hp)
                mdl.fit(Xt, yt, eval_set=(X_tab_val, y_val))
            else:
                mdl = CNNModel(**hp)
                mdl.fit((Xoh_full_train if arm == PRIMARY_ARM else X_oh_train),
                        yt, X_oh_val, y_val, verbose=False)
            models[model][arm] = mdl
            print(f"      {model} trained "
                  f"(best_epoch={mdl.training_history.get('best_epoch', 'n/a')})")

    print("\n[9] Reproducibility check: arm-B vs canonical on validation split")
    reproducibility = {}
    for model in MODELS:
        Xv = X_oh_val if model == 'cnn' else X_tab_val
        pred_b = models[model]['B'].predict(Xv)
        pred_can = canonical[model].predict(Xv)
        reproducibility[model] = {
            'rmse_b_vs_canonical': float(np.sqrt(np.mean((pred_b - pred_can) ** 2))),
            'pearson_b_vs_canonical': float(np.corrcoef(pred_b, pred_can)[0, 1]),
            'max_abs_diff': float(np.max(np.abs(pred_b - pred_can))),
        }
        r = reproducibility[model]
        print(f"    {model}: RMSE(canonical,B)={r['rmse_b_vs_canonical']:.3e} "
              f"corr={r['pearson_b_vs_canonical']:.8f}")

    print("\n[10] Internal evaluation on validation split (no selection decision)")
    internal = {}
    for model in MODELS:
        internal[model] = {}
        Xv = X_oh_val if model == 'cnn' else X_tab_val
        for arm in ARMS:
            preds = models[model][arm].predict(Xv)
            internal[model][arm] = summarize(y_val, preds)
            mm = internal[model][arm]
            print(f"    {model}/{arm}: MAE={mm['mae']:.4f} R2={mm['r2']:.4f} "
                  f"pearson={mm['pearson']:.4f} pred-SD={mm['prediction_std']:.4f}")

    print("\n[11] LOCKING: protocols frozen. Applying arms to Moreno-Mateos ONCE")
    preds_test = {m: {} for m in MODELS}
    for model in MODELS:
        Xt = X_oh_test if model == 'cnn' else X_tab_test
        for arm in ARMS:
            preds_test[model][arm] = models[model][arm].predict(Xt)

    external = {}
    external_bins = {}
    for model in MODELS:
        external[model] = {}
        external_bins[model] = {}
        for arm in ARMS:
            external[model][arm] = summarize(y_test, preds_test[model][arm])
            external_bins[model][arm] = {
                'terciles': tercile_bins(y_test, preds_test[model][arm]),
                'gc_quintiles': gc_quintile_bins(
                    y_test, preds_test[model][arm], gc_test),
            }
            em = external[model][arm]
            print(f"    {model}/{arm}: MAE={em['mae']:.4f} RMSE={em['rmse']:.4f} "
                  f"R2={em['r2']:.4f} pearson={em['pearson']:.4f} "
                  f"spearman={em['spearman']:.4f} pred-SD={em['prediction_std']:.4f}")

    print("\n[12] Paired statistics (B vs D1, external, per family)")
    ext_paired = {}
    for model in MODELS:
        ext_paired[model] = {
            'absolute_error': paired_report(
                np.abs(y_test - preds_test[model]['B']),
                np.abs(y_test - preds_test[model]['D1']),
                n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED),
            'squared_error': paired_error_tests(
                y_test, preds_test[model]['B'], preds_test[model]['D1'],
                error_type='squared_error'),
        }
        pr = ext_paired[model]['absolute_error']
        p = ext_paired[model]['squared_error']
        print(f"    {model}: d(mean|eB|-|eD|)={pr['mean_pairwise_diff']:+.4f} "
              f"[{pr['bootstrap_ci']['ci_lower']:+.4f}, "
              f"{pr['bootstrap_ci']['ci_upper']:+.4f}] dz={pr['cohens_dz']:+.3f} "
              f"t(se) p={p['t_p_value']:.4f} wilcoxon p={p['wilcoxon_p_value']:.4f}")

    print("\n[13] Per-arm external metric bootstrap CIs")
    ext_bootstrap = {}
    for model in MODELS:
        ext_bootstrap[model] = {}
        for arm in ARMS:
            ext_bootstrap[model][arm] = {
                mm: bootstrap_metric_ci(y_test, preds_test[model][arm],
                                        metric=mm, n_boot=BOOTSTRAP_N,
                                        random_state=BOOTSTRAP_SEED)
                for mm in ['r2', 'pearson', 'spearman', 'mae', 'rmse']}

    print("\n[14] External bin analysis (locking arm D1) + dispersion/scale")
    ext_bins = {}
    for model in MODELS:
        ext_bins[model] = {}
        for arm in ARMS:
            ext_bins[model][arm] = {
                'tercile': tercile_bins(y_test, preds_test[model][arm]),
                'gc_quintile': gc_quintile_bins(
                    y_test, preds_test[model][arm], gc_test)}
    for model in MODELS:
        tb = ext_bins[model]
        b, d = tb['B']['tercile'], tb['D1']['tercile']
        print(f"    {model} tercile MAE low B={b['low']['mae']:.4f} "
              f"D1={d['low']['mae']:.4f} | high B={b['high']['mae']:.4f} "
              f"D1={d['high']['mae']:.4f}")

    print("\n[15] VERDICT (frozen rule applied to locked results)")
    family_deltas = {m: external[m]['D1']['mae'] - external[m]['B']['mae']
                     for m in MODELS}
    primary_paired = ext_paired[PRIMARY_MODEL]['absolute_error']
    primary_d = primary_paired['mean_pairwise_diff']
    ci_low = primary_paired['bootstrap_ci']['ci_lower']
    ci_high = primary_paired['bootstrap_ci']['ci_upper']
    verdict = decide_verdict(primary_d, ci_low, ci_high, family_deltas)
    for m in MODELS:
        print(f"    ΔMAE(D1-B) {m}: {family_deltas[m]:+.4f}")
    print(f"    primary d(CNN) = {primary_d:+.4f} "
          f"CI [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"    VERDICT: {verdict}")
    print(f"    DECISION GATE: {DECISION_GATE_MAP[verdict]}")
    print(f"    (rule: {VERDICT_RULE[verdict]})")

    print("\n[16] Generating figures + saving results")
    figures = {}
    figures['pool_composition'] = plot_pool_composition(
        pool, figures_dir, timestamp)
    figures['external_metrics'] = plot_external_metrics(
        external, figures_dir, timestamp)
    figures['external_delta_mae'] = plot_external_delta_mae(
        external, ext_paired, figures_dir, timestamp)
    figures['pred_vs_true'] = plot_pred_vs_true(
        y_test, preds_test, figures_dir, timestamp)
    figures['external_error_by_bin'] = plot_external_error_by_bin(
        y_test, preds_test, gc_test, figures_dir, timestamp)

    # prediction CSVs (audit trail)
    for model in MODELS:
        pred_df = pd.DataFrame({
            'sequence_30mer': df_mm['normalized_sequence'],
            'true_activity': y_test,
            'predictions_B': preds_test[model]['B'],
            'predictions_D1': preds_test[model]['D1'],
        })
        pred_df.to_csv(results_dir /
                       f"external_predictions_{model}_{timestamp}.csv",
                       index=False)

    experiment_name = f"phase10_multidataset_{timestamp}"
    results = {
        'experiment_name': experiment_name,
        'phase': '10',
        'timestamp': datetime.now().isoformat(),
        'environment': {
            **environment, 'git_head': head,
            'git_status_src_scripts_tests_config': git_status(),
        },
        'input_hashes': hashes,
        'methodology': {
            'hypothesis': (
                'Adding a second real experimental training domain (DeepHF) to '
                'the canonical DeepSpCas9 corpus improves generalisation to the '
                'locked Moreno-Mateos external test relative to the '
                'single-domain canonical recipe.'),
            'arms': ARM_DEFINITIONS,
            'primary_endpoint': PRIMARY_ENDPOINT,
            'secondary_endpoints': ['RMSE', 'R2', 'Pearson', 'Spearman',
                                    'prediction dispersion (pred SD)',
                                    'activity-range bias', 'GC-quintile bias'],
            'validation_policy': (
                'Validation = canonical DeepSpCas9 15% split (1,518). Used '
                'identically by both arms (CNN early stopping / best-epoch). '
                'DeepHF rows enter the training split only. This is a '
                'conservative bias AGAINST the diversified arm and is '
                'documented.'),
            'label_harmonization': 'raw-scale identity (see '
                                   'src/multidataset/config.py)',
            'geometry_transformation': ('DeepHF 21-mer -> canonical 30-mer via '
                                        'constant flanks AAAA/AAA, documented'),
            'data_quality_exclusions': (
                'DeepHF rows with NaN Wt_Efficiency dropped at load '
                '(targetless rows carry no supervision). PAM rows that cannot '
                'be geometrically mapped are dropped. Neither is a design '
                'decision; both are documented data-quality exclusions.'),
            'duplicate_policy': (
                'Within-corpus exact 30-mer duplicates excluded at load. '
                'Cross-corpus same-guide co-occurrences retained as distinct '
                'experiments with provenance; no averaging.'),
            'external_locking': (
                'Moreno-Mateos applied exactly ONCE per arm/model after the '
                'protocol above was fixed. No iterative use; no selection on '
                'external performance.'),
            'effect_threshold': EFFECT_THRESHOLD,
            'contradiction_threshold': CONTRADICTION_THRESHOLD,
            'ci_halfwidth_guard': CI_HALFWIDTH_GUARD,
            'bootstrap': {'n': BOOTSTRAP_N, 'seed': BOOTSTRAP_SEED},
            'hyperparams_origin': 'Read at runtime from canonical model files.',
            'canonical_artifacts': CANONICAL_MODEL_FILES,
        },
        'dataset_inventory': inv.to_dict(orient='records'),
        'contamination_audit': contam,
        'pairwise_overlap': pairwise,
        'data': {'n_deepspcas9': int(len(df_dsp)),
                 'n_moreno_mateos': int(len(df_mm)),
                 'n_deephf_mapped': int(len(df_hf)),
                 'n_train_split': int(len(df_train_split)),
                 'n_validation': int(len(df_val)),
                 'n_pool_d1': psum['n_total'],
                 'pool_per_source': psum['per_source'],
                 'pool_duplicate_scan': psum['duplicate_scan'],
                 },
        'pool_path': str(pool_out),
        'internal_evaluation': internal,
        'reproducibility_check_B_vs_canonical': reproducibility,
        'external_evaluation': external,
        'external_bin_analysis': ext_bins,
        'external_paired_tests': ext_paired,
        'external_bootstrap': ext_bootstrap,
        'family_delta_mae': family_deltas,
        'verdict': verdict,
        'decision_gate': DECISION_GATE_MAP[verdict],
        'verdict_rule': VERDICT_RULE[verdict],
        'figures': figures,
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # save models (gitignored artifacts)
    import pickle
    for model in MODELS:
        for arm in ARMS:
            artifact = {"model_family": model, "arm": arm, "phase": "10",
                        "git_head": head, "timestamp": timestamp,
                        "hash_deephf": hashes['deephf_xlsx']}
            if model == 'random_forest':
                path = f"models/phase10_rf_arm{arm}_{timestamp}.pkl"
                models[model][arm].save_model(path)
            elif model == 'xgboost':
                path = f"models/phase10_xgb_arm{arm}_{timestamp}.pkl"
                models[model][arm].save_model(path)
            else:
                path = f"models/phase10_cnn_arm{arm}_{timestamp}.pt"
                models[model][arm].save_model(path)
            with open(path + '.meta.json', 'w') as f:
                json.dump(artifact, f, indent=2)
            print(f"    saved {path}")

    print(f"\n    Results: {results_path}")
    print("    Figures:")
    for k, v in figures.items():
        print(f"      - {v}")
    print(f"    Pool: {pool_out}")

    print("\n" + "=" * 72)
    print(f"Phase 10 complete - VERDICT: {verdict} | "
          f"DECISION GATE: {DECISION_GATE_MAP[verdict]}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()