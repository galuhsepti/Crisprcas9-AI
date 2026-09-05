#!/usr/bin/env python3
"""
Phase 9C - Domain-Weighted Retraining (Adaptation) Experiment.

Motivation
----------
Phases 9A/9B showed that post-hoc scale calibration learned on internal
DeepSpCas9 data does NOT rescue the locked Moreno-Mateos test (prediction-scale
compression hypothesis refuted). This phase keeps the training-only boundary
strictly intact but changes the *training recipe*: deep-spCas9 samples are
importance-reweighted so that regions that dominate the Moreno-Mateos
distribution (high activity, high GC) carry more weight during fitting.

Army / recipes (ALL fit on DeepSpCas9 TRAINING split only)
------------------------------------------------------------
  B  : uniform weights 1.0 (canonical unweighted recipe, reproducibility arm)
  L  : inverse-density weight over activity label (flatten label distribution)
  G  : inverse-density weight over full-sequence GC content (flatten GC)
  LG : element-wise product of L and G, clipped + renormalised
       [PRE-SPECIFIED PRIMARY ADAPTATION ARM]
Weights: n_bins=20 equal-width histogram over the observed train range,
w_i = 1 / p_hat(z_i), clipped to [0.1, 10.0], normalised to mean 1 (so the
weighted optimisation is on the same scale as the canonical loss).

Locking / external protocol (identical discipline to 9A/9B)
------------------------------------------------------------
- Weights are computed from DeepSpCas9 train data only. Moreno-Mateos never
  enters weight fitting, arm selection, or tuning.
- Internal model selection is on the canonical validation split. The CNN used
  validation for early stopping / best-epoch selection in Phase 5; the CNN
  retrains do the same -> documented, limited reuse (same precedent as 9B).
- External (Moreno-Mateos) is applied exactly ONCE per locked arm/model.
  LG is the pre-registered primary; B/L/G are pre-specified sensitivity arms.
- Internal sanity check: 5-fold out-of-fold retraining within the training
  split for B vs LG (RF/XGB), with LG weights recomputed per fold to avoid any
  histogram leakage into the held-out fold.

Canonical hyperparameters are read AT RUNTIME from the loaded canonical models
so the 9C retrains use exactly the canonical recipes (RF 200x/d20/sqrt,
XGB 100/d6, CNN 64-{5,7,9}-64/dropout 0.3/adam lr 1e-3/batch 32).

Outputs
-------
- results/experiments/phase9c_adaptation_<timestamp>.json
- results/figures/phase9c_*.png
- docs/phase9c_adaptation_report.md (reporting step)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, KFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.data.validation import validate_sequence
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.evaluation import (
    calculate_all_metrics,
    calculate_mae,
    calculate_rmse,
    paired_error_tests,
    bootstrap_metric_ci
)
from src.diagnostics.model_diagnostics import (
    prediction_summary,
    prediction_vs_true_regression
)
from src.adaptation import (
    uniform_weights,
    label_reweighting_weights,
    gc_reweighting_weights,
    combined_reweighting_weights,
    weight_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['random_forest', 'xgboost', 'cnn']
MODEL_LABELS = {'random_forest': 'RandomForest', 'xgboost': 'XGBoost', 'cnn': 'CNN'}
ARMS = ['B', 'L', 'G', 'LG']
ARM_LABELS = {
    'B': 'Baseline (uniform)',
    'L': 'Label-flattened',
    'G': 'GC-flattened',
    'LG': 'Label x GC (primary)',
}
PRIMARY_ARM = 'LG'

CANONICAL_MODEL_FILES = {
    'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl',
    'xgboost': 'models/xgboost_baseline_20260905_002839.pkl',
    'cnn': 'models/cnn_baseline_20260905_011720.pt',
}

N_BINS = 20
CLIP_MIN = 0.1
CLIP_MAX = 10.0

OOF_FOLDS = 5
OOF_SEED = 4242

ARM_COLORS = {'B': '#7f7f7f', 'L': '#1f77b4', 'G': '#2ca02c', 'LG': '#d62728',
              'true': '#000000'}


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


def arm_weights(arm: str, y_train: np.ndarray, gc_train: np.ndarray) -> np.ndarray:
    """Build the sample weights for one arm (train-only inputs)."""
    if arm == 'B':
        return uniform_weights(y_train)
    if arm == 'L':
        return label_reweighting_weights(y_train, n_bins=N_BINS,
                                         clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    if arm == 'G':
        return gc_reweighting_weights(gc_train, n_bins=N_BINS,
                                     clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    if arm == 'LG':
        return combined_reweighting_weights(y_train, gc_train, n_bins=N_BINS,
                                            clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    raise ValueError(f"Unknown arm: {arm}")


def summarize(y_true, y_pred) -> dict:
    """Scalar evaluation summary of predictions against true values."""
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
    """Activity tercile analysis (low / mid / high) on true activity -- the
    same low/mid/high structure used in Phase 9A/9B."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    t = np.quantile(y_true, [1 / 3, 2 / 3])
    masks = {
        'low': y_true <= t[0],
        'mid': (y_true > t[0]) & (y_true < t[1]),
        'high': y_true >= t[1],
    }
    out = {'thresholds': {'low_mid_boundary': float(t[0]), 'mid_high_boundary': float(t[1])}}
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
        }
    return out


def flattening_check(values, weights, n_bins=10) -> dict:
    """How well the weights flatten the covariate's empirical distribution."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    counts, edges = np.histogram(values, bins=n_bins)
    bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    weighted = np.zeros(n_bins)
    np.add.at(weighted, bin_idx, weights)
    weighted = weighted / weighted.sum()
    raw = counts / counts.sum()
    return {
        'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
        'rw_min': float(weighted.min()), 'rw_max': float(weighted.max()),
        'raw_range': float(raw.max() - raw.min()),
        'rw_range': float(weighted.max() - weighted.min()),
        'raw_dist': [float(v) for v in raw],
        'rw_dist': [float(v) for v in weighted],
    }


def save_figure(fig, name: str, figures_dir: Path, timestamp: str) -> str:
    path = figures_dir / f"phase9c_{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------- plotting
def plot_weight_distributions(y_train, gc_train, weights_by_arm, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, arm in enumerate(['L', 'G', 'LG']):
        ax = axes[k]
        if arm == 'L':
            z = y_train; xl = 'activity'
        elif arm == 'G':
            z = gc_train; xl = 'GC'
        else:
            z = y_train; xl = 'activity'
        ax.hist(z, bins=20, alpha=0.5, color=ARM_COLORS['B'], label='raw', density=True)
        w = weights_by_arm[arm]
        weighted_color = {'L': '#1f77b4', 'G': '#ff7f0e', 'LG': ARM_COLORS['LG']}[arm]
        ax.hist(z, bins=20, weights=w / w.sum(),
                alpha=0.5, color=weighted_color,
                label='weighted', density=True)
        ax.set_title(f"{ARM_LABELS[arm]}")
        ax.set_xlabel(xl)
        ax.legend(fontsize=8)
    fig.suptitle('Reweighting effect: empirical density (raw vs weighted)', y=1.02)
    fig.tight_layout()
    return save_figure(fig, 'weight_distributions', figures_dir, timestamp)


def plot_internal_arm_metrics(internal, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for col, metric in enumerate(['r2', 'mae']):
        ax = axes[col]
        x = np.arange(len(MODELS))
        width = 0.2
        for j, arm in enumerate(ARMS):
            vals = [internal[m][arm][metric] for m in MODELS]
            ax.bar(x + (j - 2) * width, vals, width * 0.85, color=ARM_COLORS[arm],
                   label=ARM_LABELS[arm])
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_title(f'Internal validation {metric.upper()}')
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, 'internal_arm_metrics', figures_dir, timestamp)


def plot_external_pred_distributions(y_test, preds_by_arm, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        ax.hist(y_test, bins=30, alpha=0.35, color=ARM_COLORS['true'], label='true')
        for arm in ['B', 'LG']:
            ax.hist(preds_by_arm[model][arm], bins=30, alpha=0.45,
                    color=ARM_COLORS[arm], label=ARM_LABELS[arm].split(' (')[0])
        ax.set_title(f"{MODEL_LABELS[model]} · external")
        ax.set_xlabel('activity'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'external_prediction_distributions', figures_dir, timestamp)


def plot_pred_vs_true(y_test, preds_by_arm, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        for arm in ['B', 'LG']:
            ax.scatter(y_test, preds_by_arm[model][arm], s=7, alpha=0.35,
                       color=ARM_COLORS[arm], label=ARM_LABELS[arm].split(' (')[0])
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{MODEL_LABELS[model]} · external")
        ax.set_xlabel('true'); ax.set_ylabel('prediction'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'pred_vs_true', figures_dir, timestamp)


def plot_external_error_by_bin(y_test, preds_by_arm, gc_test, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for col, (kind, bins_fn) in enumerate([
            ('activity tercile', lambda p: tercile_bins(y_test, p)),
            ('GC quintile', lambda p: gc_quintile_bins(y_test, p, gc_test))]):
        ax = axes[col]
        names = ['low', 'mid', 'high'] if kind.startswith('activity') else ['q1', 'q2', 'q3', 'q4', 'q5']
        x = np.arange(len(names))
        width = 0.25
        for j, arm in enumerate(['B', 'LG']):
            b = bins_fn(preds_by_arm['random_forest'][arm])
            vals = [b[n]['mae'] for n in names]
            ax.bar(x + (j - 1) * width, vals, width * 0.9, color=ARM_COLORS[arm],
                   label=ARM_LABELS[arm].split(' (')[0])
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title(f'RF MAE by {kind} · external')
        ax.set_ylabel('MAE'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'external_error_by_bin', figures_dir, timestamp)


# ------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("Phase 9C - Domain-Weighted Retraining (Adaptation) Experiment")
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

    # ------------------------------------------------------------ data
    print("\n1. Loading and validating data (canonical pipeline)...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid / Moreno-Mateos: {len(df_test)} valid")

    print("\n2. Reproducing the exact canonical split (seed 42, 85/15)...")
    train_idx, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    df_train_split = df_train.iloc[train_idx].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"   Train {len(df_train_split)} / Validation {len(df_val)}")

    y_train = df_train_split['activity'].values
    y_val = df_val['activity'].values
    y_test = df_test['activity'].values

    print("\n3. Loading canonical models (for hyperparameters ONLY)...")
    canonical = {
        'random_forest': RandomForestModel.load_model(CANONICAL_MODEL_FILES['random_forest']),
        'xgboost': XGBoostModel.load_model(CANONICAL_MODEL_FILES['xgboost']),
        'cnn': CNNModel.load_model(CANONICAL_MODEL_FILES['cnn']),
    }
    hyperparams = {m: canonical[m].get_params() for m in MODELS}

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
    if len(canonical['random_forest'].feature_names) != feature_extractor.get_feature_count():
        raise RuntimeError("RF feature names do not match the tabular extractor "
                           "(pipeline mismatch)")

    print("   Extracting tabular + one-hot features...")
    feat_train = feature_extractor.extract_features_batch(
        df_train_split['sequence_30mer'].tolist())
    X_tab_train = feat_train.values
    gc_train = feat_train['gc_full'].values.astype(float)
    X_oh_train = extract_one_hot_for_cnn(df_train_split['sequence_30mer'].tolist(),
                                         context_length)
    X_tab_val, X_oh_val = build_tabular_and_onehot(df_val, feature_extractor)
    X_tab_test, X_oh_test = build_tabular_and_onehot(df_test, feature_extractor)
    feat_test = feature_extractor.extract_features_batch(df_test['sequence_30mer'].tolist())
    gc_test = feat_test['gc_full'].values.astype(float)

    print("\n4. Building TRAIN-only domain weights for each arm...")
    weights_by_arm = {}
    weights_summary = {}
    flattening = {}
    for arm in ARMS:
        weights_by_arm[arm] = arm_weights(arm, y_train, gc_train)
        weights_summary[arm] = weight_summary(weights_by_arm[arm])
        ws = weights_summary[arm]
        print(f"   {arm}: mean={ws['mean']:.3f} min={ws['min']:.3f} "
              f"max={ws['max']:.3f} ESS/n={ws['ess_ratio']:.3f}")
    flattening['L_over_label'] = flattening_check(y_train, weights_by_arm['L'])
    flattening['G_over_gc'] = flattening_check(gc_train, weights_by_arm['G'])
    flattening['LG_over_label'] = flattening_check(y_train, weights_by_arm['LG'])
    flattening['LG_over_gc'] = flattening_check(gc_train, weights_by_arm['LG'])
    print(f"   label density range raw={flattening['L_over_label']['raw_range']:.4f} "
          f"-> weighted={flattening['L_over_label']['rw_range']:.4f}")
    print(f"   GC   density range raw={flattening['G_over_gc']['raw_range']:.4f} "
          f"-> weighted={flattening['G_over_gc']['rw_range']:.4f}")

    print("\n5. Retraining each model under each arm (canonical hyperparameters)...")
    models = {m: {} for m in MODELS}
    for arm in ARMS:
        w = weights_by_arm[arm]
        print(f"   --- arm {arm} ---")
        for model in MODELS:
            hp = hyperparams[model]
            if model == 'random_forest':
                mdl = RandomForestModel(**hp)
                mdl.fit(X_tab_train, y_train,
                        feature_names=canonical['random_forest'].feature_names,
                        sample_weight=None if arm == 'B' else w)
            elif model == 'xgboost':
                mdl = XGBoostModel(**hp)
                mdl.fit(X_tab_train, y_train, eval_set=(X_tab_val, y_val),
                        sample_weight=None if arm == 'B' else w)
            else:
                mdl = CNNModel(**hp)
                mdl.fit(X_oh_train, y_train, X_oh_val, y_val,
                        verbose=False,
                        sample_weight=None if arm == 'B' else w)
            models[model][arm] = mdl
            print(f"      {model} trained (best_epoch={mdl.training_history.get('best_epoch', 'n/a')})")

    print("\n6. Internal evaluation on validation split (selection step)...")
    internal = {}
    for model in MODELS:
        internal[model] = {}
        Xv = X_oh_val if model == 'cnn' else X_tab_val
        for arm in ARMS:
            preds = models[model][arm].predict(Xv)
            internal[model][arm] = summarize(y_val, preds)
            m = internal[model][arm]
            print(f"   {model}/{arm}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"R2={m['r2']:.4f} pearson={m['pearson']:.4f} "
                  f"spearman={m['spearman']:.4f} pred-SD={m['prediction_std']:.4f}")

    print(f"\n7. Pipeline reproducibility check (B arm vs canonical on validation)...")
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
        print(f"   {model}: RMSE(canonical,B)={r['rmse_b_vs_canonical']:.3e} "
              f"corr={r['pearson_b_vs_canonical']:.8f}")

    print(f"\n8. Internal OOF sanity check (B vs LG, {OOF_FOLDS}-fold within train)...")
    oof_in_train = {}
    kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=OOF_SEED)
    for model in ['random_forest', 'xgboost']:
        oof_b = np.zeros(len(df_train_split))
        oof_lg = np.zeros(len(df_train_split))
        fold_weights = {}
        for fold, (fit_idx, hold_idx) in enumerate(kf.split(X_tab_train)):
            sub_w_b = None
            sub_w_lg = arm_weights('LG', y_train[fit_idx], gc_train[fit_idx])
            if model == 'random_forest':
                gb = RandomForestModel(**hyperparams['random_forest'])
                gb.fit(X_tab_train[fit_idx], y_train[fit_idx],
                       feature_names=canonical['random_forest'].feature_names,
                       sample_weight=sub_w_b)
                gl = RandomForestModel(**hyperparams['random_forest'])
                gl.fit(X_tab_train[fit_idx], y_train[fit_idx],
                       feature_names=canonical['random_forest'].feature_names,
                       sample_weight=sub_w_lg)
            else:
                gb = XGBoostModel(**hyperparams['xgboost'])
                gb.fit(X_tab_train[fit_idx], y_train[fit_idx])
                gl = XGBoostModel(**hyperparams['xgboost'])
                gl.fit(X_tab_train[fit_idx], y_train[fit_idx], sample_weight=sub_w_lg)
            oof_b[hold_idx] = gb.predict(X_tab_train[hold_idx])
            oof_lg[hold_idx] = gl.predict(X_tab_train[hold_idx])
        sum_b = summarize(y_train, oof_b)
        sum_lg = summarize(y_train, oof_lg)
        oof_in_train[model] = {
            'B': sum_b,
            'LG': sum_lg,
            'delta_r2_LG_minus_B': float(sum_lg['r2'] - sum_b['r2']),
            'delta_mae_LG_minus_B': float(sum_lg['mae'] - sum_b['mae']),
        }
        print(f"   {model}: OOF R2 B={sum_b['r2']:.4f} LG={sum_lg['r2']:.4f} "
              f"(delta={sum_lg['r2'] - sum_b['r2']:+.4f}) | "
              f"MAE B={sum_b['mae']:.4f} LG={sum_lg['mae']:.4f} "
              f"(delta={sum_lg['mae'] - sum_b['mae']:+.4f})")

    print("\n9. Locking: primary arm = LG. Applying to Moreno-Mateos ONCE...")
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
                'gc_quintiles': gc_quintile_bins(y_test, preds_test[model][arm], gc_test),
            }
            m = external[model][arm]
            print(f"   {model}/{arm}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"R2={m['r2']:.4f} pearson={m['pearson']:.4f} "
                  f"spearman={m['spearman']:.4f} pred-SD={m['prediction_std']:.4f}")

    print("\n10. External analysis for the locked primary vs baseline (B vs LG)...")
    for model in MODELS:
        b = external_bins[model]['B']['terciles']
        lg = external_bins[model]['LG']['terciles']
        print(f"   {model} tercile low bias B={b['low']['bias']:+.4f} "
              f"LG={lg['low']['bias']:+.4f} | high bias B={b['high']['bias']:+.4f} "
              f"LG={lg['high']['bias']:+.4f}")

    print("\n11. Paired tests B vs LG (external, locked primary)...")
    external_paired = {}
    for model in MODELS:
        external_paired[model] = {
            'squared_error': paired_error_tests(
                y_test, preds_test[model]['B'], preds_test[model]['LG'],
                error_type='squared_error'),
            'absolute_error': paired_error_tests(
                y_test, preds_test[model]['B'], preds_test[model]['LG'],
                error_type='absolute_error'),
        }
        p = external_paired[model]['absolute_error']
        print(f"   {model}: mean AE diff (B-LG) = {p['mean_diff']:.4f} "
              f"(t p={p['t_p_value']:.4f})")

    print("\n12. Bootstrap 95% CIs for locked-arm external metrics (B vs LG)...")
    external_bootstrap = {}
    for model in MODELS:
        external_bootstrap[model] = {}
        for arm in ['B', 'LG']:
            external_bootstrap[model][arm] = {
                mm: bootstrap_metric_ci(y_test, preds_test[model][arm],
                                        metric=mm, n_boot=1000, random_state=42)
                for mm in ['r2', 'pearson', 'spearman', 'mae', 'rmse']
            }

    print("\n13. Generating figures...")
    figures = {}
    figures['weight_distributions'] = plot_weight_distributions(
        y_train, gc_train, weights_by_arm, figures_dir, timestamp)
    figures['internal_arm_metrics'] = plot_internal_arm_metrics(
        internal, figures_dir, timestamp)
    figures['external_prediction_distributions'] = plot_external_pred_distributions(
        y_test, preds_test, figures_dir, timestamp)
    figures['pred_vs_true'] = plot_pred_vs_true(y_test, preds_test, figures_dir, timestamp)
    figures['external_error_by_bin'] = plot_external_error_by_bin(
        y_test, preds_test, gc_test, figures_dir, timestamp)

    print("\n14. Saving results...")
    experiment_name = f"phase9c_adaptation_{timestamp}"
    results = {
        'experiment_name': experiment_name,
        'phase': '9c',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'hypothesis': ('Domain-weighted retraining on internal DeepSpCas9 '
                           'data (emphasising high-activity, high-GC regions that '
                           'dominate the Moreno-Mateos distribution) improves '
                           'external test metrics relative to the canonical '
                           'unweighted recipe, without touching external data.'),
            'arms': {
                'B': 'Uniform weights (canonical recipe reproduction).',
                'L': ('Inverse-density weights w=1/p(label), n_bins=20, '
                      'clip [0.1, 10.0], mean-1 normalised.'),
                'G': ('Inverse-density weights w=1/p(gc_full), n_bins=20, '
                      'clip [0.1, 10.0], mean-1 normalised.'),
                'LG': ('Product of L and G weights, clipped + renormalised. '
                       'PRE-SPECIFIED PRIMARY adaptation arm.'),
            },
            'weights_source': ('DeepSpCas9 canonical TRAINING split only '
                               '(85%, seed 42). Moreno-Mateos never enters '
                               'weight fitting / arm selection / tuning.'),
            'internal_selection': ('Validation split (canonical 15%, seed 42). '
                                   'CNN documented reuse of validation for '
                                   'early stopping / best-epoch selection '
                                   '(same precedent as 9A/9B).'),
            'internal_oof_sanity': (f'{OOF_FOLDS}-fold out-of-fold retraining '
                                    'within the training split, B vs LG; LG '
                                    'weights recomputed per fold '
                                    f'(seed={OOF_SEED}).'),
            'external_locking': ('All arms applied to Moreno-Mateos exactly '
                                 'once. LG is pre-registered as primary; B/L/G '
                                 'are pre-specified sensitivity arms. No '
                                 'selection or iteration on external data.'),
            'hyperparams': 'Read at runtime from the loaded canonical models.',
            'canonical_artifacts': CANONICAL_MODEL_FILES,
            'weight_recipe': {'n_bins': N_BINS, 'clip_min': CLIP_MIN,
                              'clip_max': CLIP_MAX, 'normalize_to_mean': 1.0},
            'random_seed': config['split']['random_seed'],
            'data_versions': {
                'primary': f"data/raw/{config['data']['primary_dataset']}",
                'test': f"data/raw/{config['data']['test_dataset']}",
            },
        },
        'data': {'n_train': int(len(df_train_split)),
                 'n_validation': int(len(df_val)),
                 'n_test': int(len(df_test))},
        'weights': {'summary': weights_summary, 'flattening': flattening},
        'internal_evaluation': internal,
        'reproducibility_check_B_vs_canonical': reproducibility,
        'oof_in_train_B_vs_LG': oof_in_train,
        'locked_primary_arm': PRIMARY_ARM,
        'external_evaluation': external,
        'external_bin_analysis': external_bins,
        'external_paired_tests': external_paired,
        'external_bootstrap': external_bootstrap,
        'figures': figures,
    }

    results_path = results_dir / f"{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results: {results_path}")
    print("   Figures:")
    for k, v in figures.items():
        print(f"     - {v}")

    print("\n" + "=" * 72)
    print("Phase 9C complete")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()