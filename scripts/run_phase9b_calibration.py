#!/usr/bin/env python3
"""
Phase 9B - Prediction-Scale / Calibration Hypothesis Test.

Hypothesis under test
---------------------
Part of the external-test degradation (Phase 9A) stems from prediction-scale
compression / under-dispersion: canonical predictors squeeze predictions toward
the DeepSpCas9 activity mean (RF pred-SD ~0.075 vs true ~0.294 on Moreno-Mateos;
XGB ~0.122; CNN ~0.115), with +~0.30 low-activity bias and -~0.33 high-activity
bias. If the hypothesis holds, a post-hoc scale/linear calibration learned ONLY
from internal DeepSpCas9 data should improve scale-sensitive metrics (MAE, RMSE,
R2) on the external test while leaving ranking metrics (Pearson, Spearman)
essentially unchanged, because all tested transforms are monotonic
(rank-preserving).

Method
------
- Canonical artifacts (RF/XGB/CNN) are loaded, never retrained.
- Moreno-Mateos is a LOCKED external test. It is NOT used to fit, select, or
  tune any calibration method, parameter, or multiplier. It is applied exactly
  ONCE as the final evaluation set.
- Calibration parameters are fit only on internal data: the DeepSpCas9
  validation split (canonical 15%, seed 42).
- Internal evaluation is leakage-safe: 5-fold OUT-OF-FOLD application of each
  method within the validation split (fit on K-1 folds, apply to the held-out
  fold).
- Methods: (A) raw (no calibration), (B) linear calibration (OLS),
  (C) variance/scale correction (match mean + SD to the target distribution).
- CNN caveat: the validation split was used for CNN early stopping / best-epoch
  selection in Phase 5. Fitting the CNN's calibration on validation therefore
  reuses that documented signal (limited, scalar-affine reuse). RF and XGBoost
  never used validation during fitting, so their calibration source is fully
  independent. Interpretation leans on the rank-preserving property of the
  transforms.

Outputs
-------
- results/experiments/phase9b_calibration_<timestamp>.json
- results/figures/phase9b_*.png
- docs/phase9b_calibration_report.md (written by the reporting step)
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
from sklearn.model_selection import train_test_split

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
from src.calibration import (
    LinearCalibration,
    VarianceCalibration,
    oof_calibrated_predictions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['random_forest', 'xgboost', 'cnn']
MODEL_LABELS = {'random_forest': 'RandomForest', 'xgboost': 'XGBoost', 'cnn': 'CNN'}
METHODS = ['raw', 'linear', 'variance']
METHOD_LABELS = {'raw': 'Raw', 'linear': 'Linear', 'variance': 'Variance'}

CANONICAL_MODEL_FILES = {
    'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl',
    'xgboost': 'models/xgboost_baseline_20260905_002839.pkl',
    'cnn': 'models/cnn_baseline_20260905_011720.pt',
}

CALIBRATION_OOF_FOLDS = 5
CALIBRATION_OOF_SEED = 4242

COLORS = {'raw': '#1f77b4', 'linear': '#ff7f0e', 'variance': '#2ca02c', 'true': '#7f7f7f'}


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


def calibration_factory(method: str):
    if method == 'linear':
        return lambda: LinearCalibration()
    if method == 'variance':
        return lambda: VarianceCalibration()
    raise ValueError(f"Unsupported calibration method: {method}")


def summarize(y_true, y_pred) -> dict:
    """Scalar evaluation summary of predictions against true values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = calculate_all_metrics(y_true, y_pred)
    s = prediction_summary(y_true, y_pred)
    reg = prediction_vs_true_regression(y_true, y_pred)
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
        'resolution_slope': reg['slope'],
        'resolution_intercept': reg['intercept'],
        'mape': m['mape'],
    }


def tercile_bins(y_true, y_pred) -> dict:
    """
    Activity tercile analysis (low / mid / high) on the true activity --
    the same low/mid/high structure used in Phase 9A.
    """
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


def save_figure(fig, name: str, figures_dir: Path, timestamp: str) -> str:
    path = figures_dir / f"phase9b_{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------- plotting
def plot_prediction_distributions(y_test, preds_by_method, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        ax.hist(y_test, bins=30, alpha=0.35, color=COLORS['true'], label='true')
        for method in METHODS:
            ax.hist(preds_by_method[model][method], bins=30, alpha=0.45,
                    color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(f"{MODEL_LABELS[model]} · external")
        ax.set_xlabel('activity'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'prediction_distributions', figures_dir, timestamp)


def plot_pred_vs_true(y_val, y_test, preds_by_method, method, figures_dir, timestamp):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [('validation', y_val), ('test', y_test)]
    for row, (dataset, y) in enumerate(panels):
        for k, model in enumerate(MODELS):
            ax = axes[row, k]
            ax.scatter(y, preds_by_method[model][dataset], s=8, alpha=0.4, c=COLORS[method])
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{MODEL_LABELS[model]} · {METHOD_LABELS[method]} · {dataset}")
            ax.set_xlabel('true'); ax.set_ylabel('prediction')
    fig.tight_layout()
    return save_figure(fig, f'pred_vs_true_{method}', figures_dir, timestamp)


def plot_calibration_mapping(y_val, preds_val_raw, fitted, figures_dir, timestamp):
    """Internal calibration plot: raw predictions vs true labels with the
    fitted (locked-on-internal) calibration curve overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        p = preds_val_raw[model]
        ax.scatter(p, y_val, s=8, alpha=0.4, color='C0', label='internal data', zorder=2)
        grid = np.linspace(min(0.0, p.min()), max(1.0, p.max()), 200)
        for method in ['linear', 'variance']:
            params = fitted[model][method]
            if method == 'linear':
                curve = params['intercept'] + params['slope'] * grid
                label = f"linear (intercept={params['intercept']:.3f}, slope={params['slope']:.3f})"
            else:
                curve = (params['target_mean']
                         + (grid - params['source_mean'])
                         * (params['target_std'] / params['source_std']))
                label = 'variance/scale'
            ax.plot(grid, curve, color=COLORS[method], lw=2, label=label, zorder=3)
        ax.set_title(f"{MODEL_LABELS[model]} calibration mapping (internal)")
        ax.set_xlabel('raw prediction'); ax.set_ylabel('activity')
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, 'calibration_mapping', figures_dir, timestamp)


def plot_residuals(y_test, preds_by_method, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        for method in METHODS:
            res = preds_by_method[model][method] - y_test
            ax.hist(res, bins=40, alpha=0.45, color=COLORS[method],
                    label=METHOD_LABELS[method])
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f"{MODEL_LABELS[model]} residuals · external")
        ax.set_xlabel('prediction - true'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'residuals_raw_vs_calibrated', figures_dir, timestamp)


def plot_error_by_activity_bin(y_test, preds_by_method, figures_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bin_names = ['low', 'mid', 'high']
    for k, model in enumerate(MODELS):
        ax = axes[k]
        x = np.arange(len(bin_names))
        width = 0.25
        for j, method in enumerate(METHODS):
            bins = tercile_bins(y_test, preds_by_method[model][method])
            values = [bins[b]['mae'] for b in bin_names]
            ax.bar(x + (j - 1) * width, values, width * 0.9, color=COLORS[method],
                   label=METHOD_LABELS[method])
            ax.set_xticks(x); ax.set_xticklabels(bin_names)
        ax.set_title(f"{MODEL_LABELS[model]} MAE by activity tercile · external")
        ax.set_ylabel('MAE'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'error_by_activity_bin', figures_dir, timestamp)


def plot_sd_comparison(y_val, y_test, preds_by_method_val, preds_by_method_test,
                       figures_dir, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for col, (dataset, y, preds) in enumerate([
            ('validation', y_val, preds_by_method_val),
            ('external_test', y_test, preds_by_method_test)]):
        ax = axes[col]
        x = np.arange(len(MODELS))
        width = 0.2
        true_sd = np.std(y, ddof=1)
        ax.bar(x - 2 * width, [true_sd] * len(MODELS),
               width * 0.9, color=COLORS['true'], label='true SD')
        for j, method in enumerate(METHODS):
            vals = [np.std(preds[m][method], ddof=1) for m in MODELS]
            ax.bar(x + (j - 1) * width, vals, width * 0.9, color=COLORS[method],
                   label=f'{METHOD_LABELS[method]} SD')
        ax.axhline(true_sd, color='C3', ls=':', lw=1)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15)
        ax.set_title(f'Prediction SD vs true SD · {dataset}')
        ax.set_ylabel('SD'); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'prediction_sd_comparison', figures_dir, timestamp)


# ------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("Phase 9B - Prediction-Scale / Calibration Hypothesis Test")
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

    print("\n2. Reproducing the exact canonical validation split (seed 42)...")
    _, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"   Validation {len(df_val)} / Train {len(df_train) - len(df_val)}")

    y_val = df_val['activity'].values
    y_test = df_test['activity'].values

    print("\n3. Loading canonical models (no retraining)...")
    models = {
        'random_forest': RandomForestModel.load_model(CANONICAL_MODEL_FILES['random_forest']),
        'xgboost': XGBoostModel.load_model(CANONICAL_MODEL_FILES['xgboost']),
        'cnn': CNNModel.load_model(CANONICAL_MODEL_FILES['cnn']),
    }

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
    if len(models['random_forest'].feature_names) != feature_extractor.get_feature_count():
        raise RuntimeError("RF feature names do not match the tabular extractor "
                           "(pipeline mismatch)")

    X_tab_val, X_oh_val = build_tabular_and_onehot(df_val, feature_extractor)
    X_tab_test, X_oh_test = build_tabular_and_onehot(df_test, feature_extractor)

    print("\n4. Generating raw predictions (validation + external test)...")
    preds_val_raw = {}
    preds_test_raw = {}
    for model in MODELS:
        if model == 'cnn':
            preds_val_raw[model] = models[model].predict(X_oh_val)
            preds_test_raw[model] = models[model].predict(X_oh_test)
        else:
            preds_val_raw[model] = models[model].predict(X_tab_val)
            preds_test_raw[model] = models[model].predict(X_tab_test)
        print(f"   {model}: val sd={np.std(preds_val_raw[model], ddof=1):.4f}, "
              f"test sd={np.std(preds_test_raw[model], ddof=1):.4f}")

    print("\n5. Fitting final calibration parameters on INTERNAL data "
          "(validation split) only...")
    fitted_params = {}
    for model in MODELS:
        fitted_params[model] = {}
        for method in ['linear', 'variance']:
            cal = calibration_factory(method)()
            cal.fit(y_val, preds_val_raw[model])
            fitted_params[model][method] = cal.get_params()
            print(f"   {model}/{method}: {fitted_params[model][method]}")

    print("\n6. Leakage-safe internal evaluation "
          f"({CALIBRATION_OOF_FOLDS}-fold OOF within validation, "
          f"seed={CALIBRATION_OOF_SEED})...")
    internal_oof = {}
    preds_val_oof = {model: {'raw': preds_val_raw[model]} for model in MODELS}
    for model in MODELS:
        for method in ['linear', 'variance']:
            preds_val_oof[model][method] = oof_calibrated_predictions(
                y_val, preds_val_raw[model],
                calibrator_factory=calibration_factory(method),
                n_folds=CALIBRATION_OOF_FOLDS,
                random_state=CALIBRATION_OOF_SEED
            )
    for model in MODELS:
        internal_oof[model] = {}
        for method in METHODS:
            internal_oof[model][method] = summarize(
                y_val, preds_val_oof[model][method])
            m = internal_oof[model][method]
            print(f"   {model}/{method}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"R2={m['r2']:.4f} pearson={m['pearson']:.4f} "
                  f"spearman={m['spearman']:.4f} SD-ratio={m['std_ratio']:.3f}")

    print("\n7. Applying locked calibration ONCE to Moreno-Mateos...")
    preds_test = {model: {'raw': preds_test_raw[model]} for model in MODELS}
    for model in MODELS:
        for method in ['linear', 'variance']:
            params = fitted_params[model][method]
            if method == 'linear':
                preds_test[model][method] = (params['intercept']
                                             + params['slope'] * preds_test_raw[model])
            elif method == 'variance':
                preds_test[model][method] = (params['target_mean']
                                             + (preds_test_raw[model] - params['source_mean'])
                                             * (params['target_std'] / params['source_std']))
    external = {}
    for model in MODELS:
        external[model] = {}
        for method in METHODS:
            external[model][method] = summarize(y_test, preds_test[model][method])
            m = external[model][method]
            print(f"   {model}/{method}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"R2={m['r2']:.4f} pearson={m['pearson']:.4f} "
                  f"spearman={m['spearman']:.4f} SD-ratio={m['std_ratio']:.3f}")

    print("\n8. External activity-tercile error analysis (RAW vs CALIBRATED)...")
    external_bins = {}
    for model in MODELS:
        external_bins[model] = {}
        for method in METHODS:
            external_bins[model][method] = tercile_bins(y_test, preds_test[model][method])
        b = external_bins[model]
        print(f"   {model}: low bias raw={b['raw']['low']['bias']:+.4f} vs "
              f"linear={b['linear']['low']['bias']:+.4f} vs "
              f"variance={b['variance']['low']['bias']:+.4f} | "
              f"high bias raw={b['raw']['high']['bias']:+.4f} vs "
              f"linear={b['linear']['high']['bias']:+.4f} vs "
              f"variance={b['variance']['high']['bias']:+.4f}")

    print("\n9. Paired tests RAW vs CALIBRATED on external (calibration was "
          "locked before seeing external results)...")
    # Summary table of the invariance predictions: Pearson/Spearman must be
    # unchanged by monotonic calibration (up to float rounding).
    external_ranking_check = {}
    for model in MODELS:
        external_ranking_check[model] = {}
        for method in ['linear', 'variance']:
            raw = preds_test[model]['raw']
            cal = preds_test[model][method]
            ext = external[model]
            external_ranking_check[model][method] = {
                'd_pearson': float(np.abs(ext[method]['pearson'] - ext['raw']['pearson'])),
                'd_spearman': float(np.abs(ext[method]['spearman'] - ext['raw']['spearman'])),
                'd_kendall': float(np.abs(ext[method]['kendall'] - ext['raw']['kendall'])),
            }
    paired = {}
    for model in MODELS:
        paired[model] = {}
        for method in ['linear', 'variance']:
            paired[model][method] = {
                'squared_error': paired_error_tests(
                    y_test, preds_test[model]['raw'], preds_test[model][method],
                    error_type='squared_error'),
                'absolute_error': paired_error_tests(
                    y_test, preds_test[model]['raw'], preds_test[model][method],
                    error_type='absolute_error'),
            }
            p = paired[model][method]['squared_error']
            print(f"   {model}/{method}: mean sq-err diff (raw-cal) = "
                  f"{p['mean_diff']:.4e} (t p={p['t_p_value']:.3e})")

    paired['interpretation_note'] = (
        'Calibration method and parameters were locked using internal data '
        'only; the external comparison is a single final evaluation, not a '
        'selection step. Paired tests are descriptive of the locked procedure.'
    )

    print("\n10. Bootstrap 95% CIs for external metrics (raw vs calibrated)...")
    external_bootstrap = {}
    for model in MODELS:
        external_bootstrap[model] = {}
        for method in ['linear', 'variance']:
            external_bootstrap[model][method] = {
                m: bootstrap_metric_ci(y_test, preds_test[model][method],
                                       metric=m, n_boot=1000, random_state=42)
                for m in ['r2', 'pearson', 'spearman', 'mae', 'rmse']
            }

    print("\n11. Generating figures...")
    figures = {}
    figures['prediction_distributions'] = plot_prediction_distributions(
        y_test, preds_test, figures_dir, timestamp)
    preds_for_plot = {
        m: {'validation': preds_val_raw[m], 'test': preds_test[m]['raw']}
        for m in MODELS
    }
    figures['pred_vs_true_raw'] = plot_pred_vs_true(
        y_val, y_test, preds_for_plot, 'raw', figures_dir, timestamp)
    for method in ['linear', 'variance']:
        preds_for_plot = {
            m: {'validation': preds_val_oof[m][method], 'test': preds_test[m][method]}
            for m in MODELS
        }
        figures[f'pred_vs_true_{method}'] = plot_pred_vs_true(
            y_val, y_test, preds_for_plot, method, figures_dir, timestamp)
    figures['calibration_mapping'] = plot_calibration_mapping(
        y_val, preds_val_raw, fitted_params, figures_dir, timestamp)
    figures['residuals'] = plot_residuals(y_test, preds_test, figures_dir, timestamp)
    figures['error_by_activity_bin'] = plot_error_by_activity_bin(
        y_test, preds_test, figures_dir, timestamp)
    figures['prediction_sd_comparison'] = plot_sd_comparison(
        y_val, y_test,
        {m: {'raw': preds_val_raw[m], 'linear': preds_val_oof[m]['linear'],
             'variance': preds_val_oof[m]['variance']} for m in MODELS},
        preds_test, figures_dir, timestamp)

    print("\n12. Saving results...")
    experiment_name = f"phase9b_calibration_{timestamp}"
    results = {
        'experiment_name': experiment_name,
        'phase': '9b',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'hypothesis': ('Prediction-scale compression (under-dispersion) is '
                           'a substantial driver of external-test degradation. '
                           'Post-hoc monotonic calibration learned from internal '
                           'DeepSpCas9 data should improve scale-sensitive '
                           'metrics while leaving ranking metrics unchanged.'),
            'calibration_source': ('DeepSpCas9 validation split (canonical 15%, '
                                   'seed 42). RF/XGB never used validation during '
                                   'fitting -> fully independent source.'),
            'cnn_caveat': ('The CNN used the validation split for early '
                           'stopping / best-epoch selection in Phase 5; fitting '
                           'the CNN calibration on validation is a documented, '
                           'limited (scalar-affine) reuse of that signal.'),
            'internal_evaluation': f'{CALIBRATION_OOF_FOLDS}-fold out-of-fold '
                                   'within the validation split '
                                   f'(seed={CALIBRATION_OOF_SEED})',
            'external_locking': ('Method + parameters locked on internal data; '
                                 'Moreno-Mateos applied exactly once. NEVER used '
                                 'for fitting/selection/tuning.'),
            'canonical_artifacts': CANONICAL_MODEL_FILES,
            'no_retraining': True,
            'no_morenomateos_supply': True,
            'random_seed': config['split']['random_seed'],
            'calibration_oof_folds': CALIBRATION_OOF_FOLDS,
            'calibration_oof_seed': CALIBRATION_OOF_SEED,
            'data_versions': {
                'primary': f"data/raw/{config['data']['primary_dataset']}",
                'test': f"data/raw/{config['data']['test_dataset']}",
            },
            'methods': {
                'raw': 'No calibration (identity).',
                'linear': 'y_cal = intercept + slope * y_pred; OLS fit on internal data.',
                'variance': 'y_cal = target_mean + (y_pred - source_mean) * '
                            '(target_std / source_std); moments from internal data.',
            },
        },
        'data': {'n_validation': int(len(df_val)),
                 'n_train': int(len(df_train) - len(df_val)),
                 'n_test': int(len(df_test))},
        'internal_oof_evaluation': internal_oof,
        'fitted_params': fitted_params,
        'external_evaluation': external,
        'external_ranking_invariance_check': external_ranking_check,
        'external_bin_analysis': external_bins,
        'external_paired_tests': paired,
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
    print("Phase 9B complete")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()