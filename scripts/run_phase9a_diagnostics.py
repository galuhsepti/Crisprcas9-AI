#!/usr/bin/env python3
"""
Phase 9A - Model Diagnostics & Domain-Shift Analysis.

DIAGNOSTIC ONLY. No model improvement, no retraining, no tuning, no data
modification.

Goal: characterize WHY the canonical models (RF, XGBoost, CNN) perform well
in-domain (DeepSpCas9 validation) but degrade on the Moreno-Mateos external
test set, by comparing:

  1. dataset activity-score distributions
  2. sequence-composition distributions (GC, nucleotide frequencies,
     positional frequencies / entropy, 2-mer / 3-mer frequencies)
  3. sequence overlap between datasets
  4. label distribution shift (descriptive + KS + effect size)
  5. prediction distributions of the canonical models on val and test
  6. error analysis by activity / GC / guide-region GC / prediction
     magnitude / absolute-error bins
  7. inter-model prediction agreement

Guardrails:
  - Moreno-Mateos is used ONLY as an external diagnostic set (never for any
    learning decision).
  - No canonical model is retrained; artifacts are loaded as-is.
  - Findings are reported as distribution differences / associations, not
    causal claims.
  - Only a small set of formal tests is performed (KS on labels, KS on GC,
    Welch on GC, k-mer profile correlations) to avoid shotgun testing.

Outputs:
  - results/experiments/phase9a_domain_diagnostics_<timestamp>.json
  - results/figures/phase9a_*.png
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.validation import validate_sequence
from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.diagnostics import (
    descriptive_stats,
    proportion_near_zero,
    proportion_above,
    ks_test,
    welch_ttest,
    cohens_d,
    histogram_data,
    ecdf_data,
    gc_content_array,
    nucleotide_frequency_array,
    mean_positional_frequency_matrix,
    positional_entropy,
    sequence_length_summary,
    ambiguous_character_summary,
    duplicate_summary,
    exact_sequence_overlap,
    top_differing_kmers,
    kmer_profile_correlation,
    prediction_summary,
    error_by_bins,
    model_agreement,
    prediction_vs_true_regression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODELS = ['random_forest', 'xgboost', 'cnn']
MODEL_LABELS = {'random_forest': 'RandomForest', 'xgboost': 'XGBoost', 'cnn': 'CNN'}


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


def build_tabular_and_onehot(df: pd.DataFrame, feature_extractor):
    X_tab = feature_extractor.extract_features_batch(
        df['sequence_30mer'].tolist()).values
    X_oh = extract_one_hot_for_cnn(df['sequence_30mer'].tolist(),
                                   feature_extractor.context_length)
    return X_tab, X_oh


def seq_gc_guide(df: pd.DataFrame) -> np.ndarray:
    """Guide-region GC content from the gc_guide tabular feature name."""
    fe = SequenceFeatureExtractor(
        context_length=30, guide_length=20, guide_start=4, k_values=[2, 3],
        include_one_hot=False, include_gc=True, include_composition=True,
        include_kmer=True, include_positional=True)
    names = fe.get_feature_names()
    idx = names.index('gc_guide')
    return fe.extract_features_batch(df['sequence_30mer'].tolist()).values[:, idx]


def save_figure(fig, name: str, figures_dir: Path) -> str:
    path = figures_dir / f"phase9a_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def plot_activity_distributions(y_val, y_test, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_val, bins=30, alpha=0.6, label='DeepSpCas9 val', color='C0')
    axes[0].hist(y_test, bins=30, alpha=0.6, label='Moreno-Mateos test',
                 color='C1')
    axes[0].set_title('Activity histogram')
    axes[0].set_xlabel('activity'); axes[0].set_ylabel('count')
    axes[0].legend()
    h1 = ecdf_data(y_val); h2 = ecdf_data(y_test)
    axes[1].plot(h1['x'], h1['y'], label='DeepSpCas9 val', color='C0')
    axes[1].plot(h2['x'], h2['y'], label='Moreno-Mateos test', color='C1')
    axes[1].set_title('Activity ECDF')
    axes[1].set_xlabel('activity'); axes[1].set_ylabel('ECDF')
    axes[1].legend()
    fig.tight_layout()
    return save_figure(fig, 'activity_distribution', figures_dir)


def plot_gc_histograms(gc_val, gc_test, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gc_val, bins=30, alpha=0.6, label='DeepSpCas9 val', color='C0')
    ax.hist(gc_test, bins=30, alpha=0.6, label='Moreno-Mateos test', color='C1')
    ax.set_title('GC content distribution')
    ax.set_xlabel('GC fraction'); ax.set_ylabel('count'); ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'gc_distribution', figures_dir)


def plot_nucleotide_frequencies(nt_val, nt_test, figures_dir):
    labels = ['A', 'C', 'G', 'T']
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = np.arange(4)
    ax.bar(x - width / 2, nt_val.mean(axis=0), width, label='DeepSpCas9 val',
           color='C0')
    ax.bar(x + width / 2, nt_test.mean(axis=0), width, label='Moreno-Mateos test',
           color='C1')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('Mean nucleotide frequency'); ax.set_ylabel('frequency')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'nucleotide_frequency', figures_dir)


def plot_positional_frequency(pfm_val, pfm_test, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, pfm, title in [(axes[0], pfm_val, 'DeepSpCas9 val'),
                           (axes[1], pfm_test, 'Moreno-Mateos test')]:
        im = ax.imshow(pfm, aspect='auto', origin='lower', vmin=0, vmax=0.5)
        ax.set_title(title); ax.set_xlabel('position'); ax.set_ylabel('nucleotide')
        ax.set_yticks(range(4)); ax.set_yticklabels(['A', 'C', 'G', 'T'])
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    return save_figure(fig, 'positional_frequency', figures_dir)


def plot_pred_vs_true(y_val, y_test, preds, figures_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for (dataset, y), col in zip([('validation', y_val), ('test', y_test)],
                                 range(2)):
        for k, model in enumerate(MODELS):
            ax = axes[col, k]
            ax.scatter(y, preds[model][dataset], s=8, alpha=0.4)
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', lw=0.8)
            ax.set_xlim(lims); ax.set_ylim(lims)
            r, _ = spearmanr(y, preds[model][dataset])
            ax.set_title(f"{MODEL_LABELS[model]} · {dataset}  (rho={r:.3f})")
            ax.set_xlabel('true'); ax.set_ylabel('prediction')
    fig.tight_layout()
    return save_figure(fig, 'pred_vs_true', figures_dir)


def plot_residuals(y_val, y_test, preds, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        rv = preds[model]['validation'] - y_val
        rt = preds[model]['test'] - y_test
        ax.hist(rv, bins=30, alpha=0.6, label='val', color='C0')
        ax.hist(rt, bins=30, alpha=0.6, label='test', color='C1')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f"{MODEL_LABELS[model]} residuals")
        ax.set_xlabel('prediction - true'); ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'residuals', figures_dir)


def plot_abs_error_vs_true(y_val, y_test, preds, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, model in enumerate(MODELS):
        ax = axes[k]
        ev = np.abs(preds[model]['validation'] - y_val)
        et = np.abs(preds[model]['test'] - y_test)
        ax.scatter(y_val, ev, s=6, alpha=0.4, label='val', color='C0')
        ax.scatter(y_test, et, s=6, alpha=0.4, label='test', color='C1')
        ax.set_title(f"{MODEL_LABELS[model]} |error| vs true")
        ax.set_xlabel('true activity'); ax.set_ylabel('|error|'); ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'abs_error_vs_true', figures_dir)


def plot_model_agreement(preds, figures_dir):
    import itertools
    pairs = list(itertools.combinations(MODELS, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4))
    for k, (a, b) in enumerate(pairs):
        ax = axes[k]
        pa = preds[a]['test']; pb = preds[b]['test']
        ax.scatter(pa, pb, s=8, alpha=0.4)
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', lw=0.8)
        r, _ = spearmanr(pa, pb)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]} (rho={r:.3f})")
        ax.set_xlabel(MODEL_LABELS[a]); ax.set_ylabel(MODEL_LABELS[b])
    fig.tight_layout()
    return save_figure(fig, 'model_agreement', figures_dir)


def main():
    print("=" * 70)
    print("Phase 9A - Model Diagnostics & Domain-Shift Analysis")
    print("=" * 70)

    config = load_config()
    context_length = config['data']['context_length']
    results_dir = Path(config['experiments']['output_dir'])
    figures_dir = Path(config['experiments']['figures_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    print("\n1. Loading and validating data (canonical pipeline)...")
    df_train = load_and_validate_data(f"data/raw/{config['data']['primary_dataset']}")
    df_test = load_and_validate_data(f"data/raw/{config['data']['test_dataset']}")
    print(f"   DeepSpCas9: {len(df_train)} valid / "
          f"Moreno-Mateos: {len(df_test)} valid")

    print("\n2. Reproducing the exact validation split (seed 42)...")
    _, val_idx = train_test_split(
        np.arange(len(df_train)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed']
    )
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"   Validation {len(df_val)} / Train {len(df_train) - len(df_val)}")

    y_val = df_val['activity'].values
    y_test = df_test['activity'].values
    seq_val = df_val['sequence_30mer'].tolist()
    seq_test = df_test['sequence_30mer'].tolist()
    gc_val = gc_content_array(seq_val)
    gc_test = gc_content_array(seq_test)

    print("\n3. Loading canonical models...")
    models = {
        'random_forest': RandomForestModel.load_model(
            'models/rf_baseline_fixed_20260905_001107.pkl'),
        'xgboost': XGBoostModel.load_model(
            'models/xgboost_baseline_20260905_002839.pkl'),
        'cnn': CNNModel.load_model('models/cnn_baseline_20260905_011720.pt'),
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
        raise RuntimeError("RF feature names do not match the tabular extractor "
                           "(pipeline mismatch)")

    X_tab_val, X_oh_val = build_tabular_and_onehot(df_val, feature_extractor)
    X_tab_test, X_oh_test = build_tabular_and_onehot(df_test, feature_extractor)
    gc_guide_test = seq_gc_guide(df_test)

    # -------------------------------------------------- dataset distribution
    print("\n4. Activity-score distribution comparison...")
    label_stats = {
        'validation': descriptive_stats(y_val),
        'test': descriptive_stats(y_test),
    }
    label_stats['validation']['proportion_near_zero_<=0.05'] = \
        proportion_near_zero(y_val, 0.05)
    label_stats['validation']['proportion_above_0.8'] = proportion_above(y_val, 0.8)
    label_stats['test']['proportion_near_zero_<=0.05'] = \
        proportion_near_zero(y_test, 0.05)
    label_stats['test']['proportion_above_0.8'] = proportion_above(y_test, 0.8)
    label_stats['ks_test'] = ks_test(y_val, y_test)
    label_stats['welch_ttest'] = welch_ttest(y_val, y_test)
    label_stats['cohens_d'] = cohens_d(y_val, y_test)
    label_stats['validation_histogram'] = histogram_data(y_val, bins=30)
    label_stats['test_histogram'] = histogram_data(y_test, bins=30)
    label_stats['validation_ecdf'] = ecdf_data(y_val)
    label_stats['test_ecdf'] = ecdf_data(y_test)
    print(f"   val mean={label_stats['validation']['mean']:.4f} "
          f"sd={label_stats['validation']['std']:.4f} | "
          f"test mean={label_stats['test']['mean']:.4f} "
          f"sd={label_stats['test']['std']:.4f}")
    print(f"   KS D={label_stats['ks_test']['statistic']:.4f} "
          f"p={label_stats['ks_test']['p_value']:.3e} "
          f"| Cohen's d={label_stats['cohens_d']:.4f}")

    # --------------------------------------------- sequence distribution
    print("\n5. Sequence-composition distribution comparison...")
    nt_val = nucleotide_frequency_array(seq_val)
    nt_test = nucleotide_frequency_array(seq_test)
    pfm_val = mean_positional_frequency_matrix(seq_val, context_length)
    pfm_test = mean_positional_frequency_matrix(seq_test, context_length)
    entropy_val = positional_entropy(pfm_val)
    entropy_test = positional_entropy(pfm_test)

    seq_stats = {
        'length_validation': sequence_length_summary(seq_val),
        'length_test': sequence_length_summary(seq_test),
        'ambiguous_validation': ambiguous_character_summary(seq_val),
        'ambiguous_test': ambiguous_character_summary(seq_test),
        'duplicates_validation': duplicate_summary(seq_val),
        'duplicates_test': duplicate_summary(seq_test),
        'gc': {
            'validation': descriptive_stats(gc_val),
            'test': descriptive_stats(gc_test),
            'ks_test': ks_test(gc_val, gc_test),
            'welch_ttest': welch_ttest(gc_val, gc_test),
            'cohens_d': cohens_d(gc_val, gc_test),
            'validation_histogram': histogram_data(gc_val, bins=30),
            'test_histogram': histogram_data(gc_test, bins=30),
        },
        'nucleotide_frequencies': {
            'validation_mean': {'A': float(nt_val[:, 0].mean()),
                                'C': float(nt_val[:, 1].mean()),
                                'G': float(nt_val[:, 2].mean()),
                                'T': float(nt_val[:, 3].mean())},
            'test_mean': {'A': float(nt_test[:, 0].mean()),
                          'C': float(nt_test[:, 1].mean()),
                          'G': float(nt_test[:, 2].mean()),
                          'T': float(nt_test[:, 3].mean())},
            'abs_difference': {'A': float(np.abs(nt_val[:, 0].mean() - nt_test[:, 0].mean())),
                               'C': float(np.abs(nt_val[:, 1].mean() - nt_test[:, 1].mean())),
                               'G': float(np.abs(nt_val[:, 2].mean() - nt_test[:, 2].mean())),
                               'T': float(np.abs(nt_val[:, 3].mean() - nt_test[:, 3].mean()))}
        },
        'positional_frequency_validation': pfm_val.tolist(),
        'positional_frequency_test': pfm_test.tolist(),
        'positional_entropy_validation': entropy_val.tolist(),
        'positional_entropy_test': entropy_test.tolist(),
        'kmer_2_profile': kmer_profile_correlation(seq_val, seq_test, k=2),
        'kmer_3_profile': kmer_profile_correlation(seq_val, seq_test, k=3),
        'top_differing_3mers': top_differing_kmers(seq_val, seq_test, k=3, top_k=10),
    }
    print(f"   GC val mean={seq_stats['gc']['validation']['mean']:.4f} vs "
          f"test mean={seq_stats['gc']['test']['mean']:.4f} "
          f"(d={seq_stats['gc']['cohens_d']:.4f})")
    print(f"   kmer2 profile r={seq_stats['kmer_2_profile']['pearson_across_kmers']:.4f} | "
          f"kmer3 r={seq_stats['kmer_3_profile']['pearson_across_kmers']:.4f}")

    # ------------------------------------------------- distance similarity
    print("\n6. Exact-sequence overlap between datasets...")
    overlap = exact_sequence_overlap(seq_val, seq_test)
    overlap['deepspcas9_all_vs_morenomateos'] = exact_sequence_overlap(
        df_train['sequence_30mer'].tolist(), df_test['sequence_30mer'].tolist())
    print(f"   shared={overlap['shared_unique']}, "
          f"jaccard={overlap['jaccard']:.4f} "
          f"(all-of-DeepSpCas9: shared={overlap['deepspcas9_all_vs_morenomateos']['shared_unique']})")

    # -------------------------------------------------------- predictions
    print("\n7. Generating predictions (canonical models, no retraining)...")
    preds = {}
    for model in MODELS:
        if model == 'cnn':
            preds[model] = {'validation': models[model].predict(X_oh_val),
                            'test': models[model].predict(X_oh_test)}
        else:
            preds[model] = {'validation': models[model].predict(X_tab_val),
                            'test': models[model].predict(X_tab_test)}
        r = spearmanr(y_test, preds[model]['test'])
        print(f"   {model}: test spearman={r.statistic:.4f}")

    pred_stats = {model: {'validation': prediction_summary(y_val, preds[model]['validation']),
                          'test': prediction_summary(y_test, preds[model]['test'])}
                  for model in MODELS}

    # -------------------------------------------------------- error analysis
    print("\n8. Error analysis on the external test set...")
    error_analysis = {}
    for model in MODELS:
        p = preds[model]['test']
        error_analysis[model] = {
            'by_activity_bin': error_by_bins(y_test, p, y_test, n_bins=5),
            'by_gc_bin': error_by_bins(y_test, p, gc_test, n_bins=5),
            'by_guide_gc_bin': error_by_bins(y_test, p, gc_guide_test, n_bins=5),
            'by_prediction_bin': error_by_bins(y_test, p, p, n_bins=5),
            'by_abs_error_bin': error_by_bins(y_test, p,
                                              np.abs(p - y_test), n_bins=5),
            'pred_vs_true_regression': prediction_vs_true_regression(y_test, p),
        }
        # Systematic pattern: bias per activity tercile of true activity.
        terciles = np.quantile(y_test, [1 / 3, 2 / 3])
        err = prediction_summary(y_test, p)
        error_analysis[model]['bias_flag'] = {
            'overall_bias': err['bias_mean_pred_minus_true'],
            'negative_bias_underpredict_low_true': float(
                np.mean(p[y_test <= terciles[0]] -
                        y_test[y_test <= terciles[0]])),
            'bias_mid': float(np.mean(p[(y_test > terciles[0]) &
                                        (y_test < terciles[1])] -
                                      y_test[(y_test > terciles[0]) &
                                             (y_test < terciles[1])])),
            'bias_high_true': float(np.mean(p[y_test >= terciles[1]] -
                                            y_test[y_test >= terciles[1]])),
        }
        print(f"   {model}: bias={err['bias_mean_pred_minus_true']:+.4f}")

    # ------------------------------------------------------- model agreement
    print("\n9. Model agreement on the external test set...")
    import itertools
    agreement = {}
    for a, b in itertools.combinations(MODELS, 2):
        agreement[f"{a}_vs_{b}"] = {
            'test': model_agreement(preds[a]['test'], preds[b]['test']),
            'validation': model_agreement(preds[a]['validation'],
                                          preds[b]['validation'])
        }
        print(f"   {a} vs {b}: test spearman={agreement[f'{a}_vs_{b}']['test']['spearman']:.4f}")

    # ---------------------------------------------------------- plots
    print("\n10. Generating diagnostic figures...")
    figures = {
        'activity_distribution': plot_activity_distributions(y_val, y_test, figures_dir),
        'gc_distribution': plot_gc_histograms(gc_val, gc_test, figures_dir),
        'nucleotide_frequency': plot_nucleotide_frequencies(nt_val, nt_test, figures_dir),
        'positional_frequency': plot_positional_frequency(pfm_val, pfm_test, figures_dir),
        'pred_vs_true': plot_pred_vs_true(y_val, y_test, preds, figures_dir),
        'residuals': plot_residuals(y_val, y_test, preds, figures_dir),
        'abs_error_vs_true': plot_abs_error_vs_true(y_val, y_test, preds, figures_dir),
        'model_agreement': plot_model_agreement(preds, figures_dir),
    }

    print("\n11. Saving results...")
    experiment_name = f"phase9a_domain_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = {
        'experiment_name': experiment_name,
        'phase': '9a',
        'timestamp': datetime.now().isoformat(),
        'diagnostic_only': True,
        'methodology': {
            'canonical_artifacts': {
                'random_forest': 'models/rf_baseline_fixed_20260905_001107.pkl',
                'xgboost': 'models/xgboost_baseline_20260905_002839.pkl',
                'cnn': 'models/cnn_baseline_20260905_011720.pt',
                'seed': config['split']['random_seed'],
                'val_ratio': config['split']['val_ratio'],
            },
            'geometry': {'five_prime_context': '[0:4]',
                         'guide': '[4:24]',
                         'pam': '[24:27]',
                         'three_prime_context': '[27:30]'},
            'no_retraining': True,
            'no_data_modification': True,
            'multiple_testing_note': ('Formal tests limited to: KS labels, '
                                      'KS GC, Welch GC, k-mer profile '
                                      'correlations. Large n makes small '
                                      'differences significant; effect sizes '
                                      'are reported alongside p-values.'),
            'causality_note': ('Differences are reported as distribution '
                               'differences / associations, not causal '
                               'explanations.'),
        },
        'data': {'n_validation': int(len(df_val)),
                 'n_train': int(len(df_train) - len(df_val)),
                 'n_test': int(len(df_test))},
        'label_distribution': label_stats,
        'sequence_distribution': seq_stats,
        'sequence_overlap': overlap,
        'predictions': pred_stats,
        'error_analysis': error_analysis,
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

    print("\n" + "=" * 70)
    print("Phase 9A complete (diagnostic only)")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()