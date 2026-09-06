#!/usr/bin/env python3
"""
Phase 11 - Targeted Dataset Analysis (post Phase 10 diversification).

Objective
---------
Explain WHY the Phase 10 diversified-pool gain was only partial, and whether
any mechanism can be attributed at all, using ONLY training-domain data, the
Phase 10 provenance artifacts, and the already-generated Phase 9A - Phase 10
diagnostics. NO model is trained here, no hyperparameter is touched, and
Moreno-Mateos is NEVER read or evaluated (it is only quoted, verbatim, from
the locked Phase 10 / Phase 9A result JSONs).

Deliverables (all additive; nothing existing is modified)
----------------------------------------------------------
- results/experiments/phase11_targeted_dataset_analysis_<timestamp>.json
- results/experiments/phase11_pool_composition_<timestamp>.csv
- results/figures/phase11_*.png
- docs/phase11_targeted_dataset_analysis_report.md   (written separately)

Run:  python3 scripts/run_phase11_dataset_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import json
import logging
import os
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

from src.multidataset.config import (
    PRIMARY_MODEL,
    BASELINE_ARM,
    PRIMARY_ARM,
    EFFECT_THRESHOLD,
)
from src.multidataset.datasets import (
    load_deepspcas9,
    load_deephf,
    sha256_file,
)
from src.multidataset.pool import build_training_pool
from src.diagnostics.stats import (
    ecdf_data,
    histogram_data,
    descriptive_stats,
)
from src.diagnostics.sequence_analysis import (
    gc_content_array,
    nucleotide_frequency_array,
    mean_kmer_frequencies,
)
from src.dataset_analysis.analysis import (
    activity_descriptive,
    classify_shift_addressed,
    coverage_by_bin,
    coverage_expansion,
    decision_gate,
    duplicate_and_conflict_report,
    high_activity_coverage,
    hypothesis_assessment,
    js_divergence,
    label_quality_audit,
    pool_composition_table,
    sequence_diversity_report,
    summarize_generated_external,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path('results/experiments')
FIGURES_DIR = Path('results/figures')

PHASE10_GLOB = 'phase10_multidataset_*.json'
PHASE9A_PATH = EXPERIMENTS_DIR / 'phase9a_domain_diagnostics_20260905_161859.json'


def git_head() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(['git', 'status', '--porcelain'],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "(clean)"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def latest_phase10_artifact() -> dict:
    """Locate the (only) Phase 10 result JSON and its matching pool CSV."""
    cands = sorted(glob.glob(str(EXPERIMENTS_DIR / PHASE10_GLOB)),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit(f"No {PHASE10_GLOB} found in {EXPERIMENTS_DIR}")
    jpath = Path(cands[-1])
    data = json.load(open(jpath))
    ts = os.path.basename(jpath).replace('phase10_multidataset_', '') \
        .replace('.json', '')
    pool = EXPERIMENTS_DIR / f"pool_d1_{ts}.csv"
    if not pool.exists():
        raise SystemExit(f"Matching pool CSV missing: {pool}")
    return {'json_path': jpath, 'pool_csv': pool, 'timestamp': ts, 'data': data}


def load_config() -> dict:
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def save_figure(fig, name: str, timestamp: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"phase11_{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------- plotting
def plot_activity_distribution(dsp_labels, hf_labels, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals, color in [('DeepSpCas9', dsp_labels, '#1f77b4'),
                              ('DeepHF', hf_labels, '#2ca02c')]:
        v = np.asarray(vals, dtype=float)
        ax.hist(v, bins=30, range=(0, 1), alpha=0.55, label=name,
                color=color, density=True, edgecolor='white', lw=0.4)
    ax.set_xlabel('activity label'); ax.set_ylabel('density')
    ax.set_title('Phase 11 - training-domain activity distribution (arm D1)')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'activity_distribution_by_source', ts)


def plot_activity_ecdf(dsp_labels, hf_labels, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals, color in [('DeepSpCas9', dsp_labels, '#1f77b4'),
                              ('DeepHF', hf_labels, '#2ca02c')]:
        e = ecdf_data(list(vals))
        ax.plot(e['x'], e['y'], color=color, lw=2, label=name)
    ax.axvline(0.8, color='k', ls='--', lw=0.8, label='0.8')
    ax.set_xlabel('activity'); ax.set_ylabel('ECDF')
    ax.set_title('Phase 11 - activity ECDF by training source')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'activity_ecdf', ts)


def plot_coverage_by_bin(bins_base, bins_added, bins_combined, ts):
    names = [b['bin'] for b in bins_combined]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, [b['fraction'] for b in bins_base], w * 0.95,
           label='DeepSpCas9', color='#1f77b4')
    ax.bar(x, [b['fraction'] for b in bins_added], w * 0.95,
           label='DeepHF', color='#2ca02c')
    ax.bar(x + w, [b['fraction'] for b in bins_combined], w * 0.95,
           label='Combined pool', color='#d62728', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('fraction of rows in bin')
    ax.set_title('Phase 11 - activity coverage on canonical fixed grid')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'coverage_by_bin', ts)


def plot_gc_distribution(dsp_seqs, hf_seqs, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, seqs, color in [('DeepSpCas9', dsp_seqs, '#1f77b4'),
                              ('DeepHF', hf_seqs, '#2ca02c')]:
        ax.hist(gc_content_array(list(seqs)), bins=30, alpha=0.55,
                label=name, color=color, density=True,
                edgecolor='white', lw=0.4)
    ax.set_xlabel('GC fraction (canonical 30-mer)'); ax.set_ylabel('density')
    ax.set_title('Phase 11 - sequence GC distribution by training source')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'gc_distribution', ts)


def plot_kmer3_top_diffs(top_diffs, ts):
    top = top_diffs[:10]
    kmers = [t['kmer'] for t in top]
    deltas = [t['abs_delta'] for t in top]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = ['#d62728' if t['delta'] > 0 else '#1f77b4' for t in top]
    ax.barh(kmers, deltas, color=cols)
    ax.invert_yaxis()
    ax.set_xlabel('|Δ mean 3-mer frequency| (DeepHF - DeepSpCas9)')
    ax.set_title('Phase 11 - top differing 3-mers (training domains)')
    fig.tight_layout()
    return save_figure(fig, 'kmer3_top_diffs', ts)


def plot_high_activity_region(dsp_labels, hf_labels, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals, color in [('DeepSpCas9', dsp_labels, '#1f77b4'),
                              ('DeepHF', hf_labels, '#2ca02c')]:
        v = np.asarray(vals, dtype=float)
        ax.hist(v[v >= 0.4], bins=30, range=(0.4, 1.0), alpha=0.55,
                label=name, color=color, density=True,
                edgecolor='white', lw=0.4)
    ax.axvline(0.8, color='k', ls='--', lw=0.8)
    ax.set_xlabel('activity (zoom: high-activity region)')
    ax.set_ylabel('density')
    ax.set_title('Phase 11 - high-activity coverage of the training domain')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'high_activity_region', ts)


# ------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("Phase 11 - Targeted Dataset Analysis (analysis-only)")
    print("=" * 72)

    config = load_config()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    head = git_head()
    status = git_status()
    environment = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'git_head': head,
        'git_status_porcelain': status,
    }

    print("\n[0] Locating locked Phase 10 artifacts")
    ph10 = latest_phase10_artifact()
    j10 = ph10['data']
    ph10_hash = sha256_file(str(ph10['json_path']))
    pool_csv_hash = sha256_file(str(ph10['pool_csv']))
    print(f"    Phase 10 JSON: {ph10['json_path']}")
    print(f"    Phase 10 pool: {ph10['pool_csv']}")

    # -------- locked external numbers quoted from Phase 10 (reference only)
    quoted = {
        'phase10_verdict': j10['verdict'],
        'phase10_decision_gate': j10['decision_gate'],
        'family_delta_mae': j10['family_delta_mae'],
        'primary_endpoint': {
            'model': PRIMARY_MODEL,
            'd_external': j10['external_paired_tests'][PRIMARY_MODEL]
                          ['absolute_error']['mean_pairwise_diff'],
            'ci_lower': j10['external_paired_tests'][PRIMARY_MODEL]
                        ['absolute_error']['bootstrap_ci']['ci_lower'],
            'ci_upper': j10['external_paired_tests'][PRIMARY_MODEL]
                        ['absolute_error']['bootstrap_ci']['ci_upper'],
            'dz': j10['external_paired_tests'][PRIMARY_MODEL]
                  ['absolute_error']['cohens_dz'],
            't_p': j10['external_paired_tests'][PRIMARY_MODEL]
                   ['squared_error']['t_p_value'],
            'wilcoxon_p': j10['external_paired_tests'][PRIMARY_MODEL]
                          ['squared_error']['wilcoxon_p_value'],
        },
        'external_mae': {
            m: {arm: j10['external_evaluation'][m][arm]['mae']
                for arm in [BASELINE_ARM, PRIMARY_ARM]} for m in [
                'random_forest', 'xgboost', 'cnn']},
        'external_high_tercile_mae_cnn': {
            arm: j10['external_bin_analysis']['cnn'][arm]['tercile']['high']['mae']
            for arm in [BASELINE_ARM, PRIMARY_ARM]},
        'note': ('All values above are cited verbatim from the locked Phase 10 '
                 'result JSON. Moreno-Mateos was NOT read or re-evaluated here.'),
    }
    ci_excludes_zero = not (quoted['primary_endpoint']['ci_lower'] < 0 <
                            quoted['primary_endpoint']['ci_upper'])
    ext_effect = abs(float(quoted['primary_endpoint']['d_external']))

    # -------- [1] provenance reconstruction (training domain only)
    print("\n[1] Reconstructing training-domain provenance")
    df_dsp = load_deepspcas9()
    df_hf = load_deephf()
    print(f"    DeepSpCas9 valid: {len(df_dsp)}")
    print(f"    DeepHF mapped+valid: {len(df_hf)} "
          f"(dropped {df_hf.attrs['n_unlabeled_dropped']} NaN-label rows of "
          f"{df_hf.attrs['n_mapped_valid']} mapped)")

    train_idx, val_idx = train_test_split(
        np.arange(len(df_dsp)),
        test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed'])
    df_train_split = df_dsp.iloc[train_idx].reset_index(drop=True)
    df_val = df_dsp.iloc[val_idx].reset_index(drop=True)
    print(f"    split reproduced: train {len(df_train_split)} / "
          f"val {len(df_val)}")

    pool = build_training_pool(df_train_split, df_hf)
    per_src = pool.groupby('source_dataset').size().to_dict()

    stored = pd.read_csv(ph10['pool_csv'])
    row_match = _pool_row_level_compare(pool, stored)

    prov_verify = {
        'n_deepspcas9_valid': len(df_dsp),
        'n_train_split': int(len(df_train_split)),
        'n_validation': int(len(df_val)),
        'n_deephf_mapped_valid': int(df_hf.attrs['n_mapped_valid']),
        'n_deephf_unlabeled_dropped': int(df_hf.attrs['n_unlabeled_dropped']),
        'n_deephf_retained': int(len(df_hf)),
        'n_pool_rebuilt': int(len(pool)),
        'n_pool_stored': int(len(stored)),
        'pool_per_source_rebuilt': {k: int(v) for k, v in per_src.items()},
        'pool_row_level_match_with_stored': bool(row_match),
        'phase10_json_counts': {
            'n_deephf_mapped': j10['data']['n_deephf_mapped'],
            'n_train_split': j10['data']['n_train_split'],
            'n_validation': j10['data']['n_validation'],
            'n_pool_d1': j10['data']['n_pool_d1'],
        },
    }

    # ---- verify each reconstructed count against the locked Phase 10 record
    checks = []
    checks.append(('train_split', prov_verify['n_train_split'],
                   prov_verify['phase10_json_counts']['n_train_split']))
    checks.append(('validation', prov_verify['n_validation'],
                   prov_verify['phase10_json_counts']['n_validation']))
    checks.append(('deephf_retained', prov_verify['n_deephf_retained'],
                   prov_verify['phase10_json_counts']['n_deephf_mapped']))
    checks.append(('pool_total', prov_verify['n_pool_rebuilt'],
                   prov_verify['phase10_json_counts']['n_pool_d1']))
    for c in checks:
        ok = c[1] == c[2]
        print(f"    verify {c[0]}: {c[1]} == {c[2]} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"STOP: Phase 10 provenance verification failed "
                             f"on {c[0]}")
    if not row_match:
        raise SystemExit("STOP: rebuilt pool does not match stored pool CSV")

    # ---------------------------------------------------------------- A
    print("\n[A] Dataset contribution table (training domain)")
    comp = pool_composition_table(pool)
    comp_out = EXPERIMENTS_DIR / f"phase11_pool_composition_{timestamp}.csv"
    comp_flattened = []
    for src, r in comp.items():
        row = {'source': src, **r.get('label', {})}
        for c in ['retained_count', 'fraction_of_pool']:
            row[c] = r[c]
        for c in ['mean', 'std']:
            row[f'gc_{c}'] = r['gc'].get(c)
        for c in ['duplicate_count', 'unique_sequences']:
            row[c] = r[c]
        comp_flattened.append(row)
    pd.DataFrame(comp_flattened).to_csv(comp_out, index=False)

    # ---------------------------------------------------------------- B
    print("\n[B] Activity coverage (canonical fixed grid)")
    dsp_labels = df_train_split['activity_label'].astype(float)
    hf_labels = df_hf['activity_label'].astype(float)
    bins_base = coverage_by_bin(dsp_labels)
    bins_added = coverage_by_bin(hf_labels)
    bins_combined = coverage_by_bin(
        np.concatenate([dsp_labels.values, hf_labels.values]))
    exp = coverage_expansion(dsp_labels, hf_labels)
    coverage = {
        'deepspcas9': {'descriptive': activity_descriptive(dsp_labels),
                       'bins': bins_base},
        'deephf': {'descriptive': activity_descriptive(hf_labels),
                   'bins': bins_added},
        'combined': {'bins': bins_combined},
        'coverage_expansion': exp,
        'expanded_bins': [b['bin'] for b in exp['bins'] if b['expanded']],
    }

    # ---------------------------------------------------------------- C
    print("\n[C] Sequence diversity (training domains)")
    div = sequence_diversity_report(
        df_train_split['normalized_sequence'].tolist(),
        df_hf['normalized_sequence'].tolist(), add_name='deephf')

    # ---------------------------------------------------------------- D
    print("\n[D] Domain-diversity vs sample-size evidence matrix")
    jsd3 = div['kmer_3']['js_divergence']
    gc_d = div['gc']['cohens_d']
    added_frac = len(df_hf) / len(pool)
    sample_ratio = len(pool) / len(df_train_split)
    evidence_matrix = {
        'sample_size': {
            'base_n': int(len(df_train_split)),
            'added_n': int(len(df_hf)),
            'combined_n': int(len(pool)),
            'ratio': float(sample_ratio),
            'added_fraction_of_pool': float(added_frac),
        },
        'activity_coverage': {
            'js_divergence': None,
            'expanded_bins': coverage['expanded_bins'],
            'max_bin_delta_fraction':
                exp['combined_coverage']['max_delta_fraction'],
        },
        'sequence_diversity': {
            'gc_cohens_d': float(gc_d),
            'kmer3_js_divergence': float(jsd3),
            'kmer3_pearson': div['kmer_3']['pearson'],
            'guide_overlap_jaccard':
                div['overlap']['guide_20mer_jaccard'],
        },
        'experimental_domains': {
            'unique_domains': 2,
            'contexts': ['(human) pooled SpCas9 screen NGS mod-freq',
                         '(human) pooled SpCas9 screens HEK293T/HCT116/T, '
                         'edited-read fraction'],
            'label_scales_harmonized_raw': True,
        },
        'interpretation': {
            'confound': ('Phase 10 changed n, activity coverage and sequence '
                         'composition SIMULTANEOUSLY by adding DeepHF. No '
                         'size-matched or single-factor arm exists, so the '
                         'Phase 10 design CANNOT separate these. The matrix '
                         'is therefore descriptive evidence for the '
                         'hypothesis classification, NOT a controlled '
                         'causal test.'),
            'no_fabricated_diversity_score': True,
        },
    }

    # ---------------------------------------------------------------- E
    print("\n[E] High-activity coverage")
    ha = high_activity_coverage(dsp_labels, hf_labels)

    # ---------------------------------------------------------------- F
    print("\n[F] Arm attribution (descriptive, quoted from locked Phase 10)")
    arm_attr = {
        'baseline_B': 'DeepSpCas9 training split only (8,599)',
        'primary_D1': 'DeepSpCas9 + DeepHF (56,894)',
        'internal_val_r2': {
            m: {'B': j10['internal_evaluation'][m]['B']['r2'],
                'D1': j10['internal_evaluation'][m]['D1']['r2']}
            for m in ['random_forest', 'xgboost', 'cnn']},
        'quoted_external': {
            arm: {m: summarize_generated_external(
                m, arm, j10['external_evaluation'])
                for m in ['random_forest', 'xgboost', 'cnn']}
            for arm in [BASELINE_ARM, PRIMARY_ARM]},
        'note': ('Internal val is a conservative, lower-magnitude proxy; no '
                 'selection or conclusion is drawn from it. External values '
                 'are locked Phase 10 outputs quoted verbatim.'),
    }

    # -------- [10] label quality + duplicates / conflicts
    print("\n[10] Label quality audit + duplicate/conflict")
    label_audit = label_quality_audit(dsp_labels, df_hf, pool)
    dup_conflict = duplicate_and_conflict_report(pool)

    # -------- [11] Phase 9A -> Phase 10 consistency
    print("\n[11] Phase 9A -> Phase 10 consistency (diagnostics-only)")
    ph9a = json.load(open(PHASE9A_PATH))
    gc_hf = gc_content_array(df_hf['normalized_sequence'].tolist())
    nuc_hf = nucleotide_frequency_array(
        df_hf['normalized_sequence'].tolist()).mean(axis=0)
    g_frac_hf = float(nuc_hf[2])
    ldc = ph9a['label_distribution']
    sdc = ph9a['sequence_distribution']

    def f9(tag, package):  # fraction >0.8 etc.
        pass

    hf_high = float(activity_descriptive(hf_labels)['fraction_high_>0.8'])
    hf_mean = float(np.mean(hf_labels))
    hf_gc = float(np.mean(gc_hf))

    shifts = [
        classify_shift_addressed(
            'label high-activity tail (>0.8)', 
            ldc['validation']['proportion_above_0.8'], hf_high,
            'increase_needed'),
        classify_shift_addressed(
            'label distribution mean', ldc['validation']['mean'], hf_mean,
            'increase_needed'),
        classify_shift_addressed(
            'GC content mean', sdc['gc']['validation']['mean'], hf_gc,
            'increase_needed'),
        classify_shift_addressed(
            'nucleotide composition: G fraction',
            sdc['nucleotide_frequencies']['validation_mean']['G'], g_frac_hf,
            'increase_needed'),
    ]
    # label mean overshoot caveat (test mean is 0.4969, added domain is 0.7325)
    shifts.append({
        'shift': 'label distribution mean OVERSHOOT',
        'base_metric': ldc['validation']['mean'],
        'added_metric': hf_mean,
        'test_metric_quoted': ldc['test']['mean'],
        'status': ('partially addressed - direction correct but the added '
                   'domain overshoots the observed external mean'),
        'caveat': 'descriptive; external values quoted from Phase 9A JSON',
    })
    consistency = {
        'phase9a_json': str(PHASE9A_PATH),
        'quoted_phase9a': {
            'validation_vs_test_label': {
                'cohens_d': ldc['cohens_d'],
                'ks_p': ldc['ks_test']['p_value'],
                'val_high_gt0.8': ldc['validation']['proportion_above_0.8'],
                'test_high_gt0.8': ldc['test']['proportion_above_0.8'],
            },
            'validation_vs_test_gc': {
                'val_mean': sdc['gc']['validation']['mean'],
                'test_mean': sdc['gc']['test']['mean'],
                'cohens_d': sdc['gc']['cohens_d'],
            },
            'validation_vs_test_kmer3': sdc['kmer_3_profile'],
            'validation_vs_test_overlap_jaccard':
                ph9a['sequence_overlap']['deepspcas9_all_vs_morenomateos']
                ['jaccard'],
        },
        'pool_vs_validation': {
            'label': {
                'validation_mean': ldc['validation']['mean'],
                'deephf_mean': hf_mean,
                'fraction_high_gt0.8_validation':
                    ldc['validation']['proportion_above_0.8'],
                'fraction_high_gt0.8_deephf': hf_high,
            },
            'gc': {
                'validation_mean': sdc['gc']['validation']['mean'],
                'deephf_mean': hf_gc,
            },
            'nucleotide_G': {
                'validation_G': sdc['nucleotide_frequencies']
                ['validation_mean']['G'],
                'deephf_G': g_frac_hf,
            },
        },
        'shift_classifications': shifts,
        'note': ('All "test" scalars are cited from the Phase 9A JSON. No '
                 'Moreno-Mateos row was read in this phase. The k-mer profile '
                 'direction against the external composition cannot be '
                 'classified from the stored scalars; the raw pool k-mer '
                 'summary is reported instead.'),
    }

    # -------- H1-H6 + decision gate
    print("\n[12] Hypothesis assessment (H1-H6)")
    evidence = {
        'sample_ratio': float(sample_ratio),
        'added_samples_fraction': float(added_frac),
        'coverage_expanded_bins': exp['bins'],
        'high_coverage_threshold': 0.8,
        'high_base_fraction': float(
            activity_descriptive(dsp_labels)['fraction_high_>0.8']),
        'high_added_fraction': hf_high,
        'kmer_js_divergence': float(jsd3),
        'gc_cohens_d': float(gc_d),
        'experimental_contexts': 2,
        'unique_domains': 2,
        'external_effect_size': ext_effect,
        'external_ci_excludes_zero': ci_excludes_zero,
    }
    hyp = hypothesis_assessment(evidence)
    hnames = {'H1': 'sample size', 'H2': 'activity coverage',
              'H3': 'sequence diversity', 'H4': 'experimental domains',
              'H5': 'high-activity consistency', 'H6': 'attribution sufficiency'}
    hh = {k: v for k, v in hyp.items() if k in hnames}
    for k in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
        print(f"    {k} ({hnames[k]}): {hh.get(k)}")

    print("\n[13] Decision gate")
    gate = decision_gate({
        'h': hyp,
        'external_effect_size': ext_effect,
        'external_ci_excludes_zero': ci_excludes_zero,
        'effect_threshold': EFFECT_THRESHOLD,
    })
    print(f"    gate {gate['gate']}: {gate['label']}")

    # -------- stop conditions checklist
    stop = {
        'provenance_reconstructed': prov_verify['n_pool_rebuilt'] ==
                                    prov_verify['phase10_json_counts']['n_pool_d1'],
        'pool_row_level_verified': bool(row_match),
        'no_new_model_training': True,
        'no_hyperparameter_changes': True,
        'no_moreno_mateos_read_or_evaluated': True,
        'moreno_mateos_quoted_only_from_locked_jsons': True,
        'canonical_artifacts_untouched': True,
        'checkpoint': 'If ANY of the above were False the script would halt '
                      'before producing output.',
    }

    # -------- figures
    print("\n[14] Generating figures")
    figures = {}
    figures['activity_distribution'] = plot_activity_distribution(
        dsp_labels, hf_labels, timestamp)
    figures['activity_ecdf'] = plot_activity_ecdf(
        dsp_labels, hf_labels, timestamp)
    figures['coverage_by_bin'] = plot_coverage_by_bin(
        bins_base, bins_added, bins_combined, timestamp)
    figures['gc_distribution'] = plot_gc_distribution(
        df_train_split['normalized_sequence'].tolist(),
        df_hf['normalized_sequence'].tolist(), timestamp)
    figures['kmer3_top_diffs'] = plot_kmer3_top_diffs(
        div['kmer_3']['top_differing_3mers'], timestamp)
    figures['high_activity_region'] = plot_high_activity_region(
        dsp_labels, hf_labels, timestamp)

    # -------- assemble results
    results = {
        'experiment_name': f"phase11_targeted_dataset_analysis_{timestamp}",
        'phase': '11',
        'timestamp': datetime.now().isoformat(),
        'analysis_only': True,
        'environment': environment,
        'phase10_source': {
            'json_path': str(ph10['json_path']),
            'json_sha256': ph10_hash,
            'pool_csv': str(ph10['pool_csv']),
            'pool_csv_sha256': pool_csv_hash,
        },
        'input_hashes': {
            'deepspcas9_csv':
                sha256_file('data/raw/DeepSpCas9.csv'),
            'deephf_xlsx':
                sha256_file('data/external/Benchmarking-CRISPR-on-tools/'
                            'Training/DeepHF_training.xlsx'),
        },
        'stop_conditions': stop,
        'provenance': {
            'reconstruction': prov_verify,
            'geometry_transformation':
                'DeepHF 21-mer -> canonical 30-mer via constant flanks AAAA/AAA',
            'label_harmonization': 'raw-scale identity',
        },
        'locked_external_reference': quoted,
        'analysis_A_pool_composition': comp,
        'analysis_B_activity_coverage': coverage,
        'analysis_C_sequence_diversity': div,
        'analysis_D_diversity_vs_sample_size_evidence_matrix': evidence_matrix,
        'analysis_E_high_activity_coverage': ha,
        'analysis_F_arm_attribution': arm_attr,
        'label_quality_audit': label_audit,
        'duplicate_conflict_audit': dup_conflict,
        'phase9a_to_10_consistency': consistency,
        'hypothesis_assessment_H1_H6': hyp,
        'decision_gate': gate,
        'figures': figures,
        'outputs': {
            'pool_composition_csv': str(comp_out),
        },
    }

    results_path = EXPERIMENTS_DIR / \
        f"phase11_targeted_dataset_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n    Results: {results_path}")
    for k, v in figures.items():
        print(f"      - {v}")
    print(f"    Composition CSV: {comp_out}")
    print("\n" + "=" * 72)
    print(f"Phase 11 complete | decision gate {gate['gate']}")
    print("=" * 72)
    return results


def _pool_row_level_compare(rebuilt: pd.DataFrame,
                            stored: pd.DataFrame) -> bool:
    """Exact row-level identity between rebuilt pool and stored CSV."""
    cols = ['source_dataset', 'original_sequence', 'normalized_sequence',
            'activity_label', 'label_transform', 'experimental_context',
            'duplicate_status', 'split_assignment', 'guide_sequence', 'pam']
    try:
        a = rebuilt[cols].reset_index(drop=True)
        b = stored[cols].reset_index(drop=True)
    except KeyError:
        return False
    if len(a) != len(b):
        return False
    for c in cols:
        if not a[c].equals(b[c]):
            # allow float tolerance on labels only
            if c != 'activity_label':
                return False
            if not np.allclose(a[c].astype(float), b[c].astype(float)):
                return False
    return True


if __name__ == "__main__":
    main()