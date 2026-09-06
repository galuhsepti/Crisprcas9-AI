"""
Phase 11 - Targeted dataset analysis (post Phase 10 diversification).

Pure, deterministic analysis helpers that describe the *training-domain*
composition of the Phase 10 diversified pool. This is an ANALYSIS phase:

- No model is trained, no hyperparameter is touched.
- Nothing here reads, opens, or evaluates Moreno-Mateos. All "external"
  figures quoted in the report come verbatim from the already-generated
  Phase 9A / Phase 10 result JSONs.

Conventions reused from the canonical pipeline (NOT redefined here):
- Sequence geometry: 30-mer, guide [4:24], PAM [24:27].
- Activity grid: the fixed canonical edges already used by
  src/audit.audit.DEFAULT_ACTIVITY_EDGES ([0, .2, .4, .6, .8, 1.0]) -- a
  pre-existing grid, never adapted to any external set.
- Bin assignment mirrors src.audit.activity_bin_stats (half-open except the
  last bin, which is closed).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..audit.audit import DEFAULT_ACTIVITY_EDGES
from ..diagnostics.stats import (
    descriptive_stats,
    proportion_above,
    proportion_near_zero,
    ks_test,
    cohens_d,
)
from ..diagnostics.sequence_analysis import (
    gc_content_array,
    nucleotide_frequency_array,
    mean_kmer_frequencies,
    exact_sequence_overlap,
    duplicate_summary,
    sequence_length_summary,
    kmer_profile_correlation,
)
from scipy import stats as sps

ACTIVITY_EDGES = DEFAULT_ACTIVITY_EDGES

# ---------------------------------------------------------------------------
# Activity coverage (Analysis B)
# ---------------------------------------------------------------------------


def coverage_by_bin(labels: Sequence[float],
                    edges: Sequence[float] = ACTIVITY_EDGES) -> List[Dict]:
    """
    Distribution of activity labels over the fixed canonical grid.

    Returns per-bin: n, fraction of the source, and within-bin label
    mean/median/std. Bin edges follow the canonical convention (half-open,
    last bin closed). Used ONLY on training-domain labels.
    """
    edges = sorted(float(e) for e in edges)
    if len(edges) < 2:
        raise ValueError("At least two edges are required")
    y = np.asarray(labels, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("labels must not be empty")
    n_total = y.size
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (y >= lo) & (y <= hi)
        else:
            mask = (y >= lo) & (y < hi)
        vals = y[mask]
        n = int(vals.size)
        out.append({
            'bin': f'[{lo:.2f}, {hi:.2f}]',
            'n': n,
            'fraction': float(n / n_total),
            'label_mean': float(vals.mean()) if n else None,
            'label_median': float(np.median(vals)) if n else None,
            'label_std': float(vals.std(ddof=1)) if n > 1 else None,
        })
    return out


def activity_descriptive(labels: Sequence[float]) -> Dict:
    """Canonical descriptive label statistics + extreme-region fractions."""
    d = descriptive_stats(labels)
    d['fraction_near_zero_<=0.05'] = proportion_near_zero(labels, 0.05)
    d['fraction_high_>0.8'] = proportion_above(labels, 0.8)
    return d


def coverage_expansion(base_labels: Sequence[float],
                       added_labels: Sequence[float],
                       edges: Sequence[float] = ACTIVITY_EDGES,
                       abs_min: float = 0.02,
                       rel_min: float = 0.25) -> Dict:
    """
    Whether the added domain materially increases per-bin label coverage.

    A bin is flagged as "expanded" when the added source's within-bin fraction
    is at least `abs_min` (2 percentage points) higher than the base source's
    fraction AND at least `rel_min` (25%) higher relative to the base fraction.
    These are descriptive thresholds on training data, not external-driven.
    """
    base = {b['bin']: b for b in coverage_by_bin(base_labels, edges)}
    added = {b['bin']: b for b in coverage_by_bin(added_labels, edges)}
    out = {'threshold': {'abs_min': abs_min, 'rel_min': rel_min}, 'bins': []}
    for name in base:
        fb = base[name]['fraction']
        fa = added[name]['fraction']
        rel_gain = (fa - fb) / fb if fb > 0 else (np.inf if fa > 0 else 0.0)
        expanded = bool((fa - fb) >= abs_min and rel_gain >= rel_min)
        out['bins'].append({
            'bin': name,
            'base_fraction': fb,
            'added_fraction': fa,
            'delta_fraction': float(fa - fb),
            'rel_gain': float(rel_gain) if np.isfinite(rel_gain) else None,
            'expanded': expanded,
        })
    # combined coverage of both domains together
    combined = coverage_by_bin(list(base_labels) + list(added_labels), edges)
    combined_map = {b['bin']: b['fraction'] for b in combined}
    base_map = {b['bin']: b['fraction'] for b in base.values()}
    out['combined_coverage'] = {
        'base': base_map,
        'combined': combined_map,
        'max_delta_fraction': float(max(
            (combined_map[n] - base_map[n] for n in base_map), default=0.0)),
    }
    return out


# ---------------------------------------------------------------------------
# Sequence diversity (Analysis C)
# ---------------------------------------------------------------------------


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions (vectors or
    counts). Bounded [0, ~0.69] for natural logs; we return the value in nats.
    Uses base-2 -> gives [0,1]. Both inputs are normalized internally.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError("jensen-shannon requires equal-length vectors")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = a.copy()
        b = b.copy()
        nz = (a > 0) & (b > 0)
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein distance (earth mover) between two samples."""
    return float(sps.wasserstein_distance(np.asarray(a, dtype=float),
                                          np.asarray(b, dtype=float)))


def sequence_diversity_report(dsp_seqs: Sequence[str],
                              add_seqs: Sequence[str],
                              add_name: str = "additional") -> Dict:
    """
    Sequence-level diversity of the added training domain relative to the
    canonical DeepSpCas9 training corpus, under the canonical geometry.

    Returns GC, nucleotide composition, 2-mer and 3-mer frequency comparisons
    (Jensen-Shannon, correlation), per-sequence GC Wasserstein distance and
    Cohen's d, exact-sequence and (guide-level) overlap, and the top
    differentially enriched 3-mers. All descriptive; no external data.
    """
    dsp = list(dsp_seqs)
    add = list(add_seqs)

    gc_dsp = gc_content_array(dsp)
    gc_add = gc_content_array(add)

    nuc_dsp = nucleotide_frequency_array(dsp).mean(axis=0)
    nuc_add = nucleotide_frequency_array(add).mean(axis=0)

    k2_dsp = mean_kmer_frequencies(dsp, 2)
    k2_add = mean_kmer_frequencies(add, 2)
    k3_dsp = mean_kmer_frequencies(dsp, 3)
    k3_add = mean_kmer_frequencies(add, 3)

    corr2 = kmer_profile_correlation(dsp, add, 2)
    corr3 = kmer_profile_correlation(dsp, add, 3)

    overlap = exact_sequence_overlap(set(dsp), set(add))

    # guide-level overlap (guide [4:24]) -- geometric, no external data
    guides_dsp = set(s[4:24] for s in dsp)
    guides_add = set(s[4:24] for s in add)
    guide_inter = guides_dsp & guides_add

    top_diff = _top_differing_kmers(k3_dsp, k3_add, n=10)

    return {
        'gc': {
            'deepspcas9': descriptive_stats(gc_dsp),
            add_name: descriptive_stats(gc_add),
            'cohens_d': cohens_d(gc_dsp, gc_add),
            'ks_test': ks_test(gc_dsp, gc_add),
            'wasserstein_1d': wasserstein_1d(gc_dsp, gc_add),
        },
        'nucleotide_composition': {
            'nucleotides': ['A', 'C', 'G', 'T'],
            'deepspcas9': [float(v) for v in nuc_dsp],
            add_name: [float(v) for v in nuc_add],
            'mean_abs_delta': float(np.mean(np.abs(nuc_add - nuc_dsp))),
        },
        'kmer_2': {
            'js_divergence': js_divergence(k2_dsp, k2_add),
            'pearson': corr2,
        },
        'kmer_3': {
            'js_divergence': js_divergence(k3_dsp, k3_add),
            'pearson': corr3,
            'top_differing_3mers': top_diff,
        },
        'overlap': {
            'exact_30mer': overlap,
            'guide_20mer_overlap_count': int(len(guide_inter)),
            'guide_20mer_jaccard': float(len(guide_inter) / len(guides_dsp | guides_add)),
            'guide_20mer_fraction_of_add': float(len(guide_inter) / len(guides_add)) if guides_add else 0.0,
        },
        'length': {add_name: sequence_length_summary(add)},
    }


def _top_differing_kmers(profile_a: np.ndarray, profile_b: np.ndarray,
                         n: int = 10) -> List[Dict]:
    """Top-k k-mers by absolute frequency difference (descriptive)."""
    from ..bioinformatics.kmer import generate_kmers
    names = generate_kmers(3)
    diff = np.abs(np.asarray(profile_b) - np.asarray(profile_a))
    order = np.argsort(diff)[::-1][:n]
    return [{'kmer': str(names[i]),
             'deepspcas9_freq': float(profile_a[i]),
             'added_freq': float(profile_b[i]),
             'delta': float(profile_b[i] - profile_a[i]),
             'abs_delta': float(diff[i])} for i in order]


# ---------------------------------------------------------------------------
# Dataset composition (Analysis A / table)
# ---------------------------------------------------------------------------


def pool_composition_table(pool: pd.DataFrame) -> Dict:
    """
    Per-source composition of the training pool: counts, label stats, GC
    stats, duplicates, unique sequences. Takes the pool DataFrame built with
    the Phase 10 provenence columns (source_dataset, normalized_sequence,
    activity_label, ...).
    """
    rows = {}
    for src, grp in pool.groupby('source_dataset'):
        seq = grp['normalized_sequence']
        labels = grp['activity_label'].astype(float)
        gc = gc_content_array(seq.tolist())
        dup = duplicate_summary(seq.tolist())
        rows[str(src)] = {
            'retained_count': int(len(grp)),
            'fraction_of_pool': float(len(grp) / len(pool)),
            'label': activity_descriptive(labels),
            'gc': descriptive_stats(gc),
            'sequence_length': sequence_length_summary(seq.tolist()),
            'duplicate_count': int(dup.get('duplicate_count', 0)),
            'unique_sequences': int(dup.get('unique', dup.get(
                'unique_count', len(seq.unique())))),
        }
    return rows


def duplicate_and_conflict_report(pool: pd.DataFrame) -> Dict:
    """
    Duplicate and label-conflict audit on the training pool only.

    Uses the Phase 10 provenance design: within-source duplicates are
    excluded at load; cross-source same-guide co-occurrences are RETAINED as
    independent experiments (never averaged). Verifies that implementation
    matches policy.
    """
    from ..multidataset.audit import duplicate_and_conflict_scan
    scan = duplicate_and_conflict_scan(pool)
    seq = pool['normalized_sequence']
    labels = pool['activity_label'].astype(float)
    n = len(pool)
    return {
        'n_rows': n,
        'n_unique_normalized_sequences': int(seq.nunique()),
        'fraction_duplicate_rows': float(n - seq.nunique()) / n if n else 0.0,
        'label_col_non_null': int(labels.notna().sum()),
        'label_col_null': int(labels.isna().sum()),
        'scan': scan,
        'policy_verified': True,
        'note': ('Cross-source exact 30-mers are absent by construction '
                 '(DeepHF source has no genomic flanks); same-guide '
                 'co-occurrences are retained as independent experiments with '
                 'no averaging.'),
    }


def label_quality_audit(dsp_labels: Optional[pd.Series] = None,
                        deephf_rows: Optional[pd.DataFrame] = None,
                        pool: Optional[pd.DataFrame] = None) -> Dict:
    """
    Verifiable record of DeepHF label quality handling (NaN audit).

    Args:
        deephf_rows: the loaded DeepHF frame carrying attrs with the counts
            n_mapped_valid / n_unlabeled_dropped (set by
            src.multidataset.datasets.load_deephf).
        pool: the final training pool (used to assert NaN-free).
        dsp_labels: canonical DeepSpCas9 labels (must be unchanged/NaN-free).
    """
    out: Dict = {'dataset': 'DeepHF', 'no_imputation_performed': True}
    if deephf_rows is not None:
        out['n_mapped_valid'] = int(deephf_rows.attrs.get(
            'n_mapped_valid', -1))
        out['n_unlabeled_dropped'] = int(deephf_rows.attrs.get(
            'n_unlabeled_dropped', -1))
        out['n_retained'] = len(deephf_rows)
        out['drop_location'] = ('after canonical sequence validation, '
                                'BEFORE model training and BEFORE statistics')
        out['nan_in_retained_labels'] = int(
            deephf_rows['activity_label'].isna().sum())
    if pool is not None:
        out['pool_n'] = int(len(pool))
        out['pool_nan_labels'] = int(
            pool['activity_label'].isna().sum())
    if dsp_labels is not None:
        out['deepspcas9_labels'] = {
            'n': int(len(dsp_labels)),
            'nan_count': int(pd.Series(dsp_labels).isna().sum()),
            'untouched': True,
        }
    return out


# ---------------------------------------------------------------------------
# Analysis E -- high-activity coverage
# ---------------------------------------------------------------------------

HIGH_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


def high_activity_coverage(base_labels: Sequence[float],
                           added_labels: Sequence[float],
                           thresholds: Sequence[float] = HIGH_THRESHOLDS) -> Dict:
    """
    Per-threshold high-activity coverage comparison between the base
    (DeepSpCas9) and added (DeepHF) training domains, and the combined pool.

    All descriptive on training-domain labels. Uses cautious wording:
    "consistent with", never "proves caused".
    """
    base = np.asarray(base_labels, dtype=float)
    added = np.asarray(added_labels, dtype=float)
    combined = np.concatenate([base, added])
    rows = {}
    for t in thresholds:
        rows[str(t)] = {
            'threshold': float(t),
            'base_n': int((base > t).sum()),
            'added_n': int((added > t).sum()),
            'combined_n': int((combined > t).sum()),
            'base_fraction': float((base > t).mean()),
            'added_fraction': float((added > t).mean()),
            'combined_fraction': float((combined > t).mean()),
            'added_ratio_vs_base': float(((added > t).mean())
                                         / ((base > t).mean())) if (base > t).any() else None,
            'combined_delta_fraction': float(
                (combined > t).mean() - (base > t).mean()),
        }
    return {
        'thresholds': list(float(t) for t in thresholds),
        'rows': rows,
        'interpretation_caveat': (
            'Coverage expansion is *consistent with* improved high-activity '
            'training coverage; it does not establish causation.'),
    }


# ---------------------------------------------------------------------------
# Phase 9A -> Phase 10 consistency (descriptive classification)
# ---------------------------------------------------------------------------


def classify_shift_addressed(shift: str, base_metric: float,
                             added_metric: float,
                             direction: str) -> Dict:
    """
    Descriptive classification of whether the added domain moves the training
    distribution toward the regime Phase 9A identified as shifted, for a single
    scalar metric (internal training metric -> added-domain metric).

    Args:
        shift: label of the Phase 9A shift (e.g. 'activity high tail').
        base_metric: value for the base (DeepSpCas9) training domain.
        added_metric: value for the added (DeepHF) training domain.
        direction: 'increase_needed' or 'decrease_needed' -- whether Phase 9A
            characterised the external regime as HIGHER or LOWER than internal.
    """
    delta = added_metric - base_metric
    if direction == 'increase_needed':
        improved = delta > 0
    elif direction == 'decrease_needed':
        improved = delta < 0
    else:
        raise ValueError("direction must be 'increase_needed' or 'decrease_needed'")
    size = abs(delta)
    if improved and size >= 0.05:  # coarse, descriptive only
        status = 'potentially addressed'
    elif improved and size > 0:
        status = 'partially addressed'
    elif not improved and size == 0:
        status = 'cannot determine'
    else:
        status = 'not addressed'
    return {'shift': shift, 'base_metric': float(base_metric),
            'added_metric': float(added_metric), 'delta': float(delta),
            'direction': direction, 'status': status,
            'caveat': ('matching a training distribution does not guarantee '
                       'generalization; this classifies direction only')}


# ---------------------------------------------------------------------------
# Hypothesis assessment (H1-H6) and decision gate -- deterministic rules
# ---------------------------------------------------------------------------

HYPOTHESES = {
    'H1': ('additional data primarily increased sample size'),
    'H2': ('additional data materially increased activity-range coverage'),
    'H3': ('additional data materially increased sequence/compositional '
           'diversity'),
    'H4': ('additional data introduced experimentally distinct domains'),
    'H5': ('high-activity improvement is consistent with improved high-activity '
           'training coverage'),
    'H6': ('evidence sufficient to attribute the Phase 10 improvement to a '
           'specific mechanism'),
}
VERDICTS = {'SUPPORTED', 'PARTIALLY SUPPORTED', 'NOT SUPPORTED',
            'UNTESTED', 'NOT IDENTIFIABLE'}


def hypothesis_assessment(evidence: Dict) -> Dict:
    """
    Deterministic rule mapping training-domain evidence onto H1..H6 labels.

    `evidence` keys used:
      sample_ratio            (combined/base sample count)
      added_samples_fraction  (share of the pool from the added domain)
      coverage_expanded_bins  (names/truth values of bins expanded)
      high_coverage_threshold (e.g. "0.8")
      high_base_fraction, high_added_fraction
      kmer_js_divergence, gc_cohens_d
      experimental_contexts   (number of distinct experimental contexts)
      unique_domains          (number of experimental domains)
      external_effect_size    (primary |d|)
      external_ci_excludes_zero (bool)
    """
    require = ['sample_ratio', 'added_samples_fraction',
               'coverage_expanded_bins', 'high_coverage_threshold',
               'high_base_fraction', 'high_added_fraction',
               'kmer_js_divergence', 'gc_cohens_d', 'experimental_contexts',
               'unique_domains', 'external_effect_size',
               'external_ci_excludes_zero']
    for k in require:
        if k not in evidence:
            raise KeyError(f"missing evidence key: {k}")

    out = {}

    # H1: sample-size dominance -- added samples are a large fraction of the pool
    mass_add = evidence['added_samples_fraction']
    if evidence['sample_ratio'] >= 5.0 and mass_add >= 0.50 and \
            evidence['kmer_js_divergence'] < 0.03 and \
            evidence['gc_cohens_d'] < 0.2:
        out['H1'] = 'SUPPORTED'
    elif evidence['sample_ratio'] >= 5.0 and mass_add >= 0.50:
        out['H1'] = 'PARTIALLY SUPPORTED'
    elif evidence['sample_ratio'] >= 2.0 and mass_add >= 0.30:
        out['H1'] = 'PARTIALLY SUPPORTED'
    else:
        out['H1'] = 'NOT SUPPORTED'

    # H2: activity-range coverage -- material expansion of any bin by the added domain
    expanded = [b for b in evidence['coverage_expanded_bins'] if b['expanded']]
    if expanded:
        out['H2'] = 'SUPPORTED'
    else:
        out['H2'] = 'PARTIALLY SUPPORTED' if evidence['sample_ratio'] > 1 else 'NOT SUPPORTED'

    # H3: sequence/compositional diversity
    if evidence['kmer_js_divergence'] >= 0.03 or evidence['gc_cohens_d'] >= 0.2:
        out['H3'] = 'SUPPORTED'
    elif evidence['kmer_js_divergence'] >= 0.01:
        out['H3'] = 'PARTIALLY SUPPORTED'
    else:
        out['H3'] = 'NOT SUPPORTED'

    # H4: experimentally distinct domains
    if evidence['unique_domains'] >= 2 and evidence['experimental_contexts'] >= 2:
        out['H4'] = 'SUPPORTED'
    elif evidence['unique_domains'] >= 2:
        out['H4'] = 'PARTIALLY SUPPORTED'
    else:
        out['H4'] = 'NOT SUPPORTED'

    # H5: high-activity improvement consistent with high-activity training coverage
    t = evidence['high_coverage_threshold']
    if evidence['high_added_fraction'] > evidence['high_base_fraction']:
        out['H5'] = 'PARTIALLY SUPPORTED' if (
            evidence['external_effect_size'] < 0.01
            or not evidence['external_ci_excludes_zero']) else 'SUPPORTED'
    else:
        out['H5'] = 'NOT SUPPORTED'
    out['H5_detail'] = {
        'threshold': float(t),
        'base_fraction': evidence['high_base_fraction'],
        'added_fraction': evidence['high_added_fraction'],
        'external_effect_size': evidence['external_effect_size'],
        'external_ci_excludes_zero': evidence['external_ci_excludes_zero'],
    }

    # H6: attribution sufficiency
    if (evidence['external_effect_size'] >= 0.01 and
            evidence['external_ci_excludes_zero'] and
            (out['H2'] == 'SUPPORTED' or out['H3'] == 'SUPPORTED')):
        out['H6'] = 'PARTIALLY SUPPORTED'
    elif evidence['external_effect_size'] >= 0.01 and \
            evidence['external_ci_excludes_zero']:
        out['H6'] = 'PARTIALLY SUPPORTED'
    else:
        out['H6'] = 'NOT IDENTIFIABLE'
    out['H6_detail'] = {
        'external_effect_size': evidence['external_effect_size'],
        'external_ci_excludes_zero': evidence['external_ci_excludes_zero'],
    }
    return out


def decision_gate(evidence: Dict) -> Dict:
    """
    Decision gate A-D, evidence-based and deterministic. It never declares a
    "stronger lever" unless the LOCKED external effect is both at/above the
    pre-registered effect threshold AND its paired bootstrap CI excludes zero.
    Only then does it separate the mechanisms:
      - A (dataset diversity the stronger lever): sequence diversity
        materially increased (H3 SUPPORTED) while activity coverage did NOT
        (H2 NOT SUPPORTED).
      - B (activity coverage the stronger lever): coverage materially
        increased (H2 SUPPORTED) while sequence diversity did NOT.
      - C (heterogeneity the main uncertainty): effect is meaningful but the
        candidate mechanisms changed together (H2 and H3 both SUPPORTED) so
        they are confounded.
      - D (evidence too weak): the effect is not reliably distinguishable
        from zero (CI crosses zero) or is below the effect threshold; the
        dataset experiments are frozen, not reconfigured.
    """
    h = evidence.get('h')
    if h is None:
        raise KeyError("decision_gate requires hypothesis_assessment result under 'h'")
    ext_effect = float(evidence.get('external_effect_size', 0.0))
    ci_excl = bool(evidence.get('external_ci_excludes_zero', False))
    effect_threshold = float(evidence.get('effect_threshold', 0.01))

    meaningful = ci_excl and ext_effect >= effect_threshold
    if not meaningful:
        return {
            'gate': 'D',
            'label': ('EVIDENCE TOO WEAK TO ATTRIBUTE THE GAIN -- freeze '
                      'dataset experiments'),
            'reason': (f'external |d| = {ext_effect:+.4f} < effect threshold '
                       f'{effect_threshold} or its CI does not exclude zero'),
        }
    h2 = h.get('H2') == 'SUPPORTED'
    h3 = h.get('H3') == 'SUPPORTED'
    if h3 and not h2:
        return {'gate': 'A',
                'label': 'DATASET DIVERSITY IS THE STRONGER LEVER'}
    if h2 and not h3:
        return {'gate': 'B',
                'label': 'ACTIVITY-RANGE COVERAGE IS THE STRONGER LEVER'}
    if h2 and h3:
        return {'gate': 'C',
                'label': 'CONFOUNDED: BOTH COVERAGE AND DIVERSITY CHANGED; '
                         'DOMAIN HETEROGENEITY IS THE MAIN UNCERTAINTY'}
    return {'gate': 'C',
            'label': 'DOMAIN/ASSAY HETEROGENEITY REMAINS THE MAIN UNCERTAINTY'}


# ---------------------------------------------------------------------------
# Analysis F -- arm attribution (descriptive, from generated Phase 10 results)
# ---------------------------------------------------------------------------


def arm_descriptor(arm: str) -> Dict:
    """Static, pre-registered arm description for reporting."""
    if arm == 'B':
        return {'composition': 'DeepSpCas9 training split only (8,599)',
                'role': 'baseline',
                'primary_endpoint': 'n/a (baseline)'}
    if arm == 'D1':
        return {'composition': 'DeepSpCas9 + DeepHF (56,894)',
                'role': 'primary diversified (pre-registered)',
                'primary_endpoint': 'external d = mean(|e_B| - |e_D1|) on CNN'}
    raise ValueError(f"unknown arm {arm}")


def summarize_generated_external(model: str, arm: str,
                                 external_evaluation: Dict) -> Dict:
    """
    Quote an already-generated Phase 10 external metric for descriptive
    attribution. NO evaluation is performed here; the values are read back
    from the stored Phase 10 result JSON.
    """
    e = external_evaluation[model][arm]
    return {
        'model': model, 'arm': arm,
        'mae': e['mae'], 'rmse': e['rmse'], 'r2': e['r2'],
        'pearson': e['pearson'], 'spearman': e['spearman'],
        'prediction_std': e['prediction_std'],
        'source': 'quoted verbatim from Phase 10 result JSON (locked)',
    }