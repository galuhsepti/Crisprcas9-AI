"""
Phase 12 - Controlled Data-Coverage Ablation: pool metadata + analysis.

Reuses the Phase 11 canonical grid/statistics helpers; adds the Phase-12
per-arm metadata record required by protocol Section 25 and the high-activity
summary required by Section 8.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..diagnostics.stats import descriptive_stats
from ..diagnostics.sequence_analysis import (
    duplicate_summary,
    gc_content_array,
    mean_kmer_frequencies,
    sequence_length_summary,
)
from ..dataset_analysis.analysis import (
    activity_descriptive,
    js_divergence,
)
from .config import (
    ACTIVITY_EDGES,
    HIGH_ACTIVITY_THRESHOLD,
    ACTIVITY_QUANTILES,
    GEOMETRY_TRANSFORMATION,
    LABEL_HARMONIZATION,
)


def arm_pool_metadata(pool: pd.DataFrame,
                      arm_name: str,
                      selection_method: str,
                      random_seed: Optional[int],
                      tolerances: Dict,
                      input_hashes: Dict) -> Dict:
    """
    Protocol Section 25 provenance record for a generated arm pool.
    Everything is derived from the pool rows plus the inputs the runner
    passes; deterministic for a fixed pool.
    """
    seq = pool['normalized_sequence'].astype(str)
    lab = pool['activity_label'].astype(float)
    gc = gc_content_array(seq.tolist())
    dup = duplicate_summary(seq.tolist())
    per_source = pool.groupby('source_dataset').size().to_dict()
    n = len(pool)

    acts = activity_descriptive(lab)
    acts.update({'quantiles': {
        f'p{q}': float(np.percentile(lab.values, q))
        for q in ACTIVITY_QUANTILES}})
    acts['fraction_in_bins'] = {
        f'[{ACTIVITY_EDGES[i]:.2f},{ACTIVITY_EDGES[i+1]:.2f}]': None
        for i in range(len(ACTIVITY_EDGES) - 1)}
    from ..dataset_analysis.analysis import coverage_by_bin
    for b in coverage_by_bin(lab, ACTIVITY_EDGES):
        acts['fraction_in_bins'][b['bin']] = b['fraction']

    k3 = mean_kmer_frequencies(seq.tolist(), 3)
    k3_entropy = float(-np.sum(k3[k3 > 0] * np.log2(k3[k3 > 0])))

    return {
        'arm': str(arm_name),
        'source_datasets': sorted(per_source.keys()),
        'n_samples': n,
        'n_unique_sequences': int(dup['unique']),
        'n_duplicate_rows': int(n - dup['unique']),
        'duplicate_rate': float(dup['duplicate_rate']),
        'activity': acts,
        'gc': descriptive_stats(gc),
        'kmer_stats': {'kmer3_self_entropy': k3_entropy},
        'source_proportions': {
            str(k): float(v / n) for k, v in per_source.items()},
        'experimental_contexts': sorted(
            pool.get('experimental_context', pd.Series(dtype=object))
            .dropna().unique().tolist()),
        'selection_method': selection_method,
        'random_seed': random_seed,
        'matching_tolerances': tolerances,
        'creation_timestamp': None,  # filled by the runner
        'input_hashes': input_hashes,
        'geometry_transformation': GEOMETRY_TRANSFORMATION,
        'label_harmonization': LABEL_HARMONIZATION,
    }


def high_activity_summary(labels: Sequence,
                          baseline_labels: Optional[Sequence] = None,
                          threshold: float = HIGH_ACTIVITY_THRESHOLD) -> Dict:
    """
    Section-8 high-activity inventory for a pool: count, fraction, upper
    quantiles, and (optionally) relative enrichment versus a baseline.
    """
    y = np.asarray(labels, dtype=float).ravel()
    out = {
        'threshold': float(threshold),
        'n': int(y.size),
        'n_high': int((y > threshold).sum()),
        'fraction_high': float((y > threshold).mean()),
        'q90': float(np.percentile(y, 90)),
        'q95': float(np.percentile(y, 95)),
        'q99': float(np.percentile(y, 99)),
    }
    if baseline_labels is not None:
        yb = np.asarray(baseline_labels, dtype=float).ravel()
        fb = float((yb > threshold).mean())
        out['baseline_fraction_high'] = fb
        out['relative_enrichment'] = (
            float(out['fraction_high'] / fb) if fb > 0 else None)
    return out


def feasibility_outputs(feasibility: Dict,
                        provenance: Dict,
                        environment: Dict,
                        input_hashes: Dict,
                        locked_reference: Dict) -> Dict:
    """Assemble the Phase-12 result JSON body for the STOP path."""
    return {
        'experiment_name': 'phase12_controlled_data_coverage_ablation',
        'phase': '12',
        'status': feasibility['decision'],
        'feasibility_audit': feasibility,
        'provenance': provenance,
        'environment': environment,
        'input_hashes': input_hashes,
        'locked_reference': locked_reference,
        'arms_constructed': False,
        'models_trained': [],
'external_evaluation_performed': False,
        'interpretation_rule': (
            'No causal language is used: controlled comparisons are '
            '"consistent with" a contributor, never "caused by".'),
    }