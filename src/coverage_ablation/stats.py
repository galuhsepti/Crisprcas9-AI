"""
Phase 12 - Controlled Data-Coverage Ablation: paired statistics.

Effect estimation on a COMMON validation set (the canonical 1,518-row split),
mirroring the Phase 10 conventions (paired error bootstrap on mean |e|,
Cohen's dz, paired t, Wilcoxon). Pure functions over numeric arrays; used
only if arms are trained (they are not in the infeasible path).
"""

from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps

from ..multidataset.stats import bootstrap_ci_meandiff as _boot


def paired_bootstrap_meandiff(errors_arm_a: Sequence[float],
                              errors_arm_b: Sequence[float],
                              n_boot: int = 1000,
                              seed: int = 42) -> dict:
    """Bootstrap CI and point estimate of mean(|e_A| - |e_B|), Phase-10 rule."""
    b = _boot(np.asarray(errors_arm_a, dtype=float),
              np.asarray(errors_arm_b, dtype=float), n_boot, seed)
    return {
        'mean_pairwise_diff': b['point'],
        'bootstrap_ci': {'ci_lower': b['ci_lower'],
                         'ci_upper': b['ci_upper']},
        'n_boot': b['n_boot'],
        'seed': b['seed'],
    }


def cohens_dz(diffs: Sequence[float]) -> float:
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def paired_t_and_wilcoxon(errors_arm_a: Sequence[float],
                          errors_arm_b: Sequence[float]) -> dict:
    ea = np.asarray(errors_arm_a, dtype=float)
    eb = np.asarray(errors_arm_b, dtype=float)
    t, tp = sps.ttest_rel(eb, ea)
    w, wp = sps.wilcoxon(ea, eb)
    return {'t_statistic': float(t), 't_p_value': float(tp),
            'wilcoxon_statistic': float(w), 'wilcoxon_p_value': float(wp)}


def contrast_report(true_labels: Sequence[float],
                    pred_a: Sequence[float],
                    pred_b: Sequence[float],
                    metric: str = 'mae') -> dict:
    """
    Paired effect of arm B vs arm A on a common true-label vector for one
    error metric (default MAE). Returns per-arm metric, delta, paired
    bootstrap CI of the mean signed difference, dz, and paired tests.
    """
    y = np.asarray(true_labels, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    if not (y.shape == pa.shape == pb.shape):
        raise ValueError('true/pred arrays must share shape')
    if metric == 'mae':
        ea, eb = np.abs(y - pa), np.abs(y - pb)
        def m(e): return float(np.mean(e))
    elif metric == 'rmse':
        ea, eb = (y - pa) ** 2, (y - pb) ** 2
        def m(e): return float(np.sqrt(np.mean(e)))
    elif metric == 'r2':
        mu = y.sum() / y.size
        r2a = 1 - float(np.sum((y - pa) ** 2) / np.sum((y - mu) ** 2))
        r2b = 1 - float(np.sum((y - pb) ** 2) / np.sum((y - mu) ** 2))
        return {'metric': metric, 'value_a': r2a, 'value_b': r2b,
                'delta_b_minus_a': float(r2b - r2a)}
    else:
        raise ValueError(f'unsupported metric {metric}')
    boot = paired_bootstrap_meandiff(eb, ea)  # mean(|e_B| - |e_A|)
    return {
        'metric': metric,
        'value_a': m(ea),
        'value_b': m(eb),
        'delta_b_minus_a': float(m(eb) - m(ea)),
        'mean_pairwise_diff': boot['mean_pairwise_diff'],
        'ci_lower': boot['bootstrap_ci']['ci_lower'],
        'ci_upper': boot['bootstrap_ci']['ci_upper'],
        'cohens_dz': cohens_dz(eb - ea),
        'n_boot': boot['n_boot'],
        **paired_t_and_wilcoxon(ea, eb),
    }