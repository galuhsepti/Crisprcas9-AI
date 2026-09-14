"""Locked paired metrics, hierarchical bootstrap, and bridge diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats

from src.evaluation.metrics import (
    calculate_mae,
    calculate_pearson_correlation,
    calculate_r2,
    calculate_rmse,
    calculate_spearman_correlation,
)
from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    SEEDS,
    classify_outcome,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute locked metrics and reject undefined/nonfinite results."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.size < 2 or truth.shape != prediction.shape:
        raise ValueError(
            "Metrics require paired arrays with at least two rows"
        )
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Metrics require finite values")
    if np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        raise ValueError("Correlation is undefined for constant values")
    pearson = float(calculate_pearson_correlation(truth, prediction)[0])
    spearman = float(calculate_spearman_correlation(truth, prediction)[0])
    result = {
        "spearman": spearman,
        "pearson": pearson,
        "r2": float(calculate_r2(truth, prediction)),
        "mae": float(calculate_mae(truth, prediction)),
        "rmse": float(calculate_rmse(truth, prediction)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("A required metric is nonfinite")
    return result


def paired_effects(y_true, baseline, multidomain) -> tuple[float, float]:
    """Return D-minus-baseline rho and relative RMSE gain."""
    base = regression_metrics(y_true, baseline)
    shared = regression_metrics(y_true, multidomain)
    if base["rmse"] == 0:
        raise ValueError(
            "Relative RMSE gain is undefined for zero baseline RMSE"
        )
    return (
        shared["spearman"] - base["spearman"],
        (base["rmse"] - shared["rmse"]) / base["rmse"],
    )


@dataclass(frozen=True)
class DomainComparison:
    group_ids: np.ndarray
    labels: np.ndarray
    baseline_predictions: Mapping[int, np.ndarray]
    multidomain_predictions: Mapping[int, np.ndarray]


def _validate_comparison(comparison: DomainComparison) -> None:
    n = len(comparison.labels)
    if n < 2 or len(comparison.group_ids) != n:
        raise ValueError("Comparison rows are incomplete")
    if set(comparison.baseline_predictions) != set(SEEDS):
        raise ValueError("Baseline predictions must contain all locked seeds")
    if set(comparison.multidomain_predictions) != set(SEEDS):
        raise ValueError(
            "Multi-domain predictions must contain all locked seeds"
        )
    arrays = [comparison.labels]
    arrays.extend(comparison.baseline_predictions.values())
    arrays.extend(comparison.multidomain_predictions.values())
    if any(np.asarray(array).reshape(-1).size != n for array in arrays):
        raise ValueError("Prediction lengths do not match labels")


def _group_draw_indices(group_ids, generator) -> np.ndarray:
    identifiers = np.asarray(group_ids).astype(str)
    unique = np.asarray(sorted(set(identifiers.tolist())))
    draw = generator.choice(unique, size=len(unique), replace=True)
    positions = []
    for identifier in draw:
        positions.extend(np.flatnonzero(identifiers == identifier).tolist())
    return np.asarray(positions, dtype=np.int64)


def hierarchical_paired_bootstrap(
    comparisons: Mapping[str, DomainComparison],
    *,
    iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """Execute the locked seed-then-spacer paired hierarchical bootstrap."""
    if set(comparisons) != set(DOMAINS):
        raise ValueError("Bootstrap requires exactly both locked domains")
    if iterations <= 0 or seed != 42:
        raise ValueError(
            "Bootstrap iterations must be positive and seed is locked"
        )
    for comparison in comparisons.values():
        _validate_comparison(comparison)
    generator = np.random.Generator(np.random.PCG64(seed))
    distributions = {
        domain: {"rho": [], "rmse_gain": [], "invalid_bootstrap_n": 0}
        for domain in DOMAINS
    }
    for _ in range(iterations):
        seed_positions = generator.integers(0, len(SEEDS), size=len(SEEDS))
        for domain in DOMAINS:
            comparison = comparisons[domain]
            rows = _group_draw_indices(comparison.group_ids, generator)
            effects = []
            try:
                for position in seed_positions:
                    run_seed = SEEDS[int(position)]
                    effects.append(
                        paired_effects(
                            np.asarray(comparison.labels)[rows],
                            np.asarray(
                                comparison.baseline_predictions[run_seed]
                            )[rows],
                            np.asarray(
                                comparison.multidomain_predictions[run_seed]
                            )[rows],
                        )
                    )
            except ValueError:
                distributions[domain]["invalid_bootstrap_n"] += 1
                continue
            distributions[domain]["rho"].append(
                float(np.mean([effect[0] for effect in effects]))
            )
            distributions[domain]["rmse_gain"].append(
                float(np.mean([effect[1] for effect in effects]))
            )
    evidence = {}
    for domain in DOMAINS:
        comparison = comparisons[domain]
        observed = []
        for run_seed in SEEDS:
            try:
                observed.append(
                    paired_effects(
                        comparison.labels,
                        comparison.baseline_predictions[run_seed],
                        comparison.multidomain_predictions[run_seed],
                    )
                )
            except ValueError:
                observed.append((None, None))
        domain_result = {
            "seeds": list(SEEDS),
            "rho": [effect[0] for effect in observed],
            "rmse_gain": [effect[1] for effect in observed],
            "invalid_bootstrap_n": distributions[domain][
                "invalid_bootstrap_n"
            ],
            "bootstrap_iterations": iterations,
        }
        for key in ("rho", "rmse_gain"):
            values = distributions[domain][key]
            if values:
                domain_result[f"{key}_ci"] = np.quantile(
                    values, [0.00625, 0.99375], method="linear"
                ).tolist()
                domain_result[f"{key}_descriptive_ci"] = np.quantile(
                    values, [0.025, 0.975], method="linear"
                ).tolist()
            else:
                domain_result[f"{key}_ci"] = [None, None]
                domain_result[f"{key}_descriptive_ci"] = [None, None]
        evidence[domain] = domain_result
    directional_harm = {}
    for domain, record in evidence.items():
        if any(value is None for value in record["rho"] + record["rmse_gain"]):
            directional_harm[domain] = None
            continue
        directional_harm[domain] = {
            "mean_delta_rho_below_margin": float(np.mean(record["rho"]))
            < -0.01,
            "mean_rmse_gain_below_margin": float(np.mean(record["rmse_gain"]))
            < -0.02,
        }
    return {
        "evidence": evidence,
        "outcome": classify_outcome(evidence),
        "bootstrap_seed": seed,
        "bootstrap_distributions": {
            domain: {
                "rho": distributions[domain]["rho"],
                "rmse_gain": distributions[domain]["rmse_gain"],
            }
            for domain in DOMAINS
        },
        "directional_harm": directional_harm,
    }


def bridge_metrics(labels_a, labels_b, predictions_a, predictions_b) -> dict:
    """Compute the locked descriptive 41-pair bridge diagnostics."""
    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (labels_a, labels_b, predictions_a, predictions_b)
    ]
    if any(len(values) != 41 for values in arrays):
        raise ValueError("Bridge diagnostics require exactly 41 paired rows")
    label_a, label_b, prediction_a, prediction_b = arrays
    own_a = regression_metrics(label_a, prediction_a)
    own_b = regression_metrics(label_b, prediction_b)
    head = regression_metrics(prediction_a, prediction_b)
    kendall = float(stats.kendalltau(prediction_a, prediction_b).statistic)
    rank_a = (stats.rankdata(prediction_a, method="average") - 1) / 40
    rank_b = (stats.rankdata(prediction_b, method="average") - 1) / 40
    if not np.isfinite(kendall):
        raise ValueError("Bridge Kendall correlation is undefined")
    return {
        "head_to_head_spearman": head["spearman"],
        "head_to_head_kendall": kendall,
        "head_to_own_label_spearman": {
            DOMAINS[0]: own_a["spearman"],
            DOMAINS[1]: own_b["spearman"],
        },
        "head_to_own_label_pearson": {
            DOMAINS[0]: own_a["pearson"],
            DOMAINS[1]: own_b["pearson"],
        },
        "mean_absolute_percentile_rank_disagreement": float(
            np.mean(np.abs(rank_a - rank_b))
        ),
    }
