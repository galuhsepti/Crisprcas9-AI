"""
Phase 14 stratification.

Frozen bin construction. GC-bin boundaries are derived from the INTERNAL TRAIN
distribution ONLY. The quantile constructor refuses any input that is not
explicitly tagged as the training split, so external-specific quantile binning
is impossible by construction.

Activity bins are the fixed equal-width grid [0.0,0.2,0.4,0.6,0.8,1.0] on true
activity (last bin closed). The same edges are applied unchanged to internal
validation and external evaluation.

Per-bin reporting policy:
  - n < min_n_report : point estimates only (mae_ci is None)
  - n >= min_n_boot : percentile bootstrap 95% CI for bin MAE allowed
  - n == 0          : null entry, all stats None
  - R^2 is never reported within bins (documented instability).
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .statistics import percentile_bootstrap_ci, reproducible_bootstrap_seed


def construct_gc_edges(
    train_gc: Sequence[float],
    tag: str,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    round_dp: int = 2,
    fallback_edges: Sequence[float] = (0.40, 0.50, 0.60, 0.70),
) -> List[float]:
    """
    Build frozen GC bin edges from the internal train distribution only.

    Args:
        train_gc: per-sequence GC of the INTERNAL TRAIN split.
        tag: must equal 'train'. Any other tag raises RuntimeError, preventing
            quantile binning of external (or validation) data.
        quantiles: quantile probabilities for the bin boundaries.
        round_dp: rounding precision for edge values.
        fallback_edges: fixed absolute grid used if the rounded quantiles
            collapse (fewer distinct edges than quantiles).

    Returns:
        List of internal bin split-points (4 edges -> 5 bins).
    """
    if tag != "train":
        raise RuntimeError(
            "GC quantile edges may only be constructed from the internal TRAIN "
            f"split; got tag='{tag}'. External-specific quantile binning is "
            "forbidden."
        )
    g = np.asarray(train_gc, dtype=float).ravel()
    if g.size < 50:
        raise ValueError("Train GC array too small for stable quantile edges")
    q = np.asarray(quantiles, dtype=float)
    if not (0 < q).all() or not (q < 1).all():
        raise ValueError("Quantiles must lie strictly within (0, 1)")
    if q.size < 1:
        raise ValueError("At least one quantile required")
    edges = sorted(set(np.quantile(g, q).round(round_dp).tolist()))
    if len(edges) < len(q):
        edges = sorted(float(e) for e in fallback_edges)
        if len(edges) < len(q):
            raise ValueError("Fallback GC grid is degenerate")
        source = "fallback_absolute_grid"
    else:
        edges = edges[: len(q)]
        source = "internal_train_quantile"
    return {
        "edges": edges,
        "source": source,
        "quantiles": q.tolist(),
        "round_dp": round_dp,
        "fallback_edges": [float(e) for e in fallback_edges],
    }


def bin_indices(
    values: Sequence[float],
    edges: Sequence[float],
    rightmost_closed: bool = True,
) -> np.ndarray:
    """
    Map values to bin indices 0..len(edges).

    Lower bin edges are inclusive, upper edges exclusive; the last bin is
    closed on the right when rightmost_closed is True.
    """
    v = np.asarray(values, dtype=float).ravel()
    e = np.asarray(sorted(float(x) for x in edges), dtype=float)
    if e.size < 1:
        raise ValueError("At least one bin edge is required")
    idx = np.searchsorted(e, v, side="right")
    np.clip(idx, 0, e.size, out=idx)
    if rightmost_closed:
        idx = np.where((v == e[-1]), e.size, idx)
    np.clip(idx, 0, e.size, out=idx)
    return idx


def _bin_label(lo: float, hi: float, last: bool) -> str:
    if last:
        return f"[{lo:.2f}, {hi:.2f}]"
    return f"[{lo:.2f}, {hi:.2f})"


def bin_bounds(edges: Sequence[float], lo: float, hi: float):
    """
    Return (value_lows, value_highs) for n_bins = len(edges)+1 bins defined by
    `edges`, using [lo, hi] as the outer display bounds of the axis.
    """
    e = [float(x) for x in edges]
    lows = [lo] + e
    highs = e + [hi]
    return lows, highs


def per_bin_stats(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    bin_idx: np.ndarray,
    n_bins: int,
    bin_value_min: Sequence[float],
    bin_value_max: Sequence[float],
    min_n_report: int = 20,
    min_n_boot: int = 30,
    n_boot: int = 1000,
    bootstrap_seed: int = 20260907,
) -> List[Dict]:
    """
    Per-bin error statistics on a shared frozen bin grid.

    Reported per bin: n, mean true/pred, signed bias (pred - true), MAE, RMSE,
    target SD, prediction SD, prediction/target SD ratio and (when n >=
    min_n_boot) a bootstrap 95% CI for bin MAE. R^2 is never reported within
    bins.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape or yt.shape != np.asarray(bin_idx).shape:
        raise ValueError("y_true, y_pred and bin_idx must be identically shaped")
    if len(bin_value_min) != n_bins or len(bin_value_max) != n_bins:
        raise ValueError("bin_value_min/max must have length n_bins")

    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        last = b == n_bins - 1
        label = _bin_label(bin_value_min[b], bin_value_max[b], last)
        if n == 0:
            out.append({"bin": label, "n": 0})
            continue
        ytb, ypb = yt[mask], yp[mask]
        entry: Dict = {
            "bin": label,
            "n": n,
            "mean_true": float(np.mean(ytb)),
            "mean_pred": float(np.mean(ypb)),
            "bias": float(np.mean(ypb - ytb)),
            "mae": float(np.mean(np.abs(ytb - ypb))),
            "rmse": float(np.sqrt(np.mean((ytb - ypb) ** 2))),
            "true_std": float(np.std(ytb, ddof=1)) if n > 1 else 0.0,
            "prediction_std": float(np.std(ypb, ddof=1)) if n > 1 else 0.0,
            "mae_ci": None,
            "excluded_min_n": n < min_n_report,
        }
        if n > 1 and np.std(ytb, ddof=1) > 0:
            entry["sd_ratio_pred_over_true"] = float(
                np.std(ypb, ddof=1) / np.std(ytb, ddof=1)
            )
        else:
            entry["sd_ratio_pred_over_true"] = None
        if n >= min_n_boot:
            rng = reproducible_bootstrap_seed(
                bootstrap_seed + 1 + b
            )
            entry["mae_ci"] = percentile_bootstrap_ci(
                np.abs(ytb - ypb),
                lambda v: float(np.mean(v)),
                n_boot=n_boot,
                rng=rng,
            )
        out.append(entry)
    return out


def activity_bin_edges() -> List[float]:
    """Fixed activity bin grid [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]."""
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def common_support_joint_delta(
    data_int: Dict[str, object],
    data_ext: Dict[str, object],
    n_gc_bins: int,
    n_act_bins: int,
    min_n: int = 20,
    model: str = "random_forest",
) -> Dict[str, object]:
    """
    C6 joint-stratification on COMMON-SUPPORT GC x activity cells.

    A cell is retained only when BOTH domains have n >= min_n in that cell.
    Reported cells therefore represent the common support of the two datasets,
    NOT the full datasets, and are labelled accordingly.

    Returns:
        dict with cells (each: gc_bin, activity_bin, n_int, n_ext, int_mae,
        ext_mae, delta_mae), the median delta MAE (ext - int), its bootstrap CI
        over cells, and coverage metadata.
    """
    gc_int = data_int["gc_bin"]
    gc_ext = data_ext["gc_bin"]
    act_int = data_int["activity_bin"]
    act_ext = data_ext["activity_bin"]
    mae_int = np.abs(data_int["true"] - data_int[f"pred_{model}"])
    mae_ext = np.abs(data_ext["true"] - data_ext[f"pred_{model}"])

    cells = []
    for g in range(n_gc_bins):
        for a in range(n_act_bins):
            n_int = int(np.sum((gc_int == g) & (act_int == a)))
            n_ext = int(np.sum((gc_ext == g) & (act_ext == a)))
            if n_int < min_n or n_ext < min_n:
                continue
            mi = float(np.mean(mae_int[(gc_int == g) & (act_int == a)]))
            me = float(np.mean(mae_ext[(gc_ext == g) & (act_ext == a)]))
            cells.append({
                "gc_bin": g,
                "activity_bin": a,
                "n_int": n_int,
                "n_ext": n_ext,
                "int_mae": mi,
                "ext_mae": me,
                "delta_mae": me - mi,
            })
    deltas = np.array([c["delta_mae"] for c in cells], dtype=float)
    result: Dict[str, object] = {
        "definition": (
            "common-support GC x activity cells with n >= "
            f"{min_n} in BOTH domains; restricted to common support and does "
            "not represent the full datasets"
        ),
        "model": model,
        "n_total_gc_bins": int(n_gc_bins),
        "n_total_act_bins": int(n_act_bins),
        "n_cells": int(len(cells)),
        "cells": cells,
        "median_delta_mae": float(np.median(deltas)) if deltas.size else None,
    }
    if deltas.size >= 2:
        from .statistics import bootstrap_p_value, reproducible_bootstrap_seed
        rng = reproducible_bootstrap_seed(20260907 + 60)
        boot = np.empty(1000)
        for i in range(1000):
            sel = rng.integers(0, deltas.size, size=deltas.size)
            boot[i] = np.median(deltas[sel])
        result["median_delta_mae_ci"] = {
            "ci_lower": float(np.percentile(boot, 2.5)),
            "ci_upper": float(np.percentile(boot, 97.5)),
            "p_value": bootstrap_p_value(boot),
        }
    return result