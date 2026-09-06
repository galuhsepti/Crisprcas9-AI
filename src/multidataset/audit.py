"""
Contamination and duplicate audit for Phase 10.

All checks use the canonical normalization: uppercase, ACGT-only, 30-mer with
guide [4:24] and PAM [24:27]. For candidates whose source geometry differs
(DeepHF), the constructed 30-mer from datasets.load_deephf is used.

Moreno-Mateos is LOCKED. This module only reports statistics about it; it is
never a fitting/tuning input.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .datasets import NORMALIZED_SEQ_COL, ACTIVITY_COL, SOURCE_COL

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    seq = str(seq).upper()
    return seq.translate(_COMPLEMENT)[::-1]


def overlap_report(sets: Dict[str, Set[str]]) -> Dict[str, Dict]:
    """
    Pairwise overlap statistics among named sequence sets.

    Returns, for every pair (a, b):
      - 'n_a', 'n_b', 'jaccard'
      - 'intersection_fwd' (exact matches)
      - 'intersection_rc'  (a sequence in b whose reverse complement is in a)
      - 'prop_of_b'        (|a & b| / |b|)
    """
    names = list(sets.keys())
    out: Dict[str, Dict] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa, sb = sets[a], sets[b]
            inter = sa & sb
            inter_rc = set(reverse_complement(x) for x in sb) & sa
            out[f"{a}_vs_{b}"] = {
                "n_a": len(sa),
                "n_b": len(sb),
                "intersection_fwd": len(inter),
                "intersection_rc": len(inter_rc),
                "prop_of_b": (len(inter) / len(sb)) if sb else 0.0,
                "prop_of_a": (len(inter) / len(sa)) if sa else 0.0,
                "jaccard": (len(inter) / len(sa | sb)) if (sa | sb) else 0.0,
            }
    return out


def contamination_report(external_set: Set[str],
                         candidate_sets: Dict[str, Set[str]]) -> Dict:
    """
    Moremo-Mateos contamination check for each candidate set (forward and
    reverse-complement). Returns per-candidate counts and a boolean
    'eligible_no_contamination'.
    """
    out: Dict = {"external_n": len(external_set), "candidates": {}}
    for name, cand in candidate_sets.items():
        fwd = cand & external_set
        rc_ = set(reverse_complement(x) for x in cand) & external_set
        out["candidates"][name] = {
            "n": len(cand),
            "overlap_fwd": len(fwd),
            "overlap_rc": len(rc_),
            "eligible": len(fwd) == 0 and len(rc_) == 0,
        }
    return out


def duplicate_and_conflict_scan(pool: pd.DataFrame) -> Dict:
    """
    Duplicate audit over the merged training pool.

    - within-source duplicates: (source, sequence) repeats -> report count.
    - cross-source duplicates: exact 30-mer appearing in >= 2 sources
      (pairwise), with label summaries (n / min / max / spread).
    - label conflicts: same (source, sequence) carrying multiple distinct
      labels -> count and maximum pairwise |difference| (scientific
      duplicate/conflict policy is applied in the pool builder, not here).
    """
    pool = pool.reset_index(drop=True)
    seq = pool[NORMALIZED_SEQ_COL]
    src = pool[SOURCE_COL]
    act = pool[ACTIVITY_COL].astype(float)

    within = pool.duplicated(subset=[SOURCE_COL, NORMALIZED_SEQ_COL]).sum()

    cross: Dict = {}
    seq_sources: Dict[str, List] = {}
    for s, so, a in zip(seq, src, act):
        seq_sources.setdefault(str(s), []).append((so, a))
    n_cross_duplicated_rows = sum(1 for v in seq_sources.values() if len(v) > 1)
    seq_label_conflicts = []
    for s, pairs in seq_sources.items():
        labs = sorted({round(float(a), 10) for _, a in pairs})
        if len(pairs) > 1 and len(set(so for so, _ in pairs)) > 1:
            # cross-source same-30-mer: retention as independent replicates
            cross[f"shared_{int(len(pairs))}_ways"] = cross.get(
                f"shared_{int(len(pairs))}_ways", 0) + 1
            diff = abs(float(pairs[0][1]) - float(pairs[1][1])) if len(pairs) == 2 else None
            seq_label_conflicts.append({
                "sequence": s,
                "sources": sorted({so for so, _ in pairs}),
                "labels": [round(float(a), 6) for _, a in pairs],
                "abs_label_diff": diff,
            })
    # within-source label conflicts (same source AND same sequence, different labels)
    within_conflicts = 0
    by_src_seq = pool.groupby([SOURCE_COL, NORMALIZED_SEQ_COL])[ACTIVITY_COL]
    for _, grp in by_src_seq:
        if grp.nunique() > 1:
            within_conflicts += 1

    guide_seq = pool.get("guide_sequence")
    shared_guide_cross = 0
    if guide_seq is not None:
        gsg = pool.groupby("guide_sequence")[SOURCE_COL].nunique()
        shared_guide_cross = int((gsg > 1).sum())

    return {
        "within_source_duplicate_rows": int(within),
        "within_source_label_conflicts": int(within_conflicts),
        "cross_source_exact_30mer_rows": int(n_cross_duplicated_rows),
        "cross_source_exact_30mer_pairwise": {
            k: int(v) for k, v in cross.items()},
        "cross_source_label_conflicts": [  # same-30-mer in >=2 sources
            {k: c[k] for k in ("sequence", "sources", "labels",
                               "abs_label_diff")}
            for c in seq_label_conflicts[:20]],
        "cross_source_label_conflict_count": len(seq_label_conflicts),
        "shared_guide_cross_source": int(shared_guide_cross),
    }


def label_distribution_per_source(pool: pd.DataFrame) -> Dict:
    """Per-source label distribution summary for the pool."""
    out = {}
    for src, grp in pool.groupby(SOURCE_COL):
        a = grp[ACTIVITY_COL].astype(float)
        out[str(src)] = {
            "n": int(len(grp)),
            "label_min": float(a.min()),
            "label_max": float(a.max()),
            "label_mean": float(a.mean()),
            "label_median": float(a.median()),
            "label_std": float(a.std()),
        }
    return out


def sequence_metadata_summary(pool: pd.DataFrame) -> Dict:
    """Sequence length, GC and duplicate stats across the pool."""
    lengths = pool[NORMALIZED_SEQ_COL].str.len()
    gc = np.mean(
        [np.mean([nuc in "GC" for nuc in seq])
         for seq in pool[NORMALIZED_SEQ_COL]])
    out = {
        "n": int(len(pool)),
        "seq_length_mode": int(lengths.mode().iloc[0]),
        "seq_length_min": int(lengths.min()),
        "seq_length_max": int(lengths.max()),
        "gc_mean": float(gc),
    }
    per_source_gc = {}
    for src, grp in pool.groupby(SOURCE_COL):
        gcs = [np.mean([nuc in "GC" for nuc in seq])
               for seq in grp[NORMALIZED_SEQ_COL]]
        per_source_gc[str(src)] = {"gc_mean": float(np.mean(gcs))}
    out["gc_per_source"] = per_source_gc
    return out