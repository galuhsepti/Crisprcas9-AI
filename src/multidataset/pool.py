"""
Training-pool construction for Phase 10.

Every sequence row in the diversified pool keeps full provenance so the
contribution of each training domain is auditable:
    source_dataset | original_sequence | normalized_sequence |
    activity_label | label_transform  | experimental_context |
    duplicate_status | split_assignment
"""

from typing import List

import pandas as pd

from .config import (
    LABEL_HARMONIZATION,
    GEOMETRY_TRANSFORMATION,
    PRIMARY_ARM,
    BASELINE_ARM,
)
from .datasets import (
    NORMALIZED_SEQ_COL,
    ACTIVITY_COL,
    SOURCE_COL,
)
from .audit import duplicate_and_conflict_scan, label_distribution_per_source

POOL_COLUMNS = [
    SOURCE_COL,
    "original_sequence",
    NORMALIZED_SEQ_COL,
    ACTIVITY_COL,
    "label_transform",
    "experimental_context",
    "duplicate_status",
    "split_assignment",
    "guide_sequence",
    "pam",
]


def _context_for_source(source: str) -> str:
    if source == "deepspcas9":
        return ("DeepSpCas9 pooled SpCas9 screen (human), NGS "
                "modified-read fraction (modFreq)")
    if source == "deephf":
        return ("DeepHF pooled SpCas9 screens (human, HEK293T/HCT116/T cells); "
                "21-mer source mapped to canonical 30-mer via constant-flank "
                "transformation")
    return "unknown"


def annotate_pool_rows(df: pd.DataFrame, source: str,
                       split_assignment: str) -> pd.DataFrame:
    """
    Add the canonical pool metadata columns to a validated dataset DataFrame.
    (duplicate_status defaults to 'single'; cross-source shared-guide marking
    is applied afterwards by build_training_pool.)
    """
    out = df.copy()
    out["split_assignment"] = split_assignment
    out["label_transform"] = LABEL_HARMONIZATION["method"]
    out["experimental_context"] = _context_for_source(source)
    if "original_sequence" not in out.columns:
        out["original_sequence"] = out[NORMALIZED_SEQ_COL]
    out["duplicate_status"] = "single"
    if "pam" not in out.columns:
        out["pam"] = out[NORMALIZED_SEQ_COL].str[24:27]
    if "guide_sequence" not in out.columns:
        out["guide_sequence"] = out[NORMALIZED_SEQ_COL].str[4:24]
    return out[POOL_COLUMNS]


def shared_guide_set(*frames: pd.DataFrame) -> set:
    """20 bp guides present in >= 2 of the provided training frames."""
    counted: dict = {}
    for df in frames:
        guides = (df["guide_sequence"] if "guide_sequence" in df.columns
                  else df[NORMALIZED_SEQ_COL].str[4:24])
        for g in set(guides):
            counted[str(g)] = counted.get(str(g), 0) + 1
    return {g for g, c in counted.items() if c > 1}


def build_training_pool(dsp_train: pd.DataFrame,
                        hf_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full diversified training pool (arm D1) with provenance.

    - dsp_train: DeepSpCas9 canonical training split (85%, seed 42), validated.
    - hf_train:  DeepHF validated + geometry-mapped rows.
    Both inputs must already carry the canonical data columns.
    """
    pool_dsp = annotate_pool_rows(dsp_train, "deepspcas9", "train")
    pool_hf = annotate_pool_rows(hf_train, "deephf", "train")
    shared = shared_guide_set(pool_dsp, pool_hf)
    pool_dsp["duplicate_status"] = [
        "shared_guide_cross_source" if (str(g) in shared)
        else "single" for g in pool_dsp["guide_sequence"]]
    pool_hf["duplicate_status"] = [
        "shared_guide_cross_source" if (str(g) in shared)
        else "single" for g in pool_hf["guide_sequence"]]

    pool = pd.concat([pool_dsp, pool_hf], ignore_index=True)
    if pool[NORMALIZED_SEQ_COL].duplicated().any():
        # exact duplicate sequences kept when the sources differ (independent
        # experiments) -- annotated, not silently dropped; within-source
        # duplications are already removed during loading.
        pass
    return pool


def pool_summary(pool: pd.DataFrame) -> dict:
    """Auditable summary of the diversified training pool."""
    out = {
        "n_total": int(len(pool)),
        "columns": POOL_COLUMNS,
        "per_source": label_distribution_per_source(pool),
        "duplicate_scan": duplicate_and_conflict_scan(pool),
        "label_harmonization": LABEL_HARMONIZATION,
        "geometry_transformation": GEOMETRY_TRANSFORMATION,
        "arms": {"B": BASELINE_ARM, "D1": PRIMARY_ARM},
        "split_policy": ("DeepSpCas9 85/15 seed-42 split (canonical). The "
                         "validation split is DeepSpCas9-only and is used "
                         "identically by both arms (CNN early stopping). "
                         "DeepHF rows are appended to the training split only."),
    }
    seq_len = pool["normalized_sequence"].str.len()
    out["sequence_lengths"] = {
        "n": int(len(seq_len)),
        "mode": int(seq_len.mode().iloc[0]),
        "all_equal_30": bool((seq_len == 30).all()),
    }
    return out


def write_pool_csv(pool: pd.DataFrame, path: str) -> str:
    pool.to_csv(path, index=False)
    return path