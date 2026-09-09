"""Phase 16E conservative distribution/composition audit.

Descriptive only: no modeling, label transformation, pooling, overlap rerun,
or gate decision. Moreno-Mateos is not accessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.audit.audit import DEFAULT_ACTIVITY_EDGES
from src.bioinformatics.kmer import extract_kmer_matrix, generate_kmers
from src.diagnostics.sequence_analysis import (
    ambiguous_character_summary,
    duplicate_summary,
    gc_content_array,
    nucleotide_frequency_array,
    sequence_length_summary,
)
from src.diagnostics.stats import descriptive_stats, ecdf_data, histogram_data

from .overlap import build_phase16d_overlap_audit, candidate_sequence_records, normalize_dna
from .compatibility import build_phase16c_compatibility


def canonical_deepspcas9_distribution(path: str = "data/raw/DeepSpCas9.csv") -> Dict[str, object]:
    """Distribution summary for the canonical DeepSpCas9 reference."""
    df = pd.read_csv(path, usecols=["sequence_30mer", "activity"])
    seqs = [normalize_dna(s) for s in df["sequence_30mer"]]
    labels = df["activity"].astype(float).to_numpy()
    guides = [s[4:24] for s in seqs if len(s) == 30]
    return {
        "dataset": "DeepSpCas9",
        "record_counts": {
            "raw_records": int(len(df)),
            "sequence_available": True,
            "activity_label_available": True,
            "activity_label_analyzed": True,
        },
        "sequence": sequence_distribution(seqs, guides=guides, positional=True, kmer=True),
        "activity": activity_distribution(labels),
    }


def sequence_distribution(
    sequences: Sequence[str],
    guides: Sequence[str] | None = None,
    positional: bool = False,
    kmer: bool = True,
) -> Dict[str, object]:
    """Descriptive sequence composition over observed sequences."""
    seqs = [normalize_dna(s) for s in sequences if _is_dna_sequence(normalize_dna(s))]
    if not seqs:
        return {"status": "NO_SEQUENCE_AVAILABLE"}
    gc = gc_content_array(seqs)
    nuc = nucleotide_frequency_array(seqs).mean(axis=0)
    out: Dict[str, object] = {
        "n_sequences": int(len(seqs)),
        "length": sequence_length_summary(seqs),
        "ambiguous_characters": ambiguous_character_summary(seqs),
        "duplicates": duplicate_summary(seqs),
        "gc": descriptive_stats(gc),
        "gc_histogram": histogram_data(gc, bins=20, density=False),
        "gc_ecdf": ecdf_data(gc),
        "nucleotide_composition": {
            "A": float(nuc[0]),
            "C": float(nuc[1]),
            "G": float(nuc[2]),
            "T": float(nuc[3]),
        },
    }
    if guides:
        guide_list = [normalize_dna(g) for g in guides if normalize_dna(g)]
        if guide_list:
            out["guide_gc"] = descriptive_stats(gc_content_array(guide_list))
        else:
            out["guide_gc"] = "NO_GUIDE_SEQUENCE_AVAILABLE"
    else:
        out["guide_gc"] = "NOT_ASSESSED_GUIDE_BOUNDARIES_UNVERIFIED"
    if positional and len({len(s) for s in seqs}) == 1:
        out["positional_nucleotide_composition"] = positional_counts(seqs)
    else:
        out["positional_nucleotide_composition"] = "NOT_ASSESSED_GEOMETRY_OR_ORIENTATION_UNVERIFIED"
    if kmer:
        out["kmer_2"] = kmer_summary(seqs, 2)
        out["kmer_3"] = kmer_summary(seqs, 3)
    return out


def activity_distribution(labels: Sequence[float]) -> Dict[str, object]:
    """Canonical activity distribution for established continuous labels."""
    values = np.asarray(labels, dtype=float).ravel()
    bins = fixed_activity_bins(values)
    return {
        "status": "ANALYZED_ESTABLISHED_CONTINUOUS_ACTIVITY",
        "descriptive": descriptive_stats(values),
        "histogram": histogram_data(values, bins=20, density=False),
        "ecdf": ecdf_data(values),
        "fixed_activity_bins": bins,
        "low_activity_<=0.05": int(np.sum(values <= 0.05)),
        "high_activity_>0.8": int(np.sum(values > 0.8)),
    }


def fixed_activity_bins(values: Sequence[float]) -> List[Dict[str, object]]:
    """Counts on the canonical fixed activity grid."""
    y = np.asarray(values, dtype=float).ravel()
    out = []
    edges = DEFAULT_ACTIVITY_EDGES
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (y >= lo) & (y <= hi) if i == len(edges) - 2 else (y >= lo) & (y < hi)
        out.append({
            "bin": f"[{lo:.2f}, {hi:.2f}{']' if i == len(edges) - 2 else ')'}",
            "n": int(np.sum(mask)),
            "fraction": float(np.mean(mask)) if y.size else None,
        })
    return out


def positional_counts(sequences: Sequence[str]) -> Dict[str, List[float]]:
    """Position-by-position nucleotide fractions within a verified coordinate system."""
    length = len(sequences[0])
    totals = {n: np.zeros(length, dtype=float) for n in "ACGT"}
    for seq in sequences:
        for i, nuc in enumerate(seq):
            if nuc in totals:
                totals[nuc][i] += 1
    return {n: (counts / len(sequences)).tolist() for n, counts in totals.items()}


def kmer_summary(sequences: Sequence[str], k: int) -> Dict[str, object]:
    """Mean exact k-mer composition over observed sequence geometry."""
    valid = [s for s in sequences if _is_dna_sequence(s) and len(s) >= k]
    if not valid:
        return {"status": "NOT_ASSESSED_NO_VALID_SEQUENCE", "k": k}
    matrix = extract_kmer_matrix(valid, k)
    means = matrix.mean(axis=0)
    kmers = generate_kmers(k)
    order = np.argsort(means)[::-1]
    return {
        "k": k,
        "geometry": "observed_sequence_only",
        "top_kmers": [
            {"kmer": kmers[i], "mean_frequency": float(means[i])}
            for i in order[:10]
        ],
        "mean_frequencies": {kmers[i]: float(means[i]) for i in range(len(kmers))},
    }


def compare_gc_to_canonical(candidate_gc: Sequence[float], canonical_gc: Sequence[float]) -> Dict[str, object]:
    """Descriptive GC comparison against canonical reference."""
    c = np.asarray(candidate_gc, dtype=float)
    ref = np.asarray(canonical_gc, dtype=float)
    if c.size == 0 or ref.size == 0:
        return {"status": "NOT_ASSESSED"}
    return {
        "status": "DESCRIPTIVE_ONLY",
        "mean_delta_candidate_minus_canonical": float(np.mean(c) - np.mean(ref)),
        "median_delta_candidate_minus_canonical": float(np.median(c) - np.median(ref)),
        "iqr_candidate": float(np.percentile(c, 75) - np.percentile(c, 25)),
        "iqr_canonical": float(np.percentile(ref, 75) - np.percentile(ref, 25)),
    }


def build_phase16e_distribution_audit(access_date: str | None = None) -> Dict[str, object]:
    """Build Phase 16E distribution audit from Phase 16C/16D state."""
    overlap = build_phase16d_overlap_audit(access_date)
    records = candidate_sequence_records(build_phase16c_compatibility(access_date))
    canonical = canonical_deepspcas9_distribution()
    canonical_gc = gc_content_array(pd.read_csv("data/raw/DeepSpCas9.csv", usecols=["sequence_30mer"])["sequence_30mer"].astype(str).tolist())
    candidate_reports = []
    by_overlap = {r["candidate_id"]: r for r in overlap["candidate_reports"]}
    for cid in sorted(records):
        recs = records[cid]
        seqs = [r["sequence"] for r in recs if _is_dna_sequence(r["sequence"])]
        guides = [r["guide"] for r in recs if r.get("guide") and _is_dna_sequence(r["guide"])]
        kinds = sorted({r.get("sequence_kind", "UNKNOWN") for r in recs})
        guide_boundaries_verified = kinds == ["guide_only"]
        seq_report = sequence_distribution(
            seqs,
            guides=guides if guide_boundaries_verified else None,
            positional=False,
            kmer=bool(seqs),
        )
        if isinstance(seq_report, dict) and seq_report.get("status") != "NO_SEQUENCE_AVAILABLE":
            gc_values = gc_content_array(seqs)
            gc_compare = compare_gc_to_canonical(gc_values, canonical_gc)
        else:
            gc_compare = {"status": "NOT_ASSESSED"}
        candidate_reports.append({
            "candidate_id": cid,
            "record_counts": {
                "sequence_records": len(recs),
                "sequence_available": bool(recs),
                "usable_for_sequence_distribution": bool(recs),
                "activity_label_analyzed": False,
            },
            "compatibility_status_from_16c": by_overlap[cid]["compatibility_status_from_16c"],
            "provenance_independence_from_16d": by_overlap[cid]["provenance_relationship"],
            "overlap_context_from_16d": {
                "exact_30mer_vs_deepspcas9": by_overlap[cid]["exact_30mer_vs_deepspcas9"],
                "exact_guide_vs_deepspcas9": by_overlap[cid]["exact_guide_vs_deepspcas9"],
                "rc_overlap_status": by_overlap[cid]["rc_overlap_status"],
            },
            "sequence": seq_report,
            "activity": {
                "status": "NOT_ANALYZED_LABEL_SEMANTICS_NOT_ESTABLISHED_AS_CONTINUOUS_ACTIVITY",
                "reason": "Phase 16C did not establish target-compatible continuous activity for this candidate.",
            },
            "descriptive_gc_comparison_to_deepspcas9": gc_compare,
            "limitations": [
                "Observed-sequence composition only; no sequence transformation.",
                "No positional cross-dataset comparison because candidate orientation/coordinates are unresolved.",
                "No generalization or superiority claim.",
            ],
        })
    return {
        "phase": "16E",
        "scope": "distribution_composition_audit_only",
        "status": "DISTRIBUTION_AUDIT_RECORDED",
        "canonical_reference": canonical,
        "candidate_reports": candidate_reports,
        "phase16d_context": {
            "candidate_reports": overlap["candidate_reports"],
            "pairwise_candidate_overlap": overlap["pairwise_candidate_overlap"],
        },
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_label_transformation": True,
            "no_dataset_pooling": True,
            "no_data_augmentation": True,
            "no_calibration": True,
            "no_domain_adaptation": True,
            "no_gate_decision": True,
            "no_moreno_raw_data_accessed": True,
        },
    }


def _is_dna_sequence(value: str) -> bool:
    """Accept only observed A/C/G/T strings for composition summaries."""
    return bool(value) and set(value) <= set("ACGT")
