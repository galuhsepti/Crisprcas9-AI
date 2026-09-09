"""Phase 16D exact overlap and dataset-independence audit.

Only exact sequence/guide and provenance relationships are considered. The
module intentionally performs no sequence transformation, no dataset pooling,
no distribution analysis, no Moreno access, and no model import.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .compatibility import build_phase16c_compatibility

RC_OVERLAP_NOT_ASSESSED = "RC_OVERLAP_NOT_ASSESSED"


def normalize_dna(value: object) -> str:
    """Uppercase string with whitespace stripped."""
    return str(value).strip().upper()


def _is_dna_sequence(value: str) -> bool:
    return bool(value) and set(value) <= set("ACGT")


def exact_overlap(a: Iterable[str], b: Iterable[str]) -> Dict[str, object]:
    """Exact set overlap with denominators."""
    sa = {normalize_dna(x) for x in a if normalize_dna(x)}
    sb = {normalize_dna(x) for x in b if normalize_dna(x)}
    shared = sa & sb
    union = sa | sb
    return {
        "n_a": len(sa),
        "n_b": len(sb),
        "shared_unique": len(shared),
        "fraction_of_a": len(shared) / len(sa) if sa else None,
        "fraction_of_b": len(shared) / len(sb) if sb else None,
        "jaccard": len(shared) / len(union) if union else None,
    }


def duplicate_counts(values: Sequence[str]) -> Dict[str, object]:
    """Exact duplicate summary for observed values."""
    normalized = [normalize_dna(v) for v in values if normalize_dna(v)]
    counts = Counter(normalized)
    return {
        "n_records_with_sequence": len(normalized),
        "n_unique_sequences": len(counts),
        "duplicate_sequence_count": sum(1 for c in counts.values() if c > 1),
        "duplicate_record_excess": sum(c - 1 for c in counts.values() if c > 1),
        "max_occurrences": max(counts.values()) if counts else 0,
    }


def label_conflicts(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Detect exact identical-sequence label conflicts where labels exist."""
    by_seq: Dict[str, set] = defaultdict(set)
    for record in records:
        seq = normalize_dna(record.get("sequence", ""))
        if not seq or "label" not in record:
            continue
        by_seq[seq].add(str(record["label"]))
    conflicts = {
        seq: sorted(labels) for seq, labels in by_seq.items()
        if len(labels) > 1
    }
    return {
        "n_conflicting_sequences": len(conflicts),
        "examples": [
            {"sequence": seq, "labels": labels}
            for seq, labels in list(conflicts.items())[:20]
        ],
    }


def reverse_complement_overlap(a, b, orientation_verified: bool) -> Dict[str, object]:
    """RC overlap is only legal when orientation is verified."""
    if not orientation_verified:
        return {"status": RC_OVERLAP_NOT_ASSESSED}
    comp = str.maketrans("ACGT", "TGCA")
    rc_b = [normalize_dna(s).translate(comp)[::-1] for s in b]
    return {"status": "ASSESSED", **exact_overlap(a, rc_b)}


def canonical_deepspcas9_sets(path: str = "data/raw/DeepSpCas9.csv") -> Dict[str, object]:
    """Load canonical DeepSpCas9 reference sequences for exact overlap only."""
    df = pd.read_csv(path, usecols=["sequence_30mer"])
    seq30 = [normalize_dna(s) for s in df["sequence_30mer"]]
    guides = [s[4:24] for s in seq30 if len(s) == 30]
    return {
        "n_rows": len(df),
        "seq30": seq30,
        "guides": guides,
    }


@lru_cache(maxsize=None)
def extract_crisprpredseq_records(path: str) -> List[Dict[str, object]]:
    """Extract actual 23-mer sgRNA records without transforming to 30-mer."""
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        seq = normalize_dna(row["sgRNA"])
        out.append({
            "sequence": seq,
            "guide": seq[:20] if len(seq) >= 20 else "",
            "sequence_kind": "guide_plus_pam",
            "label": row.get("label"),
        })
    return out


def _headers_from_first_row(path: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.empty:
        return pd.DataFrame()
    header_idx = 0
    for i in range(min(5, len(raw))):
        vals = [str(v) for v in raw.iloc[i].tolist()]
        if any("Perturbations" == v or "sgRNA" == v for v in vals):
            header_idx = i
            break
    frame = pd.read_excel(path, sheet_name=sheet, header=header_idx)
    return frame.dropna(how="all")


@lru_cache(maxsize=None)
def extract_doench_records(path: str) -> List[Dict[str, object]]:
    """Extract observed guide-only records from Doench supplementary workbooks."""
    workbook = pd.ExcelFile(path)
    records = []
    for sheet in workbook.sheet_names:
        frame = _headers_from_first_row(path, sheet)
        if frame.empty:
            continue
        if "sgRNA" in frame.columns:
            for _, row in frame.iterrows():
                seq = normalize_dna(row.get("sgRNA", ""))
                if _is_dna_sequence(seq):
                    records.append({
                        "sequence": seq,
                        "guide": seq if len(seq) == 20 else seq[:20],
                        "sequence_kind": "guide_only",
                        "label": row.get("Average log fold change IFNgamma - mock"),
                        "sheet": sheet,
                    })
        for col in ["Perturbations", "Most enriched perturbation"]:
            if col not in frame.columns:
                continue
            for value in frame[col].dropna():
                for seq in str(value).split(";"):
                    seq = normalize_dna(seq)
                    if _is_dna_sequence(seq) and len(seq) >= 20:
                        records.append({
                            "sequence": seq,
                            "guide": seq if len(seq) == 20 else seq[:20],
                            "sequence_kind": "guide_only",
                            "sheet": sheet,
                        })
    return records


def candidate_sequence_records(phase16c_payload: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    """Extract observed candidate sequences from files eligible for 16D audit."""
    out: Dict[str, List[Dict[str, object]]] = {}
    for candidate in phase16c_payload["records"]:
        cid = candidate["candidate_id"]
        records: List[Dict[str, object]] = []
        for file_result in candidate.get("files", []):
            path = file_result["file"]
            if cid == "crisprpredseq_2020_bmc_additional_files":
                records.extend(extract_crisprpredseq_records(path))
            elif cid == "doench_2016_orcs_publication_screens":
                records.extend(extract_doench_records(path))
        out[cid] = records
    return out


def provenance_independence_status(candidate_id: str) -> Dict[str, object]:
    """Conservative independence assessment from provenance only."""
    if candidate_id == "doench_2016_orcs_publication_screens":
        return {
            "independence_assessment": "NOT_ESTABLISHED",
            "reason": "Same original study across multiple files; prior project records indicate possible canonical overlap requiring exact audit.",
        }
    if candidate_id == "crisprpredseq_2020_bmc_additional_files":
        return {
            "independence_assessment": "NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK",
            "reason": "Phase 16B/16C provenance reports relationship to DeepCRISPR/Haeussler-used processed datasets.",
        }
    return {
        "independence_assessment": "INSUFFICIENT_SEQUENCE_INFORMATION",
        "reason": "No verified sequence file was available for Phase 16D exact audit.",
    }


def build_phase16d_overlap_audit(access_date: str | None = None) -> Dict[str, object]:
    """Build deterministic Phase 16D exact overlap/independence audit."""
    phase16c = build_phase16c_compatibility(access_date)
    canonical = canonical_deepspcas9_sets()
    records_by_candidate = candidate_sequence_records(phase16c)
    candidate_reports = []
    for cid in sorted(records_by_candidate):
        recs = records_by_candidate[cid]
        seqs = [r["sequence"] for r in recs]
        guides = [r["guide"] for r in recs if r.get("guide")]
        seq30 = [s for s in seqs if len(s) == 30]
        report = {
            "candidate_id": cid,
            "sequence_available": bool(recs),
            "sequence_record_count": len(recs),
            "sequence_kinds": sorted({r.get("sequence_kind", "UNKNOWN") for r in recs}),
            "exact_30mer_vs_deepspcas9": exact_overlap(seq30, canonical["seq30"]) if seq30 else "NO_30MER_SEQUENCE_AVAILABLE",
            "exact_guide_vs_deepspcas9": exact_overlap(guides, canonical["guides"]) if guides else "NO_GUIDE_SEQUENCE_AVAILABLE",
            "duplicate_records": duplicate_counts(seqs),
            "duplicate_sequences": duplicate_counts(seqs),
            "identical_sequence_label_conflicts": label_conflicts(recs),
            "rc_overlap_status": reverse_complement_overlap(seqs, canonical["seq30"], orientation_verified=False),
            "provenance_relationship": provenance_independence_status(cid),
            "compatibility_status_from_16c": next(
                r["compatibility_status"] for r in phase16c["records"]
                if r["candidate_id"] == cid
            ),
            "limitations": [
                "Incompatible labels are not rescued or reinterpreted.",
                "No sequence transformation was performed.",
                "RC overlap was not assessed because orientation was not verified.",
            ],
        }
        candidate_reports.append(report)

    pairwise = {}
    ids = sorted(records_by_candidate)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            lrecs = records_by_candidate[left]
            rrecs = records_by_candidate[right]
            l30 = [r["sequence"] for r in lrecs if len(r["sequence"]) == 30]
            r30 = [r["sequence"] for r in rrecs if len(r["sequence"]) == 30]
            lg = [r["guide"] for r in lrecs if r.get("guide")]
            rg = [r["guide"] for r in rrecs if r.get("guide")]
            pairwise[f"{left}_vs_{right}"] = {
                "exact_30mer": exact_overlap(l30, r30) if l30 and r30 else "NO_30MER_SEQUENCE_AVAILABLE",
                "exact_guide": exact_overlap(lg, rg) if lg and rg else "NO_GUIDE_SEQUENCE_AVAILABLE",
                "rc_overlap_status": {"status": RC_OVERLAP_NOT_ASSESSED},
            }

    return {
        "phase": "16D",
        "scope": "exact_overlap_and_independence_only",
        "status": "OVERLAP_AUDIT_RECORDED",
        "canonical_reference": {
            "dataset": "DeepSpCas9",
            "source": "data/raw/DeepSpCas9.csv",
            "n_rows": canonical["n_rows"],
            "n_unique_30mer": len(set(canonical["seq30"])),
            "n_unique_guides": len(set(canonical["guides"])),
        },
        "candidate_reports": candidate_reports,
        "pairwise_candidate_overlap": pairwise,
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_dataset_pooling": True,
            "no_label_rescue": True,
            "no_sequence_transformation": True,
            "no_distribution_analysis": True,
            "no_moreno_raw_data_accessed": True,
        },
    }
