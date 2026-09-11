"""Phase 17D independence and exact-overlap audit.

Exact set intersections, duplicates, and provenance relationships only. No
fuzzy matching, near-duplicate thresholds, reverse-complement matching without
orientation evidence, model work, label transformation, pooling, or Moreno
access.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .compatibility import PHASE17B_RESULT
from .phase17a import protected_hashes


PHASE17C_RESULT = Path("results/phase17c_sequence_label_compatibility_20260911_144924.json")
RC_OVERLAP_NOT_ASSESSED = "RC_OVERLAP_NOT_ASSESSED"

PROVENANCE_STATUSES = {
    "INDEPENDENCE_ESTABLISHED",
    "DEPENDENCY_OR_SUBSET_ESTABLISHED",
    "SHARED_SOURCE_STUDY",
    "DERIVED_OR_PROCESSED_RELATIONSHIP",
    "INDEPENDENCE_UNRESOLVED",
    "INSUFFICIENT_PROVENANCE",
}


def normalize_dna(value: object) -> str:
    return str(value).strip().upper()


def is_dna(value: str) -> bool:
    return bool(value) and set(value.upper()) <= set("ACGT")


def exact_overlap(a: Iterable[str], b: Iterable[str]) -> Dict[str, object]:
    left = {normalize_dna(v) for v in a if normalize_dna(v)}
    right = {normalize_dna(v) for v in b if normalize_dna(v)}
    shared = left & right
    return {
        "unique_a": len(left),
        "unique_b": len(right),
        "shared_unique": len(shared),
        "shared_examples": sorted(shared)[:20],
    }


def duplicate_record_summary(records: Sequence[Dict[str, object]], fields: Sequence[str]) -> Dict[str, object]:
    identities = [tuple(str(record.get(field, "")) for field in fields) for record in records]
    counts = Counter(identities)
    return {
        "total_records": len(identities),
        "unique_record_identities": len(counts),
        "duplicate_record_excess": len(identities) - len(counts),
        "max_occurrences": max(counts.values()) if counts else 0,
    }


def duplicate_sequence_summary(records: Sequence[Dict[str, object]], key: str = "sequence") -> Dict[str, object]:
    values = [normalize_dna(record.get(key, "")) for record in records if normalize_dna(record.get(key, ""))]
    counts = Counter(values)
    return {
        "total_sequence_records": len(values),
        "unique_sequences": len(counts),
        "duplicate_sequence_excess": len(values) - len(counts),
        "duplicate_sequence_count": sum(1 for count in counts.values() if count > 1),
        "max_occurrences": max(counts.values()) if counts else 0,
    }


def label_conflicts(records: Sequence[Dict[str, object]], sequence_key: str = "sequence", label_key: str = "label") -> Dict[str, object]:
    labels_by_sequence: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        seq = normalize_dna(record.get(sequence_key, ""))
        if not seq or label_key not in record:
            continue
        labels_by_sequence[seq].add(str(record.get(label_key)))
    conflicts = {
        seq: sorted(labels)
        for seq, labels in labels_by_sequence.items()
        if len(labels) >= 2
    }
    return {
        "conflict_sequence_count": len(conflicts),
        "examples": [
            {"sequence": sequence, "labels": labels}
            for sequence, labels in list(conflicts.items())[:20]
        ],
    }


def rc_overlap_status(orientation_verified: bool) -> Dict[str, object]:
    if not orientation_verified:
        return {"status": RC_OVERLAP_NOT_ASSESSED}
    return {"status": "NOT_IMPLEMENTED_IN_PHASE17D"}


@lru_cache(maxsize=None)
def canonical_deepspcas9_reference(path: str = "data/raw/DeepSpCas9.csv") -> Dict[str, object]:
    df = pd.read_csv(path, usecols=["sequence_30mer"])
    seq30 = [normalize_dna(v) for v in df["sequence_30mer"]]
    guides = [seq[4:24] for seq in seq30 if len(seq) == 30]
    valid_seq30 = [seq for seq in seq30 if len(seq) == 30 and is_dna(seq)]
    valid_guides = [seq[4:24] for seq in valid_seq30]
    return {
        "distribution_population": {
            "n_rows": int(len(df)),
            "unique_30mer": len(set(seq30)),
            "unique_guides": len(set(guides)),
        },
        "canonical_valid_modeling_population": {
            "n_rows": len(valid_seq30),
            "note": "valid length/ACGT proxy; project canonical modeling valid n is documented separately as 10,117",
            "project_documented_valid_n": 10117,
            "train_n": 8599,
            "validation_n": 1518,
            "unique_30mer": len(set(valid_seq30)),
            "unique_guides": len(set(valid_guides)),
        },
        "seq30": valid_seq30,
        "guides": valid_guides,
        "all_seq30": seq30,
        "all_guides": guides,
    }


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_crisprpred_file_records(file_record: Dict[str, object]) -> List[Dict[str, object]]:
    df = pd.read_csv(file_record["exact_local_filename"])
    records = []
    for idx, row in df.iterrows():
        seq = normalize_dna(row.get("sgRNA", ""))
        if not is_dna(seq):
            continue
        records.append({
            "candidate_id": "crisprpredseq_2020_bmc_additional_files",
            "file": file_record["exact_local_filename"],
            "row_index": int(idx),
            "sequence": seq,
            "guide": seq[:20] if len(seq) >= 20 else "",
            "geometry": "GUIDE_PLUS_PAM_23MER",
            "label": row.get("label"),
            "label_key": str(row.get("label")),
        })
    return records


def _headered_sheet(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.empty:
        return pd.DataFrame()
    header_idx = 0
    for idx in range(min(8, len(raw))):
        values = [str(value) for value in raw.iloc[idx].tolist()]
        if any(value in {"sgRNA", "Perturbations", "Most enriched perturbation"} for value in values):
            header_idx = idx
            break
    return pd.read_excel(path, sheet_name=sheet, header=header_idx).dropna(how="all")


def extract_doench_file_records(file_record: Dict[str, object]) -> List[Dict[str, object]]:
    """Extract only DNA-looking guide records from approved sequence fields."""
    path = Path(file_record["exact_local_filename"])
    workbook = pd.ExcelFile(path)
    records = []
    for sheet in workbook.sheet_names:
        frame = _headered_sheet(path, sheet)
        if frame.empty:
            continue
        if "sgRNA" in frame.columns:
            for idx, row in frame.iterrows():
                seq = normalize_dna(row.get("sgRNA", ""))
                if is_dna(seq) and len(seq) >= 20:
                    records.append({
                        "candidate_id": "doench_2016_orcs_publication_screens",
                        "file": file_record["exact_local_filename"],
                        "sheet": sheet,
                        "row_index": int(idx),
                        "sequence": seq,
                        "guide": seq if len(seq) == 20 else seq[:20],
                        "geometry": "GUIDE_ONLY",
                        "label": row.get("Average log fold change IFNgamma - mock"),
                        "label_key": str(row.get("Average log fold change IFNgamma - mock")) if "Average log fold change IFNgamma - mock" in frame.columns else "",
                    })
        for col in ["Perturbations", "Most enriched perturbation"]:
            if col not in frame.columns:
                continue
            for idx, value in frame[col].dropna().items():
                for raw_seq in str(value).split(";"):
                    seq = normalize_dna(raw_seq)
                    if is_dna(seq) and len(seq) >= 20:
                        records.append({
                            "candidate_id": "doench_2016_orcs_publication_screens",
                            "file": file_record["exact_local_filename"],
                            "sheet": sheet,
                            "row_index": int(idx),
                            "sequence": seq,
                            "guide": seq if len(seq) == 20 else seq[:20],
                            "geometry": "GUIDE_ONLY",
                            "label_key": "",
                        })
    return records


def candidate_records_from_phase17b(phase17b: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    records: Dict[str, List[Dict[str, object]]] = {}
    for candidate in phase17b["candidates"]:
        cid = candidate["candidate_id"]
        out: List[Dict[str, object]] = []
        if cid == "crisprpredseq_2020_bmc_additional_files":
            for file_record in candidate["files"]:
                out.extend(extract_crisprpred_file_records(file_record))
        elif cid == "doench_2016_orcs_publication_screens":
            for file_record in candidate["files"]:
                out.extend(extract_doench_file_records(file_record))
        records[cid] = out
    return records


def _provenance_status(candidate_id: str) -> Dict[str, str]:
    if candidate_id == "crisprpredseq_2020_bmc_additional_files":
        return {
            "status": "DERIVED_OR_PROCESSED_RELATIONSHIP",
            "reason": "Phase 17B records author-provided processed/binary supplementary files with unresolved DeepCRISPR/Haeussler source relationship.",
        }
    if candidate_id == "doench_2016_orcs_publication_screens":
        return {
            "status": "SHARED_SOURCE_STUDY",
            "reason": "Recovered workbooks are multiple publication/ORCS screen outputs from the same Doench study.",
        }
    if candidate_id == "crispron_2021_rth_tools":
        return {
            "status": "INSUFFICIENT_PROVENANCE",
            "reason": "Repository archive did not identify a secure experimental sequence table.",
        }
    return {
        "status": "INSUFFICIENT_PROVENANCE",
        "reason": "No verified primary sequence table recovered.",
    }


def _file_reports(candidate_id: str, records: List[Dict[str, object]], canonical: Dict[str, object]) -> List[Dict[str, object]]:
    by_file: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_file[record["file"]].append(record)
    reports = []
    for file_path, file_records in sorted(by_file.items()):
        seqs = [record["sequence"] for record in file_records]
        guides = [record["guide"] for record in file_records if record.get("guide")]
        report = {
            "file": file_path,
            "record_count": len(file_records),
            "geometry": sorted({record.get("geometry", "UNKNOWN") for record in file_records}),
            "exact_sequence_overlap_with_canonical_valid_30mer": exact_overlap(
                [seq for seq in seqs if len(seq) == 30],
                canonical["seq30"],
            ) if any(len(seq) == 30 for seq in seqs) else "NO_CANONICAL_30MER_GEOMETRY",
            "exact_guide_overlap_with_canonical_valid_guides": exact_overlap(guides, canonical["guides"]) if guides else "NO_VALID_GUIDE_FIELD",
            "duplicate_rows": duplicate_record_summary(file_records, ["sequence", "guide", "label_key", "sheet"]),
            "duplicate_exact_sequences": duplicate_sequence_summary(file_records),
            "duplicate_sequence_label_combinations": duplicate_record_summary(file_records, ["sequence", "label_key"]),
            "identical_sequence_label_conflicts": label_conflicts(file_records, label_key="label_key"),
            "rc_overlap_status": rc_overlap_status(orientation_verified=False),
        }
        reports.append(report)
    return reports


def _candidate_report(candidate_id: str, records: List[Dict[str, object]], canonical: Dict[str, object]) -> Dict[str, object]:
    seqs = [record["sequence"] for record in records]
    guides = [record["guide"] for record in records if record.get("guide")]
    if not records:
        return {
            "candidate_id": candidate_id,
            "sequence_availability": "INSUFFICIENT_SEQUENCE_INFORMATION",
            "geometry": "NO_VERIFIED_SEQUENCE_TABLE",
            "exact_sequence_overlap_with_canonical_valid_30mer": "INSUFFICIENT_SEQUENCE_INFORMATION",
            "exact_guide_overlap_with_canonical_valid_guides": "INSUFFICIENT_SEQUENCE_INFORMATION",
            "internal_duplicate_rows": duplicate_record_summary([], ["sequence"]),
            "duplicate_exact_sequences": duplicate_sequence_summary([]),
            "duplicate_sequence_label_combinations": duplicate_record_summary([], ["sequence", "label_key"]),
            "label_conflict_sequences": label_conflicts([], label_key="label_key"),
            "orientation_status": rc_overlap_status(orientation_verified=False),
            "provenance_relationship": _provenance_status(candidate_id),
            "file_reports": [],
        }
    return {
        "candidate_id": candidate_id,
        "sequence_availability": "VERIFIED_OBSERVED_SEQUENCE_OR_GUIDE_FIELD",
        "geometry": sorted({record.get("geometry", "UNKNOWN") for record in records}),
        "exact_sequence_overlap_with_canonical_valid_30mer": exact_overlap(
            [seq for seq in seqs if len(seq) == 30],
            canonical["seq30"],
        ) if any(len(seq) == 30 for seq in seqs) else "NO_CANONICAL_30MER_GEOMETRY",
        "exact_guide_overlap_with_canonical_valid_guides": exact_overlap(guides, canonical["guides"]) if guides else "NO_VALID_GUIDE_FIELD",
        "internal_duplicate_rows": duplicate_record_summary(records, ["sequence", "guide", "label_key", "sheet"]),
        "duplicate_exact_sequences": duplicate_sequence_summary(records),
        "duplicate_sequence_label_combinations": duplicate_record_summary(records, ["sequence", "label_key"]),
        "label_conflict_sequences": label_conflicts(records, label_key="label_key"),
        "orientation_status": rc_overlap_status(orientation_verified=False),
        "provenance_relationship": _provenance_status(candidate_id),
        "file_reports": _file_reports(candidate_id, records, canonical),
    }


def _pairwise_overlap(records_by_candidate: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    ids = sorted(records_by_candidate)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            left_records = records_by_candidate[left]
            right_records = records_by_candidate[right]
            key = f"{left}_vs_{right}"
            if not left_records or not right_records:
                out[key] = {
                    "sequence_overlap": "INSUFFICIENT_SEQUENCE_INFORMATION",
                    "guide_overlap": "INSUFFICIENT_SEQUENCE_INFORMATION",
                    "provenance_relationship": pairwise_provenance(left, right),
                }
                continue
            left_seqs = [record["sequence"] for record in left_records]
            right_seqs = [record["sequence"] for record in right_records]
            left_guides = [record["guide"] for record in left_records if record.get("guide")]
            right_guides = [record["guide"] for record in right_records if record.get("guide")]
            out[key] = {
                "sequence_overlap": exact_overlap(left_seqs, right_seqs),
                "guide_overlap": exact_overlap(left_guides, right_guides) if left_guides and right_guides else "INSUFFICIENT_SEQUENCE_INFORMATION",
                "provenance_relationship": pairwise_provenance(left, right),
            }
    return out


def pairwise_provenance(left: str, right: str) -> Dict[str, str]:
    if left == "doench_2016_orcs_publication_screens" and right == "doench_2016_orcs_publication_screens":
        return {"status": "SHARED_SOURCE_STUDY", "reason": "same candidate"}
    if {left, right} <= {"crisprpredseq_2020_bmc_additional_files", "doench_2016_orcs_publication_screens"}:
        return {
            "status": "INDEPENDENCE_UNRESOLVED",
            "reason": "Both expose guide fields, but Phase 17D does not establish independence from file/source names.",
        }
    return {
        "status": "INSUFFICIENT_PROVENANCE",
        "reason": "At least one side lacks verified primary sequence information.",
    }


def build_phase17d_independence_overlap(
    phase17b_path: Path = PHASE17B_RESULT,
    phase17c_path: Path = PHASE17C_RESULT,
) -> Dict[str, object]:
    before = protected_hashes()
    phase17b = load_json(phase17b_path)
    phase17c = load_json(phase17c_path)
    canonical = canonical_deepspcas9_reference()
    records_by_candidate = candidate_records_from_phase17b(phase17b)
    # Ensure candidates without sequence records are still represented.
    for candidate in phase17c["candidates"]:
        records_by_candidate.setdefault(candidate["candidate_id"], [])
    candidate_reports = [
        _candidate_report(candidate_id, records_by_candidate[candidate_id], canonical)
        for candidate_id in sorted(records_by_candidate)
    ]
    after = protected_hashes()
    return {
        "phase": "17D",
        "protocol_version": "phase17d-independence-overlap-v1",
        "scope": "independence_exact_overlap_duplicate_audit_only_no_acceptance",
        "phase17b_source": str(phase17b_path).replace("\\", "/"),
        "phase17c_source": str(phase17c_path).replace("\\", "/"),
        "canonical_reference": {
            "dataset": "DeepSpCas9",
            "source": "data/raw/DeepSpCas9.csv",
            "distribution_population_n": 12832,
            "canonical_modeling_valid_n": 10117,
            "train_n": 8599,
            "validation_n": 1518,
            "loaded_reference_summary": {
                "distribution_unique_30mer": canonical["distribution_population"]["unique_30mer"],
                "distribution_unique_guides": canonical["distribution_population"]["unique_guides"],
                "valid_proxy_unique_30mer": canonical["canonical_valid_modeling_population"]["unique_30mer"],
                "valid_proxy_unique_guides": canonical["canonical_valid_modeling_population"]["unique_guides"],
            },
        },
        "candidate_reports": candidate_reports,
        "candidate_to_candidate_overlap": _pairwise_overlap(records_by_candidate),
        "interpretation_safeguards": {
            "exact_overlap_not_called_leakage": True,
            "zero_overlap_not_called_independence": True,
            "no_near_duplicate_analysis": True,
            "rc_matching_not_performed_without_orientation": True,
        },
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_moreno_raw_data_accessed": True,
            "no_dataset_pooling": True,
            "no_training_dataset_constructed": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_sequence_transformation": True,
            "no_near_duplicate_analysis": True,
            "no_distance_based_matching": True,
            "no_rc_overlap_without_orientation": True,
            "no_continual_learning": True,
            "no_dataset_acceptance": True,
        },
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
