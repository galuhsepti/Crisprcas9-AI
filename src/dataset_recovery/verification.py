"""Phase 17B primary-file verification audit.

This module verifies file identity, primary/derived status, experimental
context, sequence structure, and label structure for Phase 17A candidates. It
does not perform compatibility/acceptance decisions, overlap analysis, model
work, pooling, or transformations.
"""

from __future__ import annotations

import json
import math
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .phase17a import protected_hashes, sha256_file


ALLOWED_VERIFICATION_STATUSES = {
    "PRIMARY_VERIFIED",
    "PRIMARY_VERIFIED_WITH_METADATA_GAPS",
    "PRIMARY_STATUS_UNRESOLVED",
    "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED",
    "PRIMARY_FILE_NOT_RECOVERED",
    "DERIVED_OR_PROCESSED",
    "DISCOVERY_ONLY",
    "CLEARLY_INCOMPATIBLE",
}

PHASE17A_RESULT = Path("results/phase17a_dataset_recovery_20260911_140154.json")


def load_phase17a_payload(path: Path = PHASE17A_RESULT) -> Dict[str, object]:
    """Load the Phase 17A recovery payload used as 17B input."""
    return json.loads(path.read_text(encoding="utf-8"))


def _is_dna(value: str) -> bool:
    return bool(value) and set(value.upper()) <= set("ACGT")


def _sequence_summary(values: Iterable[object]) -> Dict[str, object]:
    normalized = [str(v).strip().upper() for v in values if pd.notna(v) and str(v).strip()]
    lengths = Counter(len(v) for v in normalized)
    non_acgt = sum(1 for value in normalized if not _is_dna(value))
    return {
        "observed_record_count": len(normalized),
        "missing_sequence_count": None,
        "observed_lengths": dict(sorted(lengths.items())),
        "non_acgt_count": non_acgt,
        "orientation_documented": "UNKNOWN",
        "includes_pam": "UNKNOWN",
        "includes_flanking_nucleotides": "UNKNOWN",
        "geometry_assessment": "RAW_STRUCTURE_ONLY_NO_COMPATIBILITY_DECISION",
    }


def _numeric_summary(series: pd.Series) -> Dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.dropna()
    if values.empty:
        return {"numeric": False}
    return {
        "numeric": True,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "non_numeric_count": int(len(series) - len(values)),
    }


def _unique_summary(series: pd.Series, limit: int = 10) -> Dict[str, object]:
    counts = series.dropna().astype(str).value_counts()
    return {
        "unique_count": int(counts.shape[0]),
        "examples": counts.head(limit).to_dict(),
    }


def inspect_csv_file(path: Path) -> Dict[str, object]:
    """Inspect CSV structure only."""
    df = pd.read_csv(path)
    columns = [str(c) for c in df.columns]
    sequence_columns = [
        col for col in columns
        if any(token in col.lower() for token in ["seq", "sgrna", "guide"])
    ]
    label_columns = [
        col for col in columns
        if any(token in col.lower() for token in ["label", "activity", "score", "indel", "freq", "rate", "rank", "fold"])
    ]
    sequence = {}
    for col in sequence_columns:
        summary = _sequence_summary(df[col])
        summary["missing_sequence_count"] = int(df[col].isna().sum())
        sequence[col] = summary
    labels = {}
    for col in label_columns:
        series = df[col]
        labels[col] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_summary": _unique_summary(series),
            "numeric_summary": _numeric_summary(series),
            "documented_unit": "UNKNOWN",
            "documented_biological_meaning": "UNKNOWN_FROM_FILE_ONLY",
            "experimentally_measured_explicit": "UNKNOWN",
            "label_type_observed": "binary" if set(series.dropna().astype(str)) <= {"0", "1"} else "unknown_or_continuous_numeric",
        }
    return {
        "file_format": "csv",
        "row_count": int(len(df)),
        "column_names": columns,
        "missing_cells_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "sequence_columns": sequence,
        "label_columns": labels,
    }


def _headers_from_first_row(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.empty:
        return pd.DataFrame()
    header_idx = 0
    for idx in range(min(8, len(raw))):
        values = [str(v) for v in raw.iloc[idx].tolist()]
        if any(value in {"sgRNA", "Perturbations", "Most enriched perturbation"} for value in values):
            header_idx = idx
            break
    return pd.read_excel(path, sheet_name=sheet, header=header_idx).dropna(how="all")


def inspect_xlsx_file(path: Path) -> Dict[str, object]:
    """Inspect workbook sheets and relevant columns only."""
    xls = pd.ExcelFile(path)
    sheets = []
    for sheet in xls.sheet_names:
        frame = _headers_from_first_row(path, sheet)
        columns = [str(c) for c in frame.columns]
        sequence_cols = [
            col for col in columns
            if any(token in col.lower() for token in ["sgrna", "guide", "perturb", "spacer", "sequence"])
        ]
        label_cols = [
            col for col in columns
            if any(token in col.lower() for token in ["score", "rank", "fold", "p-value", "fdr", "q-value", "label", "activity"])
        ]
        sequence = {}
        for col in sequence_cols:
            if col in frame.columns:
                summary = _sequence_summary(frame[col])
                summary["missing_sequence_count"] = int(frame[col].isna().sum())
                sequence[col] = summary
        labels = {}
        for col in label_cols:
            if col in frame.columns:
                series = frame[col]
                labels[col] = {
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "unique_summary": _unique_summary(series),
                    "numeric_summary": _numeric_summary(series),
                    "documented_unit": "screen statistic / rank / enrichment-style output unless publication establishes otherwise",
                    "documented_biological_meaning": "SCREEN_DERIVED_STATISTIC_OR_UNRESOLVED",
                    "experimentally_measured_explicit": "SCREEN_OUTPUT_REPORTED_NOT_CANONICAL_ACTIVITY",
                    "label_type_observed": "rank_or_enrichment_or_log_fold_change_or_statistic",
                }
        sheets.append({
            "sheet_name": sheet,
            "row_count": int(len(frame)),
            "column_names": columns,
            "sequence_columns": sequence,
            "label_columns": labels,
        })
    return {
        "file_format": "xlsx",
        "sheet_count": len(xls.sheet_names),
        "sheet_names": xls.sheet_names,
        "sheets": sheets,
    }


def inspect_zip_file(path: Path) -> Dict[str, object]:
    """Inspect CRISPRon archive manifest without treating it as data."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    docs = [n for n in names if n.lower().endswith((".md", "readme", "license", "copying", "txt")) or "readme" in n.lower()]
    code = [n for n in names if n.lower().endswith((".py", ".sh", ".r", ".ipynb"))]
    data_like = [n for n in names if n.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".txt"))]
    model_like = [n for n in names if any(token in n.lower() for token in ["saved_model.pb", "variables.data", ".model.best"])]
    candidate_tables = [n for n in data_like if any(token in n.lower() for token in ["crispron", "params", "data", "train"])]
    return {
        "file_format": "zip",
        "archive_member_count": len(names),
        "documentation_or_license_files": docs,
        "code_files": code,
        "data_like_files": data_like,
        "model_artifact_like_files": model_like[:30],
        "candidate_sequence_tables": candidate_tables,
        "candidate_label_tables": candidate_tables,
        "experimental_training_table_identified": False,
        "experimental_training_table_status": "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED",
    }


def _inspect_file(path: Path, file_format: str) -> Dict[str, object]:
    if file_format == "csv":
        return inspect_csv_file(path)
    if file_format == "xlsx":
        return inspect_xlsx_file(path)
    if file_format == "zip":
        return inspect_zip_file(path)
    return {"file_format": file_format, "status": "UNINSPECTED_FORMAT"}


def _candidate_status(candidate: Dict[str, object], verified_files: List[Dict[str, object]]) -> str:
    cid = str(candidate["candidate_id"])
    if cid == "crispron_2021_rth_tools":
        return "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED"
    if cid == "crisprpredseq_2020_bmc_additional_files":
        return "DERIVED_OR_PROCESSED"
    if cid == "doench_2016_orcs_publication_screens":
        return "PRIMARY_VERIFIED_WITH_METADATA_GAPS"
    if cid == "deep_hf_2019_public_data_listing":
        return "PRIMARY_FILE_NOT_RECOVERED"
    if cid == "sgdesigner_2020_public_data_listing":
        return "PRIMARY_FILE_NOT_RECOVERED"
    if cid == "corsi_2022_free_energy_pam_context":
        return "DISCOVERY_ONLY"
    return "PRIMARY_STATUS_UNRESOLVED" if verified_files else "PRIMARY_FILE_NOT_RECOVERED"


def _primary_status(candidate_id: str) -> str:
    return {
        "crispron_2021_rth_tools": "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED",
        "crisprpredseq_2020_bmc_additional_files": "author-provided processed / binary classification supplementary dataset",
        "doench_2016_orcs_publication_screens": "publication supplementary screen/workbook outputs with metadata gaps",
        "deep_hf_2019_public_data_listing": "PRIMARY_STATUS_UNRESOLVED",
        "sgdesigner_2020_public_data_listing": "PRIMARY_STATUS_UNRESOLVED",
        "corsi_2022_free_energy_pam_context": "DISCOVERY_ONLY",
    }.get(candidate_id, "PRIMARY_STATUS_UNRESOLVED")


def _dataset_relationship(candidate: Dict[str, object]) -> str:
    cid = str(candidate["candidate_id"])
    if cid == "crisprpredseq_2020_bmc_additional_files":
        return "Publication/additional-file provenance indicates processed source relationship to DeepCRISPR/Haeussler-used datasets; independence not inferred."
    if cid == "doench_2016_orcs_publication_screens":
        return "Multiple workbooks correspond to publication/ORCS screen outputs from the same Doench study; no overlap analysis performed."
    if cid == "crispron_2021_rth_tools":
        return "Repository archive contains code/models/test outputs; primary experimental training table not identified."
    return str(candidate.get("parent_dataset", "UNKNOWN"))


def verify_phase17b(phase17a_path: Path = PHASE17A_RESULT) -> Dict[str, object]:
    """Build the Phase 17B primary-file verification audit."""
    payload17a = load_phase17a_payload(phase17a_path)
    canonical_before = protected_hashes()
    candidates = []
    for candidate in payload17a["candidates"]:
        verified_files = []
        for file_record in candidate.get("files", []):
            local_raw_path = file_record.get("local_raw_path")
            if not file_record.get("verified_local_file") or not local_raw_path:
                continue
            path = Path(str(local_raw_path))
            file_format = str(file_record.get("file_format", "")).lower()
            inspection = _inspect_file(path, file_format)
            verified_files.append({
                "exact_local_filename": str(path).replace("\\", "/"),
                "original_filename": file_record.get("original_filename"),
                "repository": candidate.get("repository"),
                "accession": candidate.get("accession"),
                "publication": candidate.get("publication"),
                "doi_or_pubmed": candidate.get("doi_or_pubmed"),
                "official_url": file_record.get("source_url") or candidate.get("official_dataset_url"),
                "sha256": sha256_file(path),
                "phase17a_sha256": file_record.get("sha256"),
                "sha256_matches_phase17a": sha256_file(path) == file_record.get("sha256"),
                "access_date": candidate.get("access_date"),
                "direct_publication_repository_link": bool(candidate.get("official_dataset_url")),
                "file_format": file_format,
                "inspection": inspection,
            })
        status = _candidate_status(candidate, verified_files)
        if status not in ALLOWED_VERIFICATION_STATUSES:
            raise ValueError(f"Unsupported Phase 17B verification status: {status}")
        candidates.append({
            "candidate_id": candidate["candidate_id"],
            "publication": candidate.get("publication"),
            "repository_accession": {
                "repository": candidate.get("repository"),
                "accession": candidate.get("accession"),
                "official_dataset_url": candidate.get("official_dataset_url"),
                "doi_or_pubmed": candidate.get("doi_or_pubmed"),
            },
            "files": verified_files,
            "primary_vs_derived_processed_status": _primary_status(str(candidate["candidate_id"])),
            "experimental_context": {
                "organism_cell_line": candidate.get("organism_cell_line"),
                "nuclease": candidate.get("nuclease"),
                "assay_type": candidate.get("assay_type"),
                "experimental_condition": candidate.get("experimental_condition"),
                "values_experimentally_measured": "UNKNOWN_OR_SCREEN_OUTPUT_REPORTED",
            },
            "sequence_structure": _summarize_sequence_structure(verified_files),
            "label_structure": _summarize_label_structure(verified_files),
            "dataset_relationship": _dataset_relationship(candidate),
            "verification_status": status,
            "unresolved_issues": _unresolved_issues(candidate, status),
        })
    canonical_after = protected_hashes()
    return {
        "phase": "17B",
        "protocol_version": "phase17b-primary-file-verification-v1",
        "scope": "primary_file_verification_only_no_acceptance",
        "phase17a_source": str(phase17a_path).replace("\\", "/"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "canonical_integrity": {
            "before": canonical_before,
            "after": canonical_after,
            "unchanged": canonical_before == canonical_after,
        },
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_moreno_raw_data_accessed": True,
            "no_dataset_pooling": True,
            "no_combined_training_dataset": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_sequence_transformation": True,
            "no_near_duplicate_analysis": True,
            "no_rc_overlap": True,
            "no_continual_learning": True,
            "no_dataset_acceptance": True,
        },
    }


def _summarize_sequence_structure(files: List[Dict[str, object]]) -> Dict[str, object]:
    summaries = []
    for item in files:
        inspection = item["inspection"]
        if inspection["file_format"] == "csv":
            summaries.append({
                "file": item["exact_local_filename"],
                "sequence_columns": inspection["sequence_columns"],
            })
        elif inspection["file_format"] == "xlsx":
            summaries.append({
                "file": item["exact_local_filename"],
                "sheet_sequence_columns": [
                    {"sheet_name": sheet["sheet_name"], "sequence_columns": sheet["sequence_columns"]}
                    for sheet in inspection["sheets"]
                ],
            })
        elif inspection["file_format"] == "zip":
            summaries.append({
                "file": item["exact_local_filename"],
                "candidate_sequence_tables": inspection["candidate_sequence_tables"],
                "experimental_training_table_identified": inspection["experimental_training_table_identified"],
            })
    return {
        "status": "RAW_STRUCTURE_INSPECTED_NO_TRANSFORMATION",
        "file_summaries": summaries,
    }


def _summarize_label_structure(files: List[Dict[str, object]]) -> Dict[str, object]:
    summaries = []
    for item in files:
        inspection = item["inspection"]
        if inspection["file_format"] == "csv":
            summaries.append({
                "file": item["exact_local_filename"],
                "label_columns": inspection["label_columns"],
            })
        elif inspection["file_format"] == "xlsx":
            summaries.append({
                "file": item["exact_local_filename"],
                "sheet_label_columns": [
                    {"sheet_name": sheet["sheet_name"], "label_columns": sheet["label_columns"]}
                    for sheet in inspection["sheets"]
                ],
            })
        elif inspection["file_format"] == "zip":
            summaries.append({
                "file": item["exact_local_filename"],
                "candidate_label_tables": inspection["candidate_label_tables"],
                "experimental_training_table_identified": inspection["experimental_training_table_identified"],
            })
    return {
        "status": "RAW_LABEL_STRUCTURE_INSPECTED_NO_TRANSFORMATION",
        "file_summaries": summaries,
    }


def _unresolved_issues(candidate: Dict[str, object], status: str) -> List[str]:
    issues = list(candidate.get("unresolved_items", []))
    if status == "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED":
        issues.append("PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED")
    if status == "PRIMARY_FILE_NOT_RECOVERED":
        issues.append("primary_file_not_recovered_locally")
    if status == "DERIVED_OR_PROCESSED":
        issues.append("processed_or_partitioned_source_relationship_requires_later_gate")
    return sorted(set(str(issue) for issue in issues))


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
