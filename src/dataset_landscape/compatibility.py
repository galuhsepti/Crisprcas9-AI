"""Phase 16C biological/measurement compatibility audit.

This module records schema-level compatibility decisions only. It does not
transform sequences, harmonize labels, pool datasets, run overlap checks, read
Moreno-Mateos, or import model code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .verification import phase16b_file_verification

COMPATIBLE = "COMPATIBLE"
CONDITIONALLY_COMPATIBLE = "CONDITIONALLY_COMPATIBLE"
DOMAIN_SHIFT_ONLY = "DOMAIN_SHIFT_ONLY"
INCOMPATIBLE = "INCOMPATIBLE"
INSUFFICIENT_PROVENANCE = "INSUFFICIENT_PROVENANCE"


def classify_label_schema(
    observation: str,
    is_experimental: bool | None,
    is_continuous: bool | None,
    measurement_definition_matches: bool | None,
    is_binary: bool = False,
    is_rank: bool = False,
    is_enrichment: bool = False,
    is_predicted_score: bool = False,
    is_assay_normalized_proxy: bool = False,
    unit_conversion_only: bool = False,
) -> Dict[str, object]:
    """Conservative Phase 16C label classification."""
    disallowed = {
        "binary": is_binary,
        "rank": is_rank,
        "enrichment": is_enrichment,
        "predicted_or_model_score": is_predicted_score,
        "assay_specific_normalized_proxy": is_assay_normalized_proxy,
    }
    if any(disallowed.values()):
        return {
            "status": INCOMPATIBLE,
            "reason": "Disallowed label type for thesis target: " + ", ".join(
                k for k, v in disallowed.items() if v
            ),
            "observation": observation,
            "unit_conversion": bool(unit_conversion_only),
            "label_harmonization_performed": False,
        }
    if is_experimental is True and is_continuous is True and measurement_definition_matches is True:
        return {
            "status": COMPATIBLE,
            "reason": "Experimental continuous measurement matches the target definition.",
            "observation": observation,
            "unit_conversion": bool(unit_conversion_only),
            "label_harmonization_performed": False,
        }
    if is_experimental is None or is_continuous is None or measurement_definition_matches is None:
        return {
            "status": INSUFFICIENT_PROVENANCE,
            "reason": "Measurement semantics are not sufficiently documented for Phase 16C.",
            "observation": observation,
            "unit_conversion": bool(unit_conversion_only),
            "label_harmonization_performed": False,
        }
    return {
        "status": CONDITIONALLY_COMPATIBLE,
        "reason": "Experimental/continuous evidence exists, but substantive target match is unresolved.",
        "observation": observation,
        "unit_conversion": bool(unit_conversion_only),
        "label_harmonization_performed": False,
    }


def classify_sequence_schema(
    sequence_kind: str,
    length: int | None,
    orientation_verified: bool,
    genomic_context_verified: bool,
    synthetic_or_constant_flanks: bool = False,
) -> Dict[str, object]:
    """Conservative Phase 16C sequence-schema classification."""
    if sequence_kind == "canonical_30mer" and length == 30 and orientation_verified and genomic_context_verified:
        status = COMPATIBLE
        reason = "Documented canonical 30-mer with verified orientation and genomic context."
        transformation = "NONE"
    elif sequence_kind in {"guide_only", "guide_plus_pam"}:
        status = CONDITIONALLY_COMPATIBLE
        reason = "Noncanonical source sequence requires explicit future transformation approval."
        transformation = "REQUIRED_NOT_PERFORMED"
    elif synthetic_or_constant_flanks:
        status = CONDITIONALLY_COMPATIBLE
        reason = "Synthetic or constant flanks cannot be treated as genuine genomic context."
        transformation = "REQUIRED_NOT_PERFORMED"
    else:
        status = INSUFFICIENT_PROVENANCE
        reason = "Sequence geometry or orientation is unresolved."
        transformation = "UNKNOWN_NOT_PERFORMED"
    return {
        "status": status,
        "reason": reason,
        "sequence_kind": sequence_kind,
        "sequence_length": length,
        "orientation_verified": orientation_verified,
        "genomic_context_verified": genomic_context_verified,
        "required_transformation": transformation,
        "transformation_status": "NOT_PERFORMED",
    }


def _csv_schema(path: str) -> Dict[str, object]:
    df = pd.read_csv(path)
    return {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "sgRNA_length_counts": {
            str(k): int(v) for k, v in df["sgRNA"].astype(str).str.len().value_counts().sort_index().items()
        } if "sgRNA" in df.columns else {},
        "label_values_observed": sorted(str(v) for v in df["label"].dropna().unique()) if "label" in df.columns else [],
    }


def _xlsx_schema(path: str) -> Dict[str, object]:
    workbook = pd.ExcelFile(path)
    sheets = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, nrows=3)
        sheets.append({
            "sheet": sheet,
            "columns_observed": [str(c) for c in frame.columns],
            "first_row_values": [str(v) for v in frame.iloc[0].tolist()] if len(frame) else [],
        })
    return {"sheets": sheets}


def build_phase16c_compatibility(access_date: str | None = None) -> Dict[str, object]:
    """Build the Phase 16C compatibility audit from verified Phase 16B files."""
    verification = phase16b_file_verification(access_date)
    records = []

    for record in verification["records"]:
        cid = record["candidate_id"]
        if cid == "crisprpredseq_2020_bmc_additional_files":
            file_results = []
            for item in record["files"]:
                schema = _csv_schema(item["local_raw_path"])
                label = classify_label_schema(
                    observation="CSV columns sgRNA,label; observed label values are 0/1.",
                    is_experimental=None,
                    is_continuous=False,
                    measurement_definition_matches=False,
                    is_binary=True,
                )
                seq = classify_sequence_schema(
                    sequence_kind="guide_plus_pam",
                    length=23,
                    orientation_verified=False,
                    genomic_context_verified=False,
                )
                file_results.append(_file_result(record, item, schema, label, seq))
            records.append(_candidate_result(
                record, file_results, INCOMPATIBLE,
                "Verified files contain binary labels and 23-mer guide+PAM schema; no label harmonization or sequence transformation performed.",
            ))
        elif cid == "doench_2016_orcs_publication_screens":
            file_results = []
            for item in record["files"]:
                schema = _xlsx_schema(item["local_raw_path"])
                is_ifn = "IFNg" in item["original_filename"]
                label = classify_label_schema(
                    observation=(
                        "Workbook contains STARS/rank/enrichment-style screen outputs"
                        if not is_ifn else
                        "Workbook contains sgRNA and average log fold change IFNgamma - mock"
                    ),
                    is_experimental=True,
                    is_continuous=True if is_ifn else None,
                    measurement_definition_matches=False,
                    is_rank=not is_ifn,
                    is_enrichment=True,
                )
                seq = classify_sequence_schema(
                    sequence_kind="guide_only",
                    length=20,
                    orientation_verified=False,
                    genomic_context_verified=False,
                )
                file_results.append(_file_result(record, item, schema, label, seq))
            records.append(_candidate_result(
                record, file_results, INCOMPATIBLE,
                "Primary files are screen-enrichment/rank/log-fold-change outputs, not substantively compatible continuous SpCas9 editing activity.",
            ))
        else:
            records.append(_candidate_result(
                record, [], INSUFFICIENT_PROVENANCE,
                "Phase 16C evidence does not resolve this candidate beyond Phase 16B.",
            ))

    return {
        "phase": "16C",
        "scope": "biological_measurement_compatibility_only",
        "status": "COMPATIBILITY_AUDIT_RECORDED",
        "records": records,
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_dataset_pooling": True,
            "no_overlap_analysis": True,
            "no_rc_overlap_analysis": True,
            "no_distribution_analysis": True,
            "no_moreno_raw_data_accessed": True,
            "label_harmonization_performed": False,
            "sequence_transformation_performed": False,
        },
    }


def _file_result(record, item, schema, label, seq) -> Dict[str, object]:
    combined = label["status"]
    if label["status"] == INCOMPATIBLE or seq["status"] == INCOMPATIBLE:
        combined = INCOMPATIBLE
    elif label["status"] == INSUFFICIENT_PROVENANCE or seq["status"] == INSUFFICIENT_PROVENANCE:
        combined = INSUFFICIENT_PROVENANCE
    elif label["status"] == CONDITIONALLY_COMPATIBLE or seq["status"] == CONDITIONALLY_COMPATIBLE:
        combined = CONDITIONALLY_COMPATIBLE
    return {
        "candidate_id": record["candidate_id"],
        "file": item["local_raw_path"],
        "observation_unit": label["observation"],
        "schema_observation": schema,
        "sequence_schema": seq["sequence_kind"],
        "sequence_length": seq["sequence_length"],
        "guide_definition": "20-nt guide indicated by observed schema" if seq["sequence_kind"] == "guide_only" else "UNKNOWN_OR_NONCANONICAL",
        "PAM_definition": "included in 23-mer sgRNA for CRISPRpred(SEQ)" if seq["sequence_kind"] == "guide_plus_pam" else "UNKNOWN",
        "orientation": "UNKNOWN",
        "activity_definition": label["observation"],
        "activity_units": "binary 0/1" if "0/1" in label["observation"] else "screen score/log fold change/rank",
        "continuous_status": "NO_BINARY" if "0/1" in label["observation"] else "NOT_TARGET_COMPATIBLE_OR_UNCLEAR",
        "experimental_status": "EXPERIMENTAL_OR_SCREEN_OUTPUT_REPORTED",
        "organism": "human or publication-specific; row-level check deferred",
        "cell_line_or_system": "publication-specific; see candidate record",
        "Cas9_variant": "UNKNOWN",
        "assay": record.get("primary_source_or_accession", "UNKNOWN"),
        "compatibility_status": combined,
        "compatibility_reason": label["reason"] + " Sequence: " + seq["reason"],
        "required_transformation": seq["required_transformation"],
        "transformation_status": seq["transformation_status"],
        "provenance_confidence": "PRIMARY_FILE_SCHEMA_OBSERVED",
        "unresolved_questions": [
            "No label harmonization approved or performed.",
            "No sequence transformation approved or performed.",
            "Independence and overlap are deferred to Phase 16D.",
        ],
    }


def _candidate_result(record, file_results: List[Dict[str, object]],
                      status: str, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": record["candidate_id"],
        "candidate_name": record["candidate_name"],
        "compatibility_status": status,
        "compatibility_reason": reason,
        "files": file_results,
        "required_transformations": sorted({
            f["required_transformation"] for f in file_results
        }) if file_results else [],
        "transformation_status": "NOT_PERFORMED",
        "unresolved_questions": record.get("unresolved_issues", []),
    }
