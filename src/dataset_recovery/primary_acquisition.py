"""Phase 17H targeted primary dataset acquisition and provenance recovery."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .phase17a import protected_hashes
from .provenance_chain import validate_provenance_chain


PHASE17G_RESULT = Path("results/phase17g_data_sufficiency_20260911_222112.json")
RAW_DIR = Path("data/phase17h/raw")
MANIFEST_DIR = Path("data/phase17h/manifests")

BLOCKING_LABEL_TERMS = ["binary", "rank", "stars", "enrichment", "log fold", "log-fold", "score"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_doi(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.rstrip(".")


def source_key(record: Dict[str, object]) -> str:
    doi = normalize_doi(str(record.get("doi", "")))
    if doi:
        return f"doi:{doi}"
    return f"title:{str(record.get('primary_publication', '')).strip().lower()}"


def duplicate_source_detection(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    keys = [source_key(record) for record in records]
    counts = Counter(keys)
    return {
        "duplicates": sorted(key for key, count in counts.items() if count > 1),
        "unique_source_count": len(counts),
        "record_count": len(records),
    }


def is_dna(value: object) -> bool:
    seq = str(value).strip().upper()
    return bool(seq) and set(seq) <= set("ACGT")


def sequence_summary(values: Iterable[object]) -> Dict[str, object]:
    seqs = [str(v).strip().upper() for v in values if pd.notna(v) and str(v).strip()]
    valid = [s for s in seqs if is_dna(s)]
    return {
        "raw_sequence_n": len(seqs),
        "valid_sequence_n": len(valid),
        "malformed_sequence_n": len(seqs) - len(valid),
        "unique_sequence_n": len(set(valid)),
        "sequence_length_distribution": dict(sorted(Counter(len(s) for s in seqs).items())),
    }


def label_profile(values: Iterable[object], label_name: str, label_definition: str = "") -> Dict[str, object]:
    name = f"{label_name} {label_definition}".lower()
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if any(term in name for term in BLOCKING_LABEL_TERMS):
        status = "SCREEN_OR_RANK_LABEL" if any(term in name for term in ["rank", "stars", "enrichment", "log fold", "log-fold"]) else "INCOMPATIBLE_LABEL"
    elif set(finite.dropna().astype(float).unique()).issubset({0.0, 1.0}) and finite.dropna().nunique() <= 2:
        status = "BINARY_LABEL"
    elif finite.dropna().nunique() >= 20:
        status = "CONTINUOUS_EXPERIMENTAL_CANDIDATE"
    else:
        status = "UNKNOWN_OR_LOW_CARDINALITY"
    return {
        "label_status": status,
        "n_labels": int(len(numeric)),
        "finite_labels": int(finite.dropna().shape[0]),
        "missing_labels": int(numeric.isna().sum()),
        "unique_label_values": int(finite.dropna().nunique()),
        "min": float(finite.min()) if not finite.empty else None,
        "max": float(finite.max()) if not finite.empty else None,
    }


def inspect_table(frame: pd.DataFrame, sequence_column: str | None, label_column: str | None, label_definition: str = "") -> Dict[str, object]:
    if not sequence_column or sequence_column not in frame.columns:
        return {"status": "MISSING_SEQUENCE_COLUMN", "row_count": int(len(frame))}
    if not label_column or label_column not in frame.columns:
        return {"status": "MISSING_LABEL_COLUMN", "row_count": int(len(frame))}
    seq = sequence_summary(frame[sequence_column])
    label = label_profile(frame[label_column], label_column, label_definition)
    return {
        "status": "INSPECTED",
        "row_count": int(len(frame)),
        "sequence_column": sequence_column,
        "label_column": label_column,
        "sequence": seq,
        "label": label,
    }


def _inspect_corsi_file(path: Path) -> Dict[str, object]:
    tables = []
    if path.suffix.lower() != ".xlsx" or path.stat().st_size < 10_000:
        return {
            "path": str(path).replace("\\", "/"),
            "file_sha256": sha256_file(path),
            "status": "NOT_VALID_XLSX_OR_ERROR_PAYLOAD",
            "tables": [],
        }
    workbook = pd.ExcelFile(path)
    all_sequences = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        table = inspect_table(
            frame,
            sequence_column="30mer_gRNA",
            label_column="Indel frequency (% avg D6-D10)",
            label_definition="continuous indel frequency percentage averaged across day 6 and day 10",
        )
        table["sheet_name"] = sheet
        tables.append(table)
        if table["status"] == "INSPECTED":
            all_sequences.extend(frame["30mer_gRNA"].astype(str).str.upper().str.strip().tolist())
    valid_union = {s for s in all_sequences if is_dna(s)}
    return {
        "path": str(path).replace("\\", "/"),
        "file_sha256": sha256_file(path),
        "status": "INSPECTED",
        "tables": tables,
        "union_unique_sequence_n": len(valid_union),
    }


def _candidate_from_corsi(retrieval_date: str) -> Dict[str, object]:
    path = RAW_DIR / "corsi_2022_supplementary_data_1.xlsx"
    failed_paths = [
        RAW_DIR / "corsi_2022_supplementary_data_2.xlsx",
        RAW_DIR / "corsi_2022_source_data.xlsx",
    ]
    inspection = _inspect_corsi_file(path)
    chain = validate_provenance_chain({
        "primary_publication": "Corsi et al. 2022. CRISPR/Cas9 gRNA activity depends on free energy changes and on the target PAM context.",
        "primary_file": str(path),
        "table_or_sheet": "processed_Dox-; processed_Dox+",
        "sequence_column": "30mer_gRNA",
        "label_column": "Indel frequency (% avg D6-D10)",
        "primary_experimental_measurement": True,
    })
    unique_n = int(inspection.get("union_unique_sequence_n", 0))
    preliminary = "PROMISING" if chain["status"] == "VERIFIED_PRIMARY" and unique_n >= 1000 else "INSUFFICIENT_EVIDENCE"
    return {
        "candidate_id": "corsi_2022_pam_context_supplementary_data_1",
        "name": "Corsi 2022 PAM-context SpCas9 cleavage-efficiency supplementary data",
        "primary_publication": "Corsi et al. 2022. CRISPR/Cas9 gRNA activity depends on free energy changes and on the target PAM context.",
        "doi": "10.1038/s41467-022-30515-0",
        "publication_year": 2022,
        "Cas_system": "SpCas9",
        "organism": "human cell context / publication-level evidence",
        "cell_type": "NOT_FULLY_RESOLVED_FILE_LEVEL",
        "assay_type": "targeted SpCas9 cleavage / indel measurement",
        "experimental_measurement": "Indel frequency (% avg D6-D10)",
        "label_name": "Indel frequency (% avg D6-D10)",
        "label_definition": "continuous indel frequency percentage averaged across day 6 and day 10; no rescaling performed",
        "label_units_or_scale": "percent, 0-100 observed",
        "sequence_column": "30mer_gRNA",
        "sequence_length": 30,
        "PAM_information": "PAM and PAM(4bp) columns present",
        "guide_count_raw": sum(t.get("row_count", 0) for t in inspection.get("tables", [])),
        "guide_count_unique": unique_n,
        "primary_file": str(path).replace("\\", "/"),
        "primary_file_url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-022-30515-0/MediaObjects/41467_2022_30515_MOESM3_ESM.xlsx",
        "repository_url": None,
        "supplementary_file_name": "Supplementary Data 1",
        "file_format": "xlsx",
        "file_sha256": inspection.get("file_sha256"),
        "retrieval_status": "RECOVERED",
        "retrieval_date": retrieval_date,
        "provenance_status": chain["status"],
        "compatibility_status": "POTENTIALLY_COMPATIBLE_REQUIRES_PHASE17G_REEVALUATION",
        "preliminary_status": preliminary,
        "blocking_reasons": [
            "Requires formal Phase 17G re-evaluation before any acceptance.",
            "Potential unique observations remain below total 2000 threshold without another compatible dataset.",
            "Cell/experiment metadata require full provenance review.",
        ],
        "evidence": {
            "provenance_chain": chain,
            "file_inspection": inspection,
            "failed_or_non_primary_downloads": [
                {
                    "path": str(p).replace("\\", "/"),
                    "size": p.stat().st_size if p.exists() else 0,
                    "sha256": sha256_file(p) if p.exists() else None,
                    "status": "HTML_ERROR_OR_NOT_VALID_XLSX",
                }
                for p in failed_paths
            ],
        },
    }


def _ledger_entries() -> List[Dict[str, object]]:
    return [
        {
            "search_source": "Phase 17 carryover + GitHub/Nature/publication search",
            "search_terms": ["CRISPRon", "10,592 SpCas9 gRNAs", "s41467-021-23576-0"],
            "candidate_publication": "Xiang/Xu et al. 2021 CRISPRon",
            "doi": "10.1038/s41467-021-23576-0",
            "primary_experimental_data_exist": "PUBLICATION_REPORTS_GENERATION_OF_10592_SPCAS9_GRNAS",
            "row_level_guide_sequences_exist": "NOT_RECOVERED_IN_PHASE17H",
            "continuous_on_target_activity_exists": "LIKELY_PUBLICATION_LEVEL_NOT_FILE_VERIFIED",
            "approximate_n": 10592,
            "exclusion_reason": "Primary experimental table not identified in recovered repository archive; no new table recovered in Phase 17H.",
            "next_action": "Locate official Supplementary Data or author-deposited table with row-level gRNA activity.",
        },
        {
            "search_source": "Nature article supplementary files",
            "search_terms": ["Corsi 2022", "Supplementary Data 1", "Indel frequency", "30mer_gRNA"],
            "candidate_publication": "Corsi et al. 2022 PAM-context SpCas9 cleavage efficiency",
            "doi": "10.1038/s41467-022-30515-0",
            "primary_experimental_data_exist": True,
            "row_level_guide_sequences_exist": True,
            "continuous_on_target_activity_exists": True,
            "approximate_n": 1022,
            "exclusion_reason": None,
            "next_action": "Send through Phase 17G re-evaluation with explicit provenance and independence checks.",
        },
        {
            "search_source": "Phase 17 carryover / prior reports",
            "search_terms": ["sgDesigner", "sgRNA Designer", "on-target activity"],
            "candidate_publication": "sgDesigner / Broad sgRNA Designer lineage",
            "doi": "NOT_ESTABLISHED",
            "primary_experimental_data_exist": "NOT_ESTABLISHED",
            "row_level_guide_sequences_exist": "NOT_RECOVERED",
            "continuous_on_target_activity_exists": "NOT_FILE_VERIFIED",
            "approximate_n": None,
            "exclusion_reason": "No primary experimental file recovered; possible overlap with already assessed Doench/Rule Set records.",
            "next_action": "Manual primary-source tracing required before further audit.",
        },
        {
            "search_source": "Phase 17 carryover",
            "search_terms": ["CRISPRpred", "binary label", "BMC supplementary"],
            "candidate_publication": "CRISPRpred(SEQ) 2020",
            "doi": "10.1186/s12859-020-3531-9",
            "primary_experimental_data_exist": "PROCESSED_SUPPLEMENTARY_FILES_ONLY",
            "row_level_guide_sequences_exist": True,
            "continuous_on_target_activity_exists": False,
            "approximate_n": 16749,
            "exclusion_reason": "Known incompatible binary labels and 23-mer geometry; not re-rescued.",
            "next_action": "None unless primary continuous experimental source is discovered.",
        },
        {
            "search_source": "Phase 17 carryover",
            "search_terms": ["Doench ORCS", "STARS", "rank", "screen"],
            "candidate_publication": "Doench et al. 2016 ORCS screens",
            "doi": "PMID:26780180",
            "primary_experimental_data_exist": True,
            "row_level_guide_sequences_exist": True,
            "continuous_on_target_activity_exists": False,
            "approximate_n": 194653,
            "exclusion_reason": "Known incompatible screen/rank/enrichment/log-fold-change target; not re-rescued.",
            "next_action": "None for current continuous editing-activity target.",
        },
    ]


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def build_phase17h_primary_acquisition(retrieval_date: str | None = None) -> Dict[str, object]:
    retrieval_date = retrieval_date or date.today().isoformat()
    before = protected_hashes()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [_candidate_from_corsi(retrieval_date)]
    promising = [c["candidate_id"] for c in candidates if c["preliminary_status"] == "PROMISING"]
    blocked = [
        entry["candidate_publication"]
        for entry in _ledger_entries()
        if entry.get("exclusion_reason")
    ]
    potential_total = sum(c["guide_count_unique"] for c in candidates if c["preliminary_status"] == "PROMISING")
    files_recovered = [
        {
            "path": c["primary_file"],
            "sha256": c["file_sha256"],
            "source_url": c["primary_file_url"],
            "candidate_id": c["candidate_id"],
        }
        for c in candidates
        if c["retrieval_status"] == "RECOVERED"
    ]
    after = protected_hashes()
    payload = {
        "phase": "17H",
        "objective": "Targeted primary dataset acquisition and provenance recovery for future Phase 17G re-evaluation.",
        "canonical_protection": {
            "no_moreno_access": True,
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_label_transformation": True,
            "no_sequence_reconstruction": True,
            "no_phase18": True,
        },
        "previous_phase": {
            "phase": "17G",
            "decision": "NO_GO",
            "accepted_candidate_count": 0,
            "additional_required": 2000,
        },
        "sources_investigated": _ledger_entries(),
        "investigated_source_registry": duplicate_source_detection(_ledger_entries()),
        "files_recovered": files_recovered,
        "candidate_datasets": candidates,
        "potential_total_unique_observations": potential_total,
        "phase17g_minimum_total_new_unique_n": 2000,
        "remaining_deficit_relative_to_2000": max(0, 2000 - potential_total),
        "promising_candidates": promising,
        "blocked_candidates": blocked,
        "next_action": "Run a supervisor-approved Phase 17G re-evaluation only after additional compatible primary datasets are recovered or Corsi evidence is fully reviewed.",
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
    }
    write_manifest(MANIFEST_DIR / "phase17h_investigated_sources.json", {"sources_investigated": payload["sources_investigated"]})
    return payload


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    write_manifest(path, payload)
