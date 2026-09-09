"""Phase 16B primary-file verification records."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, List

from .discovery import build_phase16a_inventory
from .provenance import PENDING_PRIMARY_VERIFICATION

RAW_DIR = Path("data/phase16/raw")


def sha256_file(path: Path) -> str | None:
    """Return SHA-256 for an acquired file, or None when it is absent."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _file_record(filename: str, source_url: str, file_format: str,
                 provenance_role: str) -> Dict[str, object]:
    path = RAW_DIR / filename
    return {
        "original_filename": filename,
        "source_url": source_url,
        "local_raw_path": str(path).replace("\\", "/"),
        "file_format": file_format,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "verified_local_file": path.exists(),
        "provenance_role": provenance_role,
    }


def phase16b_file_verification(access_date: str | None = None) -> Dict[str, object]:
    """Build the Phase 16B candidate-by-candidate file verification audit."""
    if access_date is None:
        access_date = date.today().isoformat()
    inventory = build_phase16a_inventory(access_date)
    by_id = {candidate["candidate_id"]: candidate for candidate in inventory["candidates"]}

    records: List[Dict[str, object]] = [
        {
            "candidate_id": "doench_2016_orcs_publication_screens",
            "candidate_name": by_id["doench_2016_orcs_publication_screens"]["candidate_name"],
            "source_study_id": "doench_2016_nbt_3437",
            "source_publication": by_id["doench_2016_orcs_publication_screens"]["source_publication"],
            "primary_source_or_accession": "BioGRID ORCS Dataset 19; PubMed:26780180",
            "primary_file_status": "PRIMARY_SUPPLEMENTARY_FILES_OBTAINED",
            "candidate_status_after_16b": PENDING_PRIMARY_VERIFICATION,
            "files": [
                _file_record(
                    "doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx",
                    "https://orcs.thebiogrid.org/uploads/processed/5aeb49259de3c/STable_06_Vem_STARSOutputs.xlsx",
                    "xlsx",
                    "publisher/ORCS-hosted publication supplementary file",
                ),
                _file_record(
                    "doench2016_orcs_STable_09_Sel_STARSOutput.xlsx",
                    "https://orcs.thebiogrid.org/uploads/processed/5e4d50e13c7d6/STable%2009%20Sel_STARSOutput.xlsx",
                    "xlsx",
                    "publisher/ORCS-hosted publication supplementary file",
                ),
                _file_record(
                    "doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx",
                    "https://orcs.thebiogrid.org/uploads/processed/5af0c639e9d96/STable%2012%20NegativeSelection_individual_STARS.xlsx",
                    "xlsx",
                    "publisher/ORCS-hosted publication supplementary file",
                ),
                _file_record(
                    "doench2016_orcs_STable_15_IFNg_data_STARS.xlsx",
                    "https://orcs.thebiogrid.org/uploads/processed/5afb153562639/STable_15_IFNg_data_STARS.xlsx",
                    "xlsx",
                    "publisher/ORCS-hosted publication supplementary file",
                ),
            ],
            "provenance_relationship": (
                "Multiple supplementary screen tables from the same original "
                "Doench 2016 study; independence from canonical data is not "
                "assessed in Phase 16B."
            ),
            "unresolved_issues": [
                "Row-level label semantics not assessed.",
                "Sequence geometry not assessed.",
                "Subset/overlap relationship not assessed.",
            ],
        },
        {
            "candidate_id": "crisprpredseq_2020_bmc_additional_files",
            "candidate_name": by_id["crisprpredseq_2020_bmc_additional_files"]["candidate_name"],
            "source_study_id": "rafid_2020_crisprpredseq",
            "source_publication": by_id["crisprpredseq_2020_bmc_additional_files"]["source_publication"],
            "primary_source_or_accession": "BMC Bioinformatics DOI:10.1186/s12859-020-3531-9 supplementary files",
            "primary_file_status": "OFFICIAL_SUPPLEMENTARY_FILES_OBTAINED",
            "candidate_status_after_16b": PENDING_PRIMARY_VERIFICATION,
            "files": [
                _file_record(
                    "crisprpredseq_additional_file_1.csv",
                    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM1_ESM.csv",
                    "csv",
                    "publisher-hosted supplementary file",
                ),
                _file_record(
                    "crisprpredseq_additional_file_2.csv",
                    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM2_ESM.csv",
                    "csv",
                    "publisher-hosted supplementary file",
                ),
                _file_record(
                    "crisprpredseq_additional_file_3.csv",
                    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM3_ESM.csv",
                    "csv",
                    "publisher-hosted supplementary file",
                ),
                _file_record(
                    "crisprpredseq_additional_file_4.csv",
                    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM4_ESM.csv",
                    "csv",
                    "publisher-hosted supplementary file",
                ),
            ],
            "provenance_relationship": (
                "Article reports HCT116, HEK293, HeLa, and HL60 files as "
                "data used by previous DeepCRISPR/Haeussler work; independent "
                "identity cannot be assumed."
            ),
            "unresolved_issues": [
                "Whether files are primary experimental data or processed reuse remains unresolved.",
                "Original upstream source relationship requires verification.",
                "Label and sequence definitions deferred to Phase 16C.",
            ],
        },
        {
            "candidate_id": "crispron_2021_rth_tools",
            "candidate_name": by_id["crispron_2021_rth_tools"]["candidate_name"],
            "source_study_id": "xu_2021_crispron",
            "source_publication": by_id["crispron_2021_rth_tools"]["source_publication"],
            "primary_source_or_accession": "RTH-tools/crispron GitHub repository",
            "primary_file_status": "SOURCE_ARCHIVE_OBTAINED_PRIMARY_EXPERIMENTAL_FILE_UNRESOLVED",
            "candidate_status_after_16b": "INSUFFICIENT_PROVENANCE",
            "files": [
                _file_record(
                    "crispron_main.zip",
                    "https://github.com/RTH-tools/crispron/archive/refs/heads/main.zip",
                    "zip",
                    "official repository source archive; raw experimental table not identified in 16B",
                ),
            ],
            "provenance_relationship": (
                "Official repository archive was obtained, but the primary "
                "experimental sgRNA training table was not located from this "
                "archive during Phase 16B."
            ),
            "unresolved_issues": [
                "Actual primary experimental data filename unresolved.",
                "Repository archive appears to contain model artifacts/test output rather than raw study table.",
            ],
        },
        {
            "candidate_id": "deep_hf_2019_public_data_listing",
            "candidate_name": by_id["deep_hf_2019_public_data_listing"]["candidate_name"],
            "source_study_id": "wang_2019_deephf",
            "source_publication": by_id["deep_hf_2019_public_data_listing"]["source_publication"],
            "primary_source_or_accession": "DeepHF publication/source links; prior Phase 10-12 project records",
            "primary_file_status": "PRIMARY_FILE_NOT_OBTAINED_IN_16B_PRIOR_EVIDENCE_EXISTS",
            "candidate_status_after_16b": PENDING_PRIMARY_VERIFICATION,
            "files": [],
            "provenance_relationship": (
                "DeepHF was already analyzed in Phases 10-12 and must not be "
                "counted as a new independent dataset merely because it was "
                "rediscovered. Phase 16B did not restore the primary DeepHF "
                "file in data/phase16/raw."
            ),
            "unresolved_issues": [
                "Primary file unavailable in current Phase 16B acquisition.",
                "Role requires supervisor decision because DeepHF is prior evidence, not new discovery.",
            ],
        },
        {
            "candidate_id": "sgdesigner_2020_public_data_listing",
            "candidate_name": by_id["sgdesigner_2020_public_data_listing"]["candidate_name"],
            "source_study_id": "sgdesigner_2020",
            "source_publication": by_id["sgdesigner_2020_public_data_listing"]["source_publication"],
            "primary_source_or_accession": "public_data_crisprCas9 catalog link only",
            "primary_file_status": "INSUFFICIENT_PRIMARY_SOURCE_VERIFICATION",
            "candidate_status_after_16b": "INSUFFICIENT_PROVENANCE",
            "files": [],
            "provenance_relationship": (
                "Only catalog-level source-link evidence was retained in "
                "Phase 16B; no official primary file was obtained."
            ),
            "unresolved_issues": [
                "Primary repository URL and original data file unresolved.",
                "Publication identity requires primary-source confirmation.",
            ],
        },
    ]
    return {
        "phase": "16B",
        "scope": "primary_file_verification_only",
        "status": "PRIMARY_FILE_VERIFICATION_RECORDED",
        "access_date": access_date,
        "candidate_count": len(records),
        "records": records,
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_label_compatibility_decision": True,
            "no_sequence_transformation": True,
            "no_overlap_analysis": True,
            "no_distribution_analysis": True,
            "no_moreno_raw_data_accessed": True,
        },
    }
