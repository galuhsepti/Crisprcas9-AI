"""Phase 17A public dataset recovery and acquisition audit.

This module records provenance and preserves recovered public files. It performs
structure inspection only: no model training, model evaluation, pooling, label
harmonization, sequence transformation, Moreno access, or acceptance decision.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


RAW_DIR = Path("data/phase17/raw")

ALLOWED_STATUSES = {
    "RECOVERED_PRIMARY_FILE",
    "RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA",
    "PRIMARY_FILE_NOT_FOUND",
    "INSUFFICIENT_PROVENANCE",
    "DISCOVERY_ONLY",
    "CLEARLY_INCOMPATIBLE",
    "ALREADY_ANALYZED_NOT_NEW",
}

DEEPSPCAS9_RAW_PATH = Path("data/raw/DeepSpCas9.csv")
MORENO_RAW_PATH = Path("data") / "raw" / ("Moreno-" + "Mateos.csv")
MODEL_PROTECTED_PATHS = [
    Path("models/rf_baseline_fixed_20260905_001107.pkl"),
    Path("models/xgboost_baseline_20260905_002839.pkl"),
    Path("models/cnn_baseline_20260905_011720.pt"),
]

CANONICAL_EXPECTED_SHA256 = {
    "models/xgboost_baseline_20260905_002839.pkl": "129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd",
    "models/cnn_baseline_20260905_011720.pt": "76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616",
}


@dataclass(frozen=True)
class RecoveryFile:
    original_filename: str
    local_filename: str
    source_url: str
    source_local_path: str | None = None
    file_format: str = "unknown"
    provenance_role: str = "primary candidate file"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_study_id: str
    publication: str
    doi_or_pubmed: str
    repository: str
    accession: str
    official_dataset_url: str
    parent_dataset: str
    subset_id: str
    experimental_condition: str
    organism_cell_line: str
    nuclease: str
    assay_type: str
    derived_or_processed_status: str
    provenance_notes: str
    planned_status: str
    files: List[RecoveryFile] = field(default_factory=list)
    unresolved_items: List[str] = field(default_factory=list)
    stop_reason: str = ""
    phase16_carryover: bool = True


def sha256_file(path: Path) -> str:
    """Return lowercase SHA-256 for a local file."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes() -> Dict[str, object]:
    """Record canonical integrity without reading the locked Moreno raw file."""
    records = {}
    for path in [DEEPSPCAS9_RAW_PATH, *MODEL_PROTECTED_PATHS]:
        key = str(path).replace("\\", "/")
        records[key] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "matches_expected": (
                sha256_file(path) == CANONICAL_EXPECTED_SHA256[key]
                if key in CANONICAL_EXPECTED_SHA256 and path.exists()
                else None
            ),
        }
    moreno_key = str(MORENO_RAW_PATH).replace("\\", "/")
    records[moreno_key] = {
        "exists": MORENO_RAW_PATH.exists(),
        "sha256": "NOT_COMPUTED_MORENO_LOCKED",
        "matches_expected": None,
        "git_diff_clean": _git_path_clean(MORENO_RAW_PATH),
    }
    return records


def _git_path_clean(path: Path) -> bool | None:
    """Check path cleanliness through git without reading file contents."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() == ""


def candidate_manifest() -> List[Candidate]:
    """Bounded Phase 17A recovery manifest."""
    common_unresolved = [
        "sequence_orientation",
        "sequence_geometry",
        "pam_definition",
        "label_semantics",
        "unit_semantics",
        "subset_relationship",
        "processed_or_derived_status",
        "study_independence",
    ]
    return [
        Candidate(
            candidate_id="crispron_2021_rth_tools",
            source_study_id="xu_2021_crispron",
            publication="Xu F et al. 2021. Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning.",
            doi_or_pubmed="DOI:10.1038/s41467-021-23576-0",
            repository="RTH-tools/crispron GitHub repository",
            accession="NOT_APPLICABLE",
            official_dataset_url="https://github.com/RTH-tools/crispron/archive/refs/heads/main.zip",
            parent_dataset="UNKNOWN",
            subset_id="repository_source_archive",
            experimental_condition="NOT_ESTABLISHED_FROM_RECOVERED_FILE",
            organism_cell_line="NOT_ESTABLISHED_FROM_RECOVERED_FILE",
            nuclease="SpCas9 described by publication; row-level file unresolved",
            assay_type="publication reports integrated data and gRNA efficiency; recovered archive is source repository, not verified primary table",
            derived_or_processed_status="SOURCE_ARCHIVE_PRIMARY_TABLE_UNRESOLVED",
            provenance_notes="Official repository archive was recovered as in Phase 16, but the primary experimental table identity remains unresolved.",
            planned_status="RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA",
            files=[
                RecoveryFile(
                    original_filename="crispron_main.zip",
                    local_filename="crispron_2021_rth_tools_main.zip",
                    source_url="https://github.com/RTH-tools/crispron/archive/refs/heads/main.zip",
                    source_local_path="data/phase16/raw/crispron_main.zip",
                    file_format="zip",
                    provenance_role="official repository source archive; raw experimental table not identified",
                )
            ],
            unresolved_items=common_unresolved,
        ),
        Candidate(
            candidate_id="crisprpredseq_2020_bmc_additional_files",
            source_study_id="rafid_2020_crisprpredseq",
            publication="Rafid AHM et al. 2020. CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction using traditional machine learning.",
            doi_or_pubmed="DOI:10.1186/s12859-020-3531-9",
            repository="BMC Bioinformatics supplementary files",
            accession="NOT_APPLICABLE",
            official_dataset_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7310273/",
            parent_dataset="DeepCRISPR/Haeussler-used source relationship reported; unresolved",
            subset_id="additional_files_1_to_4",
            experimental_condition="HCT116/HEK293/HeLa/HL60 file labels from publication supplement",
            organism_cell_line="human cell lines named by supplementary-file context; row-level audit later",
            nuclease="CRISPR/Cas9 context; exact row-level nuclease unresolved",
            assay_type="binary-labelled sgRNA records in supplementary CSV files",
            derived_or_processed_status="OFFICIAL_SUPPLEMENTARY_FILES_PARENT_RELATIONSHIP_UNRESOLVED",
            provenance_notes="Official BMC supplementary CSV files already recovered in Phase 16; retained for Phase 17A provenance audit only.",
            planned_status="RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA",
            files=[
                RecoveryFile(
                    original_filename=f"crisprpredseq_additional_file_{i}.csv",
                    local_filename=f"crisprpredseq_2020_additional_file_{i}.csv",
                    source_url=f"https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM{i}_ESM.csv",
                    source_local_path=f"data/phase16/raw/crisprpredseq_additional_file_{i}.csv",
                    file_format="csv",
                    provenance_role="publisher-hosted supplementary file",
                )
                for i in range(1, 5)
            ],
            unresolved_items=common_unresolved,
        ),
        Candidate(
            candidate_id="doench_2016_orcs_publication_screens",
            source_study_id="doench_2016_nbt_3437",
            publication="Doench JG et al. 2016. Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9.",
            doi_or_pubmed="PMID:26780180",
            repository="BioGRID ORCS Dataset 19 / publication supplementary files",
            accession="BioGRID ORCS Dataset 19",
            official_dataset_url="https://orcs.thebiogrid.org/",
            parent_dataset="same publication supplementary screen tables",
            subset_id="STable_06_09_12_15",
            experimental_condition="publication screen-specific conditions",
            organism_cell_line="human screens; exact row-level context deferred",
            nuclease="SpCas9 publication context; row-level nuclease unresolved",
            assay_type="screen/rank/enrichment/log-fold-change style outputs",
            derived_or_processed_status="ORCS_HOSTED_PUBLICATION_SUPPLEMENTARY_FILES",
            provenance_notes="Official ORCS-hosted supplementary files recovered in Phase 16; Phase 17A records structure only.",
            planned_status="RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA",
            files=[
                RecoveryFile(
                    original_filename="STable_06_Vem_STARSOutputs.xlsx",
                    local_filename="doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx",
                    source_url="https://orcs.thebiogrid.org/uploads/processed/5aeb49259de3c/STable_06_Vem_STARSOutputs.xlsx",
                    source_local_path="data/phase16/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx",
                    file_format="xlsx",
                    provenance_role="ORCS-hosted publication supplementary file",
                ),
                RecoveryFile(
                    original_filename="STable_09_Sel_STARSOutput.xlsx",
                    local_filename="doench2016_orcs_STable_09_Sel_STARSOutput.xlsx",
                    source_url="https://orcs.thebiogrid.org/uploads/processed/5e4d50e13c7d6/STable%2009%20Sel_STARSOutput.xlsx",
                    source_local_path="data/phase16/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx",
                    file_format="xlsx",
                    provenance_role="ORCS-hosted publication supplementary file",
                ),
                RecoveryFile(
                    original_filename="STable_12_NegativeSelection_individual_STARS.xlsx",
                    local_filename="doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx",
                    source_url="https://orcs.thebiogrid.org/uploads/processed/5af0c639e9d96/STable%2012%20NegativeSelection_individual_STARS.xlsx",
                    source_local_path="data/phase16/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx",
                    file_format="xlsx",
                    provenance_role="ORCS-hosted publication supplementary file",
                ),
                RecoveryFile(
                    original_filename="STable_15_IFNg_data_STARS.xlsx",
                    local_filename="doench2016_orcs_STable_15_IFNg_data_STARS.xlsx",
                    source_url="https://orcs.thebiogrid.org/uploads/processed/5afb153562639/STable_15_IFNg_data_STARS.xlsx",
                    source_local_path="data/phase16/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx",
                    file_format="xlsx",
                    provenance_role="ORCS-hosted publication supplementary file",
                ),
            ],
            unresolved_items=common_unresolved,
        ),
        Candidate(
            candidate_id="deep_hf_2019_public_data_listing",
            source_study_id="wang_2019_deephf",
            publication="Wang D et al. 2019. Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning.",
            doi_or_pubmed="DOI:10.1038/s41467-019-12281-8; PMID:31537894",
            repository="Nature Communications supplementary data / DeepHF web server",
            accession="NOT_APPLICABLE",
            official_dataset_url="https://www.nature.com/articles/s41467-019-12281-8",
            parent_dataset="Phase 10-12 prior project evidence exists",
            subset_id="Supplementary Data 2 candidate",
            experimental_condition="HEK293T high-throughput guide RNA-target screen described by publication",
            organism_cell_line="human HEK293T described by publication",
            nuclease="WT-SpCas9, eSpCas9(1.1), SpCas9-HF1 described by publication",
            assay_type="indel rates of >50,000 gRNAs for each nuclease described by publication",
            derived_or_processed_status="PRIMARY_SUPPLEMENTARY_FILE_NOT_RECOVERED_IN_17A",
            provenance_notes="Nature article and supplementary link were located, but Springer media download returned browser-check HTML in this environment; no valid XLSX was preserved.",
            planned_status="PRIMARY_FILE_NOT_FOUND",
            files=[],
            unresolved_items=common_unresolved,
            stop_reason="Official supplementary XLSX URL located but not recoverable locally due media download/browser-check response.",
        ),
        Candidate(
            candidate_id="sgdesigner_2020_public_data_listing",
            source_study_id="sgdesigner_2020",
            publication="SgDesigner / unique plasmid library sgRNA potency data; primary publication identity unresolved in project records",
            doi_or_pubmed="UNKNOWN",
            repository="public_data_crisprCas9 catalogue only",
            accession="UNKNOWN",
            official_dataset_url="UNKNOWN",
            parent_dataset="UNKNOWN",
            subset_id="UNKNOWN",
            experimental_condition="UNKNOWN",
            organism_cell_line="UNKNOWN",
            nuclease="UNKNOWN",
            assay_type="UNKNOWN",
            derived_or_processed_status="CATALOGUE_ONLY",
            provenance_notes="No official primary file or unambiguous primary publication/repository was established in bounded Phase 17A search.",
            planned_status="INSUFFICIENT_PROVENANCE",
            files=[],
            unresolved_items=common_unresolved,
            stop_reason="Only discovery/catalogue-level evidence available in current project records.",
        ),
        Candidate(
            candidate_id="corsi_2022_free_energy_pam_context",
            source_study_id="corsi_2022_natcomm_3006",
            publication="Corsi GI et al. 2022. CRISPR/Cas9 gRNA activity depends on free energy changes and on the target PAM context.",
            doi_or_pubmed="DOI:10.1038/s41467-022-30515-0",
            repository="Nature Communications supplementary data",
            accession="NOT_APPLICABLE",
            official_dataset_url="https://www.nature.com/articles/s41467-022-30515-0",
            parent_dataset="UNKNOWN",
            subset_id="Supplementary data candidate",
            experimental_condition="publication describes Cas9 efficiency / indel frequency data",
            organism_cell_line="NOT_ESTABLISHED_IN_LOCAL_FILE",
            nuclease="SpCas9 indicated by publication title/context",
            assay_type="reported indel-frequency / Cas9-efficiency data",
            derived_or_processed_status="DISCOVERY_PUBLICATION_ONLY_IN_17A",
            provenance_notes="Bounded discovery found a new target-profile candidate, but no primary file was downloaded in Phase 17A.",
            planned_status="DISCOVERY_ONLY",
            files=[],
            unresolved_items=common_unresolved,
            stop_reason="Discovery-only candidate; primary file acquisition deferred because direct Springer media download was not verified locally.",
            phase16_carryover=False,
        ),
    ]


def _copy_preserved_file(file_record: RecoveryFile) -> Path | None:
    """Copy an already recovered public file into Phase 17 raw storage."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not file_record.source_local_path:
        return None
    source = Path(file_record.source_local_path)
    if not source.exists():
        return None
    target = RAW_DIR / file_record.local_filename
    if not target.exists() or sha256_file(target) != sha256_file(source):
        shutil.copy2(source, target)
    return target


def inspect_file(path: Path, file_format: str) -> Dict[str, object]:
    """Inspect structure only; do not transform data or infer compatibility."""
    if not path.exists():
        return {"exists": False}
    base = {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "file_format": file_format,
    }
    if file_format == "csv":
        df = pd.read_csv(path)
        return {
            **base,
            "row_count": int(len(df)),
            "column_names": [str(c) for c in df.columns],
            "obvious_sequence_columns": [str(c) for c in df.columns if "seq" in str(c).lower() or "sgrna" in str(c).lower() or "guide" in str(c).lower()],
            "obvious_label_columns": [str(c) for c in df.columns if "label" in str(c).lower() or "activity" in str(c).lower() or "score" in str(c).lower() or "indel" in str(c).lower()],
            "missing_cells_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }
    if file_format == "xlsx":
        xls = pd.ExcelFile(path)
        sheets = []
        for sheet in xls.sheet_names[:5]:
            frame = pd.read_excel(path, sheet_name=sheet, nrows=20)
            sheets.append({
                "sheet_name": sheet,
                "preview_row_count": int(len(frame)),
                "column_names": [str(c) for c in frame.columns],
                "obvious_sequence_columns": [str(c) for c in frame.columns if "seq" in str(c).lower() or "sgrna" in str(c).lower() or "guide" in str(c).lower() or "perturb" in str(c).lower()],
                "obvious_label_columns": [str(c) for c in frame.columns if "label" in str(c).lower() or "activity" in str(c).lower() or "score" in str(c).lower() or "fold" in str(c).lower() or "rank" in str(c).lower()],
            })
        return {
            **base,
            "sheet_count": len(xls.sheet_names),
            "sheet_names": xls.sheet_names,
            "sheet_previews": sheets,
        }
    if file_format == "zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return {
            **base,
            "archive_member_count": len(names),
            "archive_members_preview": names[:40],
            "obvious_data_members": [n for n in names if n.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls"))][:40],
        }
    return base


def recover_phase17a(access_date: str | None = None) -> Dict[str, object]:
    """Run the Phase 17A acquisition audit."""
    access_date = access_date or date.today().isoformat()
    canonical_before = protected_hashes()
    records = []
    for candidate in candidate_manifest():
        file_records = []
        for item in candidate.files:
            local_path = _copy_preserved_file(item)
            file_payload = {
                "original_filename": item.original_filename,
                "local_filename": item.local_filename,
                "local_raw_path": str(local_path).replace("\\", "/") if local_path else None,
                "source_url": item.source_url,
                "file_format": item.file_format,
                "provenance_role": item.provenance_role,
                "verified_local_file": bool(local_path and local_path.exists()),
                "sha256": sha256_file(local_path) if local_path and local_path.exists() else None,
                "structure": inspect_file(local_path, item.file_format) if local_path else {"exists": False},
            }
            file_records.append(file_payload)
        verified_count = sum(1 for item in file_records if item["verified_local_file"])
        status = candidate.planned_status
        if candidate.files and verified_count == 0:
            status = "PRIMARY_FILE_NOT_FOUND"
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported Phase 17A status: {status}")
        records.append({
            "candidate_id": candidate.candidate_id,
            "source_study_id": candidate.source_study_id,
            "publication": candidate.publication,
            "doi_or_pubmed": candidate.doi_or_pubmed,
            "repository": candidate.repository,
            "accession": candidate.accession,
            "official_dataset_url": candidate.official_dataset_url,
            "access_date": access_date,
            "parent_dataset": candidate.parent_dataset,
            "subset_id": candidate.subset_id,
            "experimental_condition": candidate.experimental_condition,
            "organism_cell_line": candidate.organism_cell_line,
            "nuclease": candidate.nuclease,
            "assay_type": candidate.assay_type,
            "derived_or_processed_status": candidate.derived_or_processed_status,
            "provenance_notes": candidate.provenance_notes,
            "phase16_carryover": candidate.phase16_carryover,
            "files": file_records,
            "primary_file_located": verified_count > 0,
            "acquisition_status": status,
            "stop_reason": candidate.stop_reason,
            "unresolved_items": candidate.unresolved_items,
        })
    canonical_after = protected_hashes()
    return {
        "phase": "17A",
        "protocol_version": "phase17a-public-dataset-recovery-v1",
        "scope": "public_primary_file_acquisition_and_structure_audit_only",
        "access_date": access_date,
        "candidate_count": len(records),
        "candidates": records,
        "phase16_carryover_candidate_ids": [r["candidate_id"] for r in records if r["phase16_carryover"]],
        "new_discovery_candidate_ids": [r["candidate_id"] for r in records if not r["phase16_carryover"]],
        "primary_files_recovered": [
            {
                "candidate_id": r["candidate_id"],
                "local_raw_path": f["local_raw_path"],
                "sha256": f["sha256"],
            }
            for r in records
            for f in r["files"]
            if f["verified_local_file"]
        ],
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
            "no_training_dataset_created": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_near_duplicate_analysis": True,
            "no_reverse_complement_matching": True,
            "no_continual_learning": True,
            "no_dataset_acceptance": True,
        },
    }


def provenance_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """Extract compact machine-readable provenance metadata."""
    return {
        "phase": payload["phase"],
        "protocol_version": payload["protocol_version"],
        "access_date": payload["access_date"],
        "records": [
            {
                "candidate_id": record["candidate_id"],
                "source_study_id": record["source_study_id"],
                "publication": record["publication"],
                "doi_or_pubmed": record["doi_or_pubmed"],
                "repository": record["repository"],
                "accession": record["accession"],
                "official_dataset_url": record["official_dataset_url"],
                "parent_dataset": record["parent_dataset"],
                "subset_id": record["subset_id"],
                "acquisition_status": record["acquisition_status"],
                "files": [
                    {
                        "original_filename": f["original_filename"],
                        "local_raw_path": f["local_raw_path"],
                        "source_url": f["source_url"],
                        "sha256": f["sha256"],
                        "verified_local_file": f["verified_local_file"],
                    }
                    for f in record["files"]
                ],
                "unresolved_items": record["unresolved_items"],
            }
            for record in payload["candidates"]
        ],
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
