"""Conservative provenance schema for Phase 16A.

Phase 16A records discovery evidence only. It does not classify compatibility,
load Moreno-Mateos raw data, run overlap analyses, or touch any model artifact.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

UNKNOWN = "UNKNOWN"
PENDING_PRIMARY_VERIFICATION = "PENDING_PRIMARY_VERIFICATION"

REQUIRED_IDENTITY_FIELDS = [
    "source_study_id",
    "source_publication",
    "source_repository",
    "parent_dataset",
    "subset_id",
    "experimental_condition",
    "library_design",
    "derived_or_processed_status",
]

REQUIRED_PROVENANCE_FIELDS = [
    "candidate_id",
    "candidate_name",
    *REQUIRED_IDENTITY_FIELDS,
    "accession",
    "source_url",
    "access_date",
    "original_filename",
    "local_raw_path",
    "sha256",
    "file_format",
    "assay",
    "organism",
    "cell_line_or_system",
    "Cas9_variant",
    "sequence_definition",
    "PAM_definition",
    "strand_orientation_documentation",
    "activity_measurement",
    "activity_units",
    "continuous_status",
    "experimental_measured_status",
    "original_preprocessing",
    "duplicate_policy",
    "known_subset_or_processed_relationships",
    "discovery_status",
    "evidence_notes",
]


def explicit_unknown_record() -> Dict[str, str | None]:
    """Return explicit unknown defaults for nullable Phase 16A fields."""
    return {field: UNKNOWN for field in REQUIRED_PROVENANCE_FIELDS}


def make_record(**overrides: object) -> Dict[str, object]:
    """Create a complete provenance record with explicit unknowns."""
    record: Dict[str, object] = explicit_unknown_record()
    record.update(overrides)
    record["discovery_status"] = PENDING_PRIMARY_VERIFICATION
    return record


def missing_required_fields(record: Mapping[str, object]) -> List[str]:
    """List required schema fields absent from a candidate record."""
    return [field for field in REQUIRED_PROVENANCE_FIELDS if field not in record]


def unknown_identity_fields(record: Mapping[str, object]) -> List[str]:
    """List identity fields still unknown after discovery."""
    return [
        field for field in REQUIRED_IDENTITY_FIELDS
        if record.get(field) in (None, "", UNKNOWN)
    ]


def validate_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Validate Phase 16A provenance completeness without accepting data."""
    missing = missing_required_fields(record)
    unknown_identity = unknown_identity_fields(record)
    return {
        "candidate_id": record.get("candidate_id", UNKNOWN),
        "schema_complete": not missing,
        "missing_required_fields": missing,
        "unknown_identity_fields": unknown_identity,
        "status_is_pending_primary_verification": (
            record.get("discovery_status") == PENDING_PRIMARY_VERIFICATION
        ),
        "identity_not_name_only": bool(
            record.get("candidate_name")
            and any(record.get(field) not in (None, "", UNKNOWN)
                    for field in REQUIRED_IDENTITY_FIELDS)
        ),
    }


def validate_inventory(records: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Validate all candidate records in deterministic order."""
    return [validate_record(record) for record in records]
