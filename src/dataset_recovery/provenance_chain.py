"""Utilities for Phase 17H provenance-chain validation."""

from __future__ import annotations

from typing import Dict


PROVENANCE_STATUSES = {"VERIFIED_PRIMARY", "PARTIALLY_VERIFIED", "UNVERIFIED", "NOT_PRIMARY"}


def validate_provenance_chain(record: Dict[str, object]) -> Dict[str, object]:
    """Validate publication -> file -> table -> sequence -> label evidence."""
    required = [
        "primary_publication",
        "primary_file",
        "table_or_sheet",
        "sequence_column",
        "label_column",
    ]
    missing = [field for field in required if not record.get(field)]
    primary_measurement = bool(record.get("primary_experimental_measurement"))
    if not missing and primary_measurement:
        status = "VERIFIED_PRIMARY"
    elif len(missing) < len(required) and primary_measurement:
        status = "PARTIALLY_VERIFIED"
    elif missing:
        status = "UNVERIFIED"
    else:
        status = "NOT_PRIMARY"
    return {
        "status": status,
        "missing": missing,
        "chain_complete": status == "VERIFIED_PRIMARY",
    }
