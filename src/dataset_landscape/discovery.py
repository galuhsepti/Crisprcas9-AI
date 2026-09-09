"""Phase 16A discovery assembly."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

from .candidates import candidate_inventory
from .provenance import PENDING_PRIMARY_VERIFICATION, validate_inventory


DISCOVERY_SOURCES_CONSULTED = [
    {
        "label": "BioGRID ORCS Dataset 19",
        "url": "https://orcs.thebiogrid.org/Dataset/19",
        "role": "primary repository / publication supplementary file index",
    },
    {
        "label": "RTH-tools CRISPRon GitHub",
        "url": "https://github.com/RTH-tools/crispron",
        "role": "official project repository",
    },
    {
        "label": "CRISPRon webserver",
        "url": "https://rth.dk/resources/crispr/crispron/",
        "role": "official project page",
    },
    {
        "label": "CRISPRpred(SEQ) BMC Bioinformatics article",
        "url": "https://link.springer.com/article/10.1186/s12859-020-3531-9",
        "role": "publisher-hosted supplementary-file index",
    },
    {
        "label": "dagrate/public_data_crisprCas9",
        "url": "https://github.com/dagrate/public_data_crisprCas9",
        "role": "public catalog used only to discover primary source links",
    },
]


def build_phase16a_inventory(access_date: str | None = None) -> Dict[str, object]:
    """Build deterministic Phase 16A discovery/provenance inventory."""
    if access_date is None:
        access_date = date.today().isoformat()
    candidates = candidate_inventory(access_date)
    validations = validate_inventory(candidates)
    stop = len(candidates) == 0
    return {
        "phase": "16A",
        "scope": "discovery_provenance_inventory_only",
        "status": "STOP_NO_AUDITABLE_PUBLIC_PRIMARY_SOURCE" if stop else "DISCOVERY_INVENTORY_CREATED",
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_calibration": True,
            "no_moreno_raw_data_accessed": True,
            "no_rc_overlap_analysis": True,
            "exact_overlap_only_no_distance_rule": True,
        },
        "access_date": access_date,
        "candidate_count": len(candidates),
        "all_candidates_initial_status": PENDING_PRIMARY_VERIFICATION,
        "discovery_sources_consulted": DISCOVERY_SOURCES_CONSULTED,
        "candidates": candidates,
        "provenance_validation": validations,
        "stop_condition": {
            "triggered": stop,
            "reason": None if not stop else "No auditable public primary source/file identified.",
        },
    }
