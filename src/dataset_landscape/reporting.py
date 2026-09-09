"""Reporting helpers for Phase 16A."""

from __future__ import annotations

from typing import Mapping


def render_phase16a_report(inventory: Mapping[str, object]) -> str:
    """Render a concise human-readable Phase 16A report."""
    lines = [
        "# Phase 16A Dataset Discovery / Provenance Inventory",
        "",
        "**Status:** " + str(inventory["status"]),
        "**Scope:** Discovery and provenance inventory only.",
        "",
        "No model was trained, tuned, calibrated, adapted, or evaluated. "
        "Moreno raw data was not accessed. No reverse-complement overlap or "
        "near-duplicate threshold analysis was performed.",
        "",
        "## Discovery Sources Consulted",
        "",
    ]
    for source in inventory["discovery_sources_consulted"]:
        lines.append(
            f"- {source['label']}: {source['url']} ({source['role']})"
        )
    lines.extend(["", "## Candidate Inventory", ""])
    lines.append(
        "| candidate_id | candidate_name | source_repository | status | primary evidence |"
    )
    lines.append("|---|---|---|---|---|")
    for cand in inventory["candidates"]:
        evidence = cand.get("original_filename") or "UNKNOWN"
        lines.append(
            "| {candidate_id} | {candidate_name} | {source_repository} | "
            "{discovery_status} | {evidence} |".format(
                candidate_id=cand["candidate_id"],
                candidate_name=cand["candidate_name"],
                source_repository=cand["source_repository"],
                discovery_status=cand["discovery_status"],
                evidence=evidence,
            )
        )
    lines.extend(["", "## Provenance Questions Remaining", ""])
    for validation in inventory["provenance_validation"]:
        cid = validation["candidate_id"]
        unknown = validation["unknown_identity_fields"]
        if unknown:
            lines.append(f"- `{cid}` unknown identity fields: {', '.join(unknown)}")
        else:
            lines.append(f"- `{cid}` identity fields populated for Phase 16A discovery.")
    lines.extend([
        "",
        "## Explicit Limits",
        "",
        "- Every candidate remains `PENDING_PRIMARY_VERIFICATION`.",
        "- Dataset identity is not determined from candidate name alone.",
        "- Literature mention alone is not treated as primary-data verification.",
        "- Compatibility, overlap, distribution analysis, and gate decisions are deferred.",
    ])
    return "\n".join(lines) + "\n"
