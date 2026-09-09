#!/usr/bin/env python3
"""Phase 16C - compatibility audit only."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_landscape.compatibility import build_phase16c_compatibility


def _git(args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 16C Compatibility Audit",
        "",
        "**Status:** " + payload["status"],
        "**Scope:** Biological/measurement compatibility only.",
        "",
        "No label harmonization, sequence transformation, overlap analysis, "
        "distribution analysis, pooling, model training, or model evaluation "
        "was performed. Moreno raw data was not accessed.",
        "",
        "## Candidate Decisions",
        "",
        "| candidate_id | compatibility_status | reason | transformations |",
        "|---|---|---|---|",
    ]
    for record in payload["records"]:
        transforms = ", ".join(record["required_transformations"]) or "NONE"
        lines.append(
            f"| {record['candidate_id']} | {record['compatibility_status']} | "
            f"{record['compatibility_reason']} | {transforms} |"
        )
    lines.extend(["", "## File-Level Evidence", ""])
    for record in payload["records"]:
        lines.append(f"### {record['candidate_id']}")
        lines.append("")
        if not record["files"]:
            lines.append("- No new Phase 16C file-level evidence beyond unresolved Phase 16B status.")
            lines.append("")
            continue
        for file_result in record["files"]:
            lines.append(f"- File: `{file_result['file']}`")
            lines.append(f"  Status: `{file_result['compatibility_status']}`")
            lines.append(f"  Sequence schema: `{file_result['sequence_schema']}` length `{file_result['sequence_length']}`")
            lines.append(f"  Activity: {file_result['activity_definition']}")
            lines.append(f"  Required transformation: `{file_result['required_transformation']}`; performed: `NOT_PERFORMED`")
        lines.append("")
    lines.extend([
        "## Locks",
        "",
        "- No Moreno raw data accessed.",
        "- No model training/evaluation performed.",
        "- No dataset pooling performed.",
        "- No canonical Phase 3-15 artifacts modified.",
    ])
    return "\n".join(lines) + "\n"


def main() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = build_phase16c_compatibility(datetime.now().date().isoformat())
    payload["timestamp"] = datetime.now().isoformat()
    payload["git"] = {"head": _git(["rev-parse", "HEAD"]), "status_before_output": _git(["status", "--porcelain"])}
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"phase16c_compatibility_audit_{timestamp}.json"
    report_path = docs_dir / "phase16c_compatibility_audit_report.md"
    payload["artifacts"] = {"compatibility_json": str(result_path), "report_md": str(report_path)}
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report_path.write_text(render_report(payload))
    print("Phase 16C compatibility audit complete")
    print(f"Compatibility: {result_path}")
    print(f"Report: {report_path}")
    print("No Moreno raw data accessed")
    print("No model training/evaluation performed")
    print("No dataset pooling performed")
    return payload


if __name__ == "__main__":
    main()
