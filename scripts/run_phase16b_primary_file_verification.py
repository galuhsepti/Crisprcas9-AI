#!/usr/bin/env python3
"""Phase 16B - primary experimental file verification only."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_landscape.verification import phase16b_file_verification


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "(clean)"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 16B Primary File Verification",
        "",
        "**Status:** " + payload["status"],
        "**Scope:** Primary-file verification only.",
        "",
        "No label compatibility decision, sequence transformation, overlap "
        "analysis, distribution analysis, model training, or model evaluation "
        "was performed. Moreno raw data was not accessed.",
        "",
        "## Candidate Status",
        "",
        "| candidate_id | primary_file_status | status_after_16b | files verified |",
        "|---|---|---|---:|",
    ]
    for record in payload["records"]:
        n_files = sum(1 for item in record["files"] if item["verified_local_file"])
        lines.append(
            f"| {record['candidate_id']} | {record['primary_file_status']} | "
            f"{record['candidate_status_after_16b']} | {n_files} |"
        )
    lines.extend(["", "## Verified Files", ""])
    for record in payload["records"]:
        lines.append(f"### {record['candidate_id']}")
        lines.append("")
        lines.append(record["provenance_relationship"])
        lines.append("")
        if not record["files"]:
            lines.append("- No local primary file verified in Phase 16B.")
        for item in record["files"]:
            lines.append(
                f"- `{item['local_raw_path']}`; sha256 `{item['sha256']}`; "
                f"source: {item['source_url']}"
            )
        lines.append("")
        if record["unresolved_issues"]:
            lines.append("Unresolved:")
            for issue in record["unresolved_issues"]:
                lines.append(f"- {issue}")
            lines.append("")
    lines.extend([
        "## Locks",
        "",
        "- No Moreno raw data accessed.",
        "- No model training/evaluation performed.",
        "- No canonical Phase 3-15 artifacts modified.",
    ])
    return "\n".join(lines) + "\n"


def main() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    access_date = datetime.now().date().isoformat()
    payload = phase16b_file_verification(access_date)
    payload["timestamp"] = datetime.now().isoformat()
    payload["git"] = {
        "head": git_head(),
        "status_before_output": git_status(),
    }

    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase16b_primary_file_verification_{timestamp}.json"
    report_path = docs_dir / "phase16b_primary_file_verification_report.md"
    payload["artifacts"] = {
        "verification_json": str(result_path),
        "report_md": str(report_path),
    }
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report_path.write_text(render_report(payload))
    print("Phase 16B primary-file verification complete")
    print(f"Verification: {result_path}")
    print(f"Report: {report_path}")
    print("No Moreno raw data accessed")
    print("No model training/evaluation performed")
    return payload


if __name__ == "__main__":
    main()
