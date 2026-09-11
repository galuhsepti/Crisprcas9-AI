#!/usr/bin/env python3
"""Run Phase 17A public dataset recovery/acquisition audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.phase17a import dump_json, provenance_payload, recover_phase17a


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or "(clean)"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def _file_summary(files: list[dict]) -> str:
    if not files:
        return "none"
    verified = [f for f in files if f["verified_local_file"]]
    if not verified:
        return "none verified"
    return "<br>".join(
        f"`{f['local_raw_path']}`<br>`{f['sha256']}`"
        for f in verified
    )


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17A - Public Dataset Recovery and Acquisition Audit",
        "",
        "## 1. Objective",
        "",
        "Phase 17A is an acquisition/provenance audit only. It recovers public candidate primary files where possible, preserves original files under `data/phase17/raw/`, records SHA-256 identities, and inspects file structure only.",
        "",
        "No model training, model evaluation, tuning, pooling, label transformation, label harmonization, Moreno access, reverse-complement matching, near-duplicate analysis, continual learning, or dataset acceptance was performed.",
        "",
        "## 2. Candidate Summary",
        "",
        "| Candidate | Carry-over/New | Primary file located? | Status | Local files / SHA-256 | Key unresolved items |",
        "|---|---|---:|---|---|---|",
    ]
    for record in payload["candidates"]:
        carry = "Phase 16 carry-over" if record["phase16_carryover"] else "New discovery"
        unresolved = ", ".join(record["unresolved_items"][:4])
        lines.append(
            f"| `{record['candidate_id']}` | {carry} | {record['primary_file_located']} | "
            f"`{record['acquisition_status']}` | {_file_summary(record['files'])} | {unresolved} |"
        )
    lines.extend([
        "",
        "## 3. Recovered Primary Files",
        "",
    ])
    recovered = payload["primary_files_recovered"]
    if not recovered:
        lines.append("No primary files were recovered.")
    for item in recovered:
        lines.append(f"- `{item['local_raw_path']}`; SHA-256 `{item['sha256']}`; candidate `{item['candidate_id']}`")
    lines.extend([
        "",
        "## 4. Candidates Not Recovered",
        "",
    ])
    for record in payload["candidates"]:
        if not record["primary_file_located"]:
            reason = record["stop_reason"] or record["provenance_notes"]
            lines.append(f"- `{record['candidate_id']}`: {record['acquisition_status']} - {reason}")
    lines.extend([
        "",
        "## 5. Structure Inspection Only",
        "",
        "The audit records row counts, column names, obvious sequence columns, obvious label columns, sheet names, missingness, and duplicate rows where technically available. It does not decide compatibility with the canonical target.",
        "",
        "## 6. Canonical Integrity",
        "",
        f"Canonical protected-file hashes unchanged: `{payload['canonical_integrity']['unchanged']}`.",
        "",
        "## 7. Final Phase 17A Status",
        "",
        "Phase 17A stops after public-file recovery and provenance/structure recording. Dataset acceptance is deferred to later gates.",
    ])
    return "\n".join(lines) + "\n"


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = recover_phase17a(access_date=now.date().isoformat())
    payload["timestamp"] = now.isoformat()
    payload["git"] = {
        "head": git_head(),
        "status_before_output": git_status(),
    }

    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    recovery_path = results_dir / f"phase17a_dataset_recovery_{stamp}.json"
    provenance_path = results_dir / f"phase17a_provenance_{stamp}.json"
    report_path = docs_dir / "phase17a_dataset_recovery_report.md"

    payload["artifacts"] = {
        "dataset_recovery_json": str(recovery_path).replace("\\", "/"),
        "provenance_json": str(provenance_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(recovery_path, payload)
    dump_json(provenance_path, provenance_payload(payload))
    report_path.write_text(render_report(payload), encoding="utf-8")

    print(f"Phase 17A dataset recovery audit complete: {recovery_path}")
    print(f"Phase 17A provenance metadata: {provenance_path}")
    print(f"Phase 17A report: {report_path}")
    print("No modeling, Moreno access, pooling, label transformation, or dataset acceptance performed")
    return payload


if __name__ == "__main__":
    main()
