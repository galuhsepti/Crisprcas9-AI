#!/usr/bin/env python3
"""Run Phase 17B primary-file verification audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.verification import dump_json, verify_phase17b


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


def _file_list(files: list[dict]) -> str:
    if not files:
        return "not recovered"
    return "<br>".join(f"`{f['exact_local_filename']}`<br>`{f['sha256']}`" for f in files)


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17B - Primary-File Verification Audit",
        "",
        "## 1. Objective",
        "",
        "Phase 17B verifies primary-file identity, provenance, raw sequence structure, and raw label structure for Phase 17A candidates. It does not make final dataset acceptance decisions.",
        "",
        "No modeling, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, overlap analysis, reverse-complement matching, continual learning, or dataset acceptance was performed.",
        "",
        "## 2. Candidate Verification Table",
        "",
        "| Candidate | Publication | Repository/Accession | Local file(s) | Primary/derived/processed status | Verification status | Unresolved issues |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in payload["candidates"]:
        repo = record["repository_accession"]["repository"]
        accession = record["repository_accession"]["accession"]
        unresolved = ", ".join(record["unresolved_issues"][:6])
        lines.append(
            f"| `{record['candidate_id']}` | {record['publication']} | {repo}; {accession} | "
            f"{_file_list(record['files'])} | {record['primary_vs_derived_processed_status']} | "
            f"`{record['verification_status']}` | {unresolved} |"
        )
    lines.extend(["", "## 3. CRISPRon Archive Analysis", ""])
    crispron = _by_id(payload, "crispron_2021_rth_tools")
    if crispron["files"]:
        inspection = crispron["files"][0]["inspection"]
        lines.append(f"- Archive members: {inspection['archive_member_count']}")
        lines.append(f"- Documentation/code files observed: {len(inspection['documentation_or_license_files'])} documentation/license, {len(inspection['code_files'])} code files.")
        lines.append(f"- Data-like files observed: {inspection['data_like_files']}")
        lines.append(f"- Model-artifact-like files observed: {len(inspection['model_artifact_like_files'])}")
        lines.append("- Experimental training table found: `False`")
        lines.append("- Preserved status: `PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED`")
    lines.extend(["", "## 4. CRISPRpred File-by-File Analysis", ""])
    crisprpred = _by_id(payload, "crisprpredseq_2020_bmc_additional_files")
    for item in crisprpred["files"]:
        inspection = item["inspection"]
        lines.append(f"### `{item['exact_local_filename']}`")
        lines.append(f"- Rows: {inspection['row_count']}")
        lines.append(f"- Columns: `{inspection['column_names']}`")
        lines.append(f"- Sequence columns: `{list(inspection['sequence_columns'])}`")
        lines.append(f"- Label columns: `{list(inspection['label_columns'])}`")
        lines.append("- File interpretation: processed/partitioned binary-label supplementary CSV; no merge or label conversion performed.")
        lines.append("")
    lines.extend(["## 5. Doench Workbook-by-Workbook Analysis", ""])
    doench = _by_id(payload, "doench_2016_orcs_publication_screens")
    for item in doench["files"]:
        inspection = item["inspection"]
        lines.append(f"### `{item['exact_local_filename']}`")
        lines.append(f"- Sheet count: {inspection['sheet_count']}")
        lines.append(f"- Sheet names: `{inspection['sheet_names']}`")
        lines.append("- File interpretation: publication/ORCS screen workbook; screen-derived statistics are not transformed into activity.")
        lines.append("")
    lines.extend(["## 6. DeepHF / SgDesigner / Corsi Recovery Status", ""])
    for cid in ["deep_hf_2019_public_data_listing", "sgdesigner_2020_public_data_listing", "corsi_2022_free_energy_pam_context"]:
        record = _by_id(payload, cid)
        lines.append(f"- `{cid}`: `{record['verification_status']}`; {record['primary_vs_derived_processed_status']}.")
    lines.extend([
        "",
        "## 7. Canonical Integrity",
        "",
        f"Canonical protected state unchanged: `{payload['canonical_integrity']['unchanged']}`.",
        "",
        "## 8. Final Phase 17B Status",
        "",
        "Primary-file verification is complete for recovered files; acceptance remains deferred.",
    ])
    return "\n".join(lines) + "\n"


def _by_id(payload: dict, candidate_id: str) -> dict:
    return next(record for record in payload["candidates"] if record["candidate_id"] == candidate_id)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = verify_phase17b()
    payload["timestamp"] = now.isoformat()
    payload["git"] = {
        "head": git_head(),
        "status_before_output": git_status(),
    }
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase17b_primary_file_verification_{stamp}.json"
    report_path = docs_dir / "phase17b_primary_file_verification_report.md"
    payload["artifacts"] = {
        "verification_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17B primary-file verification complete: {result_path}")
    print(f"Phase 17B report: {report_path}")
    print("No modeling, Moreno access, pooling, transformation, or acceptance performed")
    return payload


if __name__ == "__main__":
    main()
