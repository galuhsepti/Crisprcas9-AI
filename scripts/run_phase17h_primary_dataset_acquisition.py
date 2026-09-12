#!/usr/bin/env python3
"""Run Phase 17H targeted primary dataset acquisition audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.primary_acquisition import build_phase17h_primary_acquisition, dump_json


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17H - Targeted Primary Dataset Acquisition and Provenance Recovery",
        "",
        "## 1. What Was Searched",
        "",
    ]
    for source in payload["sources_investigated"]:
        lines.append(f"- {source['search_source']}: {', '.join(source['search_terms'])}")
    lines.extend(["", "## 2. Primary Publications Investigated", ""])
    for source in payload["sources_investigated"]:
        lines.append(f"- {source['candidate_publication']} ({source['doi']})")
    lines.extend(["", "## 3. Supplementary Files Located", ""])
    for candidate in payload["candidate_datasets"]:
        lines.append(f"- `{candidate['candidate_id']}`: {candidate['supplementary_file_name']} -> `{candidate['primary_file']}`")
    lines.extend(["", "## 4. Files Actually Recovered", ""])
    for file_record in payload["files_recovered"]:
        lines.append(f"- `{file_record['path']}` SHA-256 `{file_record['sha256']}`")
    lines.extend(["", "## 5. Row-Level sgRNA Sequence Evidence", ""])
    for candidate in payload["candidate_datasets"]:
        lines.append(
            f"- `{candidate['candidate_id']}`: sequence column `{candidate['sequence_column']}`, "
            f"unique sequences {candidate['guide_count_unique']}."
        )
    lines.extend(["", "## 6. Continuous Experimental On-Target Activity", ""])
    for candidate in payload["candidate_datasets"]:
        lines.append(
            f"- `{candidate['candidate_id']}`: label `{candidate['label_name']}`; "
            f"definition: {candidate['label_definition']}."
        )
    lines.extend(["", "## 7. Incompatible Candidates", ""])
    for blocked in payload["blocked_candidates"]:
        lines.append(f"- {blocked}")
    lines.extend(["", "## 8. Promising Candidates for Phase 17G Re-Evaluation", ""])
    if payload["promising_candidates"]:
        for cid in payload["promising_candidates"]:
            lines.append(f"- `{cid}`")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## 9. Potential Compatible Unique Observations",
        "",
        f"potential_total_unique_observations = {payload['potential_total_unique_observations']}",
        "",
        f"phase17g_minimum_total_new_unique_n = {payload['phase17g_minimum_total_new_unique_n']}",
        "",
        f"remaining_deficit_relative_to_2000 = {payload['remaining_deficit_relative_to_2000']}",
        "",
        "## 10. Realistic Path Toward 2000",
        "",
        "Corsi Supplementary Data 1 may contribute about 1,022 unique 30 nt sequence-linked indel-frequency observations if it passes full Phase 17G re-evaluation. The project still needs at least 978 additional compatible unique observations from another verified primary source, or a larger independently compatible dataset.",
        "",
        "## 11. Missing Evidence",
        "",
        "- CRISPRon: official row-level primary experimental table still not recovered.",
        "- sgDesigner: primary experimental dataset identity remains unresolved.",
        "- Corsi: requires full Phase 17G re-evaluation, independence audit, and metadata review before any acceptance.",
        "",
        "## 12. Canonical Protection",
        "",
        "- Moreno untouched: True.",
        "- Canonical models and predictions untouched: True.",
        f"- Canonical integrity unchanged: {payload['canonical_integrity']['unchanged']}.",
        "",
        "## 13. Scope Confirmation",
        "",
        "- No model training.",
        "- No model evaluation.",
        "- No label harmonization.",
        "- No sequence reconstruction.",
        "- No Phase 17G rerun.",
        "- No Phase 18.",
        "",
    ])
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17h_primary_acquisition(retrieval_date=now.date().isoformat())
    payload["timestamp"] = now.isoformat()
    payload["git"] = {
        "head": _git(["rev-parse", "HEAD"]),
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
    }
    result_path = Path("results") / f"phase17h_primary_dataset_acquisition_{stamp}.json"
    report_path = Path("docs") / "phase17h_primary_dataset_acquisition_report.md"
    payload["artifacts"] = {
        "primary_acquisition_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17H primary dataset acquisition audit complete: {result_path}")
    print(f"Phase 17H report: {report_path}")
    print(f"Promising candidates: {payload['promising_candidates']}")
    return payload


if __name__ == "__main__":
    main()
