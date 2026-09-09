#!/usr/bin/env python3
"""Phase 16D - exact overlap and independence audit only."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_landscape.overlap import build_phase16d_overlap_audit


def _git(args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 16D Exact Overlap / Independence Audit",
        "",
        "**Status:** " + payload["status"],
        "**Scope:** Exact overlap and provenance independence only.",
        "",
        "No label rescue, sequence transformation, dataset pooling, "
        "distribution analysis, model training, or model evaluation was "
        "performed. Moreno raw data was not accessed.",
        "",
        "## Canonical Reference",
        "",
        f"- DeepSpCas9 rows: {payload['canonical_reference']['n_rows']}",
        f"- Unique 30-mers: {payload['canonical_reference']['n_unique_30mer']}",
        f"- Unique guides: {payload['canonical_reference']['n_unique_guides']}",
        "",
        "## Candidate Results",
        "",
        "| candidate_id | sequence_available | records | 30mer overlap | guide overlap | duplicate excess | label conflicts | RC status | independence |",
        "|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for record in payload["candidate_reports"]:
        ov30 = record["exact_30mer_vs_deepspcas9"]
        ovg = record["exact_guide_vs_deepspcas9"]
        ov30_txt = ov30 if isinstance(ov30, str) else str(ov30["shared_unique"])
        ovg_txt = ovg if isinstance(ovg, str) else str(ovg["shared_unique"])
        lines.append(
            f"| {record['candidate_id']} | {record['sequence_available']} | "
            f"{record['sequence_record_count']} | {ov30_txt} | {ovg_txt} | "
            f"{record['duplicate_sequences']['duplicate_record_excess']} | "
            f"{record['identical_sequence_label_conflicts']['n_conflicting_sequences']} | "
            f"{record['rc_overlap_status']['status']} | "
            f"{record['provenance_relationship']['independence_assessment']} |"
        )
    lines.extend(["", "## Pairwise Candidate Overlap", ""])
    for key, value in payload["pairwise_candidate_overlap"].items():
        guide = value["exact_guide"]
        guide_txt = guide if isinstance(guide, str) else str(guide["shared_unique"])
        seq30 = value["exact_30mer"]
        seq30_txt = seq30 if isinstance(seq30, str) else str(seq30["shared_unique"])
        lines.append(f"- `{key}`: exact 30-mer `{seq30_txt}`, exact guide `{guide_txt}`, RC `{value['rc_overlap_status']['status']}`")
    lines.extend([
        "",
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
    payload = build_phase16d_overlap_audit(datetime.now().date().isoformat())
    payload["timestamp"] = datetime.now().isoformat()
    payload["git"] = {"head": _git(["rev-parse", "HEAD"]), "status_before_output": _git(["status", "--porcelain"])}
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"phase16d_overlap_independence_audit_{timestamp}.json"
    report_path = docs_dir / "phase16d_overlap_independence_audit_report.md"
    payload["artifacts"] = {"overlap_json": str(result_path), "report_md": str(report_path)}
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report_path.write_text(render_report(payload))
    print("Phase 16D overlap/independence audit complete")
    print(f"Overlap: {result_path}")
    print(f"Report: {report_path}")
    print("No Moreno raw data accessed")
    print("No model training/evaluation performed")
    print("No dataset pooling performed")
    return payload


if __name__ == "__main__":
    main()
