#!/usr/bin/env python3
"""Run Phase 17D independence and exact-overlap audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.independence import build_phase17d_independence_overlap, dump_json


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


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17D - Independence and Exact Overlap Audit",
        "",
        "## 1. Objective",
        "",
        "Phase 17D audits exact sequence/guide overlap, internal duplication, label conflicts, and provenance relationships. It does not decide acceptance.",
        "",
        "No model training, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, near-duplicate analysis, reverse-complement matching, continual learning, or dataset acceptance was performed.",
        "",
        "## 2. Candidate Overlap Table",
        "",
        "| Candidate | Sequence availability | Geometry | Exact 30-mer overlap | Exact guide overlap | Duplicate sequence excess | Label-conflict sequences | Provenance status |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for record in payload["candidate_reports"]:
        seq_overlap = record["exact_sequence_overlap_with_canonical_valid_30mer"]
        guide_overlap = record["exact_guide_overlap_with_canonical_valid_guides"]
        seq_shared = seq_overlap.get("shared_unique") if isinstance(seq_overlap, dict) else seq_overlap
        guide_shared = guide_overlap.get("shared_unique") if isinstance(guide_overlap, dict) else guide_overlap
        dup = record["duplicate_exact_sequences"]["duplicate_sequence_excess"]
        conflicts = record["label_conflict_sequences"]["conflict_sequence_count"]
        prov = record["provenance_relationship"]["status"]
        lines.append(
            f"| `{record['candidate_id']}` | {record['sequence_availability']} | {record['geometry']} | "
            f"{seq_shared} | {guide_shared} | {dup} | {conflicts} | `{prov}` |"
        )
    lines.extend(["", "## 3. CRISPRpred File-by-File Results", ""])
    crisprpred = _by_id(payload, "crisprpredseq_2020_bmc_additional_files")
    for item in crisprpred["file_reports"]:
        guide = item["exact_guide_overlap_with_canonical_valid_guides"]
        guide_shared = guide.get("shared_unique") if isinstance(guide, dict) else guide
        lines.append(
            f"- `{item['file']}`: records {item['record_count']}; guide overlap {guide_shared}; "
            f"duplicate sequence excess {item['duplicate_exact_sequences']['duplicate_sequence_excess']}; "
            f"label-conflict sequences {item['identical_sequence_label_conflicts']['conflict_sequence_count']}."
        )
    lines.extend(["", "## 4. Doench Workbook-by-Workbook Results", ""])
    doench = _by_id(payload, "doench_2016_orcs_publication_screens")
    for item in doench["file_reports"]:
        guide = item["exact_guide_overlap_with_canonical_valid_guides"]
        guide_shared = guide.get("shared_unique") if isinstance(guide, dict) else guide
        lines.append(
            f"- `{item['file']}`: records {item['record_count']}; guide overlap {guide_shared}; "
            f"duplicate sequence excess {item['duplicate_exact_sequences']['duplicate_sequence_excess']}; "
            f"label-conflict sequences {item['identical_sequence_label_conflicts']['conflict_sequence_count']}."
        )
    lines.extend(["", "## 5. Candidate-to-Candidate Overlap", ""])
    for pair, result in payload["candidate_to_candidate_overlap"].items():
        guide = result["guide_overlap"]
        guide_shared = guide.get("shared_unique") if isinstance(guide, dict) else guide
        lines.append(f"- `{pair}`: guide overlap {guide_shared}; provenance `{result['provenance_relationship']['status']}`.")
    lines.extend([
        "",
        "## 6. Interpretation Safeguards",
        "",
        "- Exact overlap was not called leakage automatically.",
        "- Zero overlap was not treated as proof of independence.",
        "- No near-duplicate analysis was performed.",
        "- Reverse-complement matching was not performed without verified orientation.",
        "",
        "## 7. Final Phase 17D Status",
        "",
        "Independence/overlap audit complete; acceptance remains deferred.",
    ])
    return "\n".join(lines) + "\n"


def _by_id(payload: dict, candidate_id: str) -> dict:
    return next(record for record in payload["candidate_reports"] if record["candidate_id"] == candidate_id)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17d_independence_overlap()
    payload["timestamp"] = now.isoformat()
    payload["git"] = {"head": git_head(), "status_before_output": git_status()}
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase17d_independence_overlap_{stamp}.json"
    report_path = docs_dir / "phase17d_independence_overlap_report.md"
    payload["artifacts"] = {
        "independence_overlap_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17D independence/overlap audit complete: {result_path}")
    print(f"Phase 17D report: {report_path}")
    print("No modeling, Moreno access, pooling, transformations, fuzzy matching, or acceptance performed")
    return payload


if __name__ == "__main__":
    main()
