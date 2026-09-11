#!/usr/bin/env python3
"""Run Phase 17C sequence/label compatibility audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.compatibility import build_phase17c_compatibility, dump_json


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


def _by_id(payload: dict, candidate_id: str) -> dict:
    return next(record for record in payload["candidates"] if record["candidate_id"] == candidate_id)


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 17C - Sequence and Label Compatibility Audit",
        "",
        "## 1. Objective",
        "",
        "Phase 17C audits whether recovered Phase 17 candidates have observed sequence structure and documented label semantics compatible with the thesis target. It does not accept datasets or create training data.",
        "",
        "No model training, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, 30-mer construction, reverse-complement matching, near-duplicate analysis, continual learning, or dataset acceptance was performed.",
        "",
        "## 2. Candidate-Level Compatibility Table",
        "",
        "| Candidate | Sequence status | Observed sequence geometry | Label status | Label semantics | Overall status |",
        "|---|---|---|---|---|---|",
    ]
    for record in payload["candidates"]:
        lines.append(
            f"| `{record['candidate_id']}` | `{record['sequence_status']}` | "
            f"{record['observed_sequence_geometry']} | `{record['label_status']}` | "
            f"{record['label_semantics']} | `{record['overall_status']}` |"
        )
    lines.extend(["", "## 3. CRISPRpred File-by-File Results", ""])
    crisprpred = _by_id(payload, "crisprpredseq_2020_bmc_additional_files")
    for seq_result in crisprpred["sequence_file_results"]:
        file_labels = [
            item for item in crisprpred["label_file_results"]
            if item.get("file") == seq_result.get("file")
        ]
        label_status = ", ".join(sorted({item["label_status"] for item in file_labels}))
        lines.append(f"- `{seq_result['file']}`: sequence `{seq_result['sequence_status']}`; lengths `{seq_result['length_distribution']}`; labels `{label_status}`.")
    lines.extend(["", "## 4. Doench Workbook-by-Workbook Results", ""])
    doench = _by_id(payload, "doench_2016_orcs_publication_screens")
    for seq_result in doench["sequence_file_results"]:
        lines.append(
            f"- `{seq_result.get('file')}` / sheet `{seq_result.get('sheet_name')}`: "
            f"sequence `{seq_result.get('sequence_status')}`; geometry {seq_result.get('observed_geometry')}; no transformation performed."
        )
    lines.extend(["", "## 5. CRISPRon Results", ""])
    crispron = _by_id(payload, "crispron_2021_rth_tools")
    lines.append("Verified experimental activity table exists: `False`.")
    lines.append(f"Sequence status: `{crispron['sequence_status']}`.")
    lines.append(f"Label status: `{crispron['label_status']}`.")
    lines.extend(["", "## 6. Unrecovered Candidates", ""])
    for cid in ["deep_hf_2019_public_data_listing", "sgdesigner_2020_public_data_listing", "corsi_2022_free_energy_pam_context"]:
        record = _by_id(payload, cid)
        lines.append(f"- `{cid}`: sequence `{record['sequence_status']}`, label `{record['label_status']}`, overall `{record['overall_status']}`.")
    lines.extend([
        "",
        "## 7. Safety Checks",
        "",
        "- Numeric ranges were not treated as activity without source semantics.",
        "- Binary labels were not converted into continuous labels.",
        "- Guide-only or 23-mer sequences were not converted into canonical 30-mers.",
        "- PAM/flanking sequence was not invented.",
        "- Rank/enrichment/log-fold-change fields were not treated as activity.",
        "- Processed benchmark files were not treated as independent primary experiments.",
        "",
        "## 8. Final Phase 17C Status",
        "",
        "Sequence/label compatibility audit complete; acceptance remains deferred.",
    ])
    return "\n".join(lines) + "\n"


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17c_compatibility()
    payload["timestamp"] = now.isoformat()
    payload["git"] = {"head": git_head(), "status_before_output": git_status()}
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase17c_sequence_label_compatibility_{stamp}.json"
    report_path = docs_dir / "phase17c_sequence_label_compatibility_report.md"
    payload["artifacts"] = {
        "compatibility_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17C compatibility audit complete: {result_path}")
    print(f"Phase 17C report: {report_path}")
    print("No modeling, Moreno access, pooling, transformation, 30-mer construction, or acceptance performed")
    return payload


if __name__ == "__main__":
    main()
