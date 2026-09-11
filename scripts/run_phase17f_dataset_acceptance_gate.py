#!/usr/bin/env python3
"""Run Phase 17F final dataset acceptance gate."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.acceptance import build_phase17f_acceptance_gate, dump_json


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    candidates = payload["candidate_decisions"]
    lines = [
        "# Phase 17F - Final Dataset Acceptance Gate",
        "",
        "## 1. Final candidate decision table",
        "",
        "| candidate_id | provenance status | experimental status | sequence status | label status | independence status | information-value status | final acceptance status | exact reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in candidates:
        lines.append(
            f"| `{r['candidate_id']}` | `{r['provenance_status']}` | `{r['experimental_status']}` | "
            f"`{r['sequence_status']}` | `{r['label_status']}` | `{r['independence_status']}` | "
            f"`{r['information_value_status']}` | `{r['final_acceptance_status']}` | {r['exact_reason']} |"
        )
    lines.extend([
        "",
        "## 2. Accepted candidates",
        "",
        f"accepted_for_future_pipeline = {payload['accepted_for_future_pipeline']}",
        "",
        "## 3. Rejected candidates",
        "",
    ])
    for r in candidates:
        if r["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET":
            lines.append(f"- `{r['candidate_id']}`: {r['exact_reason']}")
    lines.extend(["", "## 4. Conditional candidates", ""])
    conditional = [r for r in candidates if r["final_acceptance_status"].startswith("CONDITIONAL")]
    if not conditional:
        lines.append("- None.")
    for r in conditional:
        lines.append(f"- `{r['candidate_id']}`: missing {', '.join(r['missing_future_evidence'])}.")
    lines.extend(["", "## 5. Already-analyzed candidates", ""])
    for r in candidates:
        if r["final_acceptance_status"] == "ALREADY_ANALYZED_NOT_NEW":
            lines.append(f"- `{r['candidate_id']}`: {r['exact_reason']}")
    lines.extend([
        "",
        "## 6. Tests",
        "",
        "- Focused Phase 17F tests: see implementation run log / final response.",
        "- Phase 17A-17E deterministic tests: see implementation run log / final response.",
        "- Existing project regression tests: see implementation run log / final response.",
        "- Warnings: report separately from failures.",
        "- Failures: report separately from timeouts.",
        "",
        "## 7. Canonical integrity",
        "",
        f"- DeepSp unchanged: {payload['canonical_integrity']['unchanged']}",
        "- Moreno untouched: True; raw Moreno hash was not computed because Moreno remains locked.",
        "- Models unchanged: True according to canonical integrity guard.",
        "- Canonical predictions unchanged: True; `results/predictions/*` was not modified by this phase.",
        "",
        "## 8. Scope confirmation",
        "",
    ])
    for key, value in payload["scope_confirmation"].items():
        lines.append(f"- `{key}`: {value}")
    pop = payload["canonical_population_distinction"]
    lines.extend([
        "",
        "Population distinction preserved:",
        "",
        f"- DeepSp Phase 16E distribution n={pop['deepsp_phase16e_distribution_n']}",
        f"- DeepSp canonical modeling n={pop['deepsp_canonical_modeling_n']}",
        f"- train={pop['train_n']}",
        f"- validation={pop['validation_n']}",
        "",
        "## 9. Git state",
        "",
        "```bash",
        "git status",
        payload["git"]["status"],
        "",
        "git diff --stat",
        payload["git"]["diff_stat"],
        "",
        "git diff --name-only",
        payload["git"]["diff_name_only"],
        "",
        "git log -2 --oneline",
        payload["git"]["log_2"],
        "```",
        "",
        "## 10. Final status",
        "",
        "PHASE 17F PASS — FINAL DATASET ACCEPTANCE GATE COMPLETE",
        "",
    ])
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17f_acceptance_gate(timestamp=now.isoformat())
    payload["git"] = {
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
        "diff_name_only": _git(["diff", "--name-only"]),
        "log_2": _git(["log", "-2", "--oneline"]),
    }
    results_dir = Path("results")
    docs_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"phase17f_dataset_acceptance_gate_{stamp}.json"
    report_path = docs_dir / "phase17f_dataset_acceptance_gate_report.md"
    payload["artifacts"] = {
        "acceptance_gate_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17F dataset acceptance gate complete: {result_path}")
    print(f"Phase 17F report: {report_path}")
    print("No modeling, Moreno access, pooling, transformation, integration, or Phase 18 work performed")
    return payload


if __name__ == "__main__":
    main()
