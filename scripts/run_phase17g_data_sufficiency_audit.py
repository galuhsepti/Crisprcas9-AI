#!/usr/bin/env python3
"""Run Phase 17G data sufficiency and readiness assessment."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.data_sufficiency import build_phase17g_data_sufficiency, dump_json


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    thresholds = payload["thresholds"]
    lines = [
        "# Phase 17G - Data Sufficiency and Readiness Assessment",
        "",
        "## 1. Why Phase 17G Exists",
        "",
        "Phase 17G defines an objective, configurable GO / NO-GO framework for deciding when recovered public CRISPR-Cas9 datasets are sufficiently usable to justify a future modeling phase. It is an audit-only readiness assessment, not a model-improvement experiment.",
        "",
        "## 2. Why Not Phase 18",
        "",
        "Phase 17F accepted no dataset for future pipeline integration. Phase 17G therefore records what would be required before moving beyond data recovery; it does not train, evaluate, tune, pool, or integrate any dataset.",
        "",
        "## 3. Hard Scientific Compatibility Requirements",
        "",
        "- Verifiable primary experimental source.",
        "- Compatible SpCas9 on-target task evidence.",
        "- Directly supported sequence geometry; no constructed 30-mers or inferred flanks.",
        "- Explicit continuous experimental activity label; no binary, rank, enrichment, log-fold-change, STARS, classifier-label, or score rescue.",
        "- Acceptable independence/provenance relationship without unresolved derivation/leakage concern.",
        "",
        "## 4. Project-Specific Policy Thresholds",
        "",
        f"- minimum_candidate_unique_n = {thresholds['minimum_candidate_unique_n']}",
        f"- recommended_candidate_unique_n = {thresholds['recommended_candidate_unique_n']}",
        f"- minimum_total_new_unique_n = {thresholds['minimum_total_new_unique_n']}",
        f"- minimum_continuous_label_unique_values = {thresholds['minimum_continuous_label_unique_values']}",
        "",
        "These are conservative project-management thresholds relative to the DeepSpCas9 canonical modeling population of 10,117. They are not universal CRISPR biology or machine-learning laws.",
        "",
        "## 5. Candidate Results",
        "",
        "| candidate_id | provenance | task | sequence valid n | unique seq n | label status | independence | quantity | information | final candidate status |",
        "|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for c in payload["candidates"]:
        lines.append(
            f"| `{c['candidate_id']}` | `{c['provenance_gate']['status']}` | `{c['task_compatibility_gate']['status']}` | "
            f"{c['valid_sequence_n']} | {c['unique_sequence_n']} | `{c['label_gate']['status']}` | "
            f"`{c['independence_gate']['status']}` | `{c['quantity_gate']['status']}` | "
            f"`{c['information_value_gate']['status']}` | `{c['final_candidate_status']}` |"
        )
    lines.extend([
        "",
        "## 6. Aggregate Accepted New Data",
        "",
        f"accepted_candidate_count = {payload['aggregate']['accepted_candidate_count']}",
        "",
        f"total_new_unique_compatible_observations = {payload['aggregate']['total_new_unique_compatible_observations']}",
        "",
        f"additional_compatible_unique_observations_required = {payload['additional_compatible_unique_observations_required']}",
        "",
        "## 7. Final Decision",
        "",
        payload["final_decision"],
        "",
        "## 8. Blocking Reasons",
        "",
    ])
    for reason in payload["decision_reasons"]:
        lines.append(f"- {reason}")
    for c in payload["candidates"]:
        for reason in c["reasons"]:
            lines.append(f"- `{c['candidate_id']}`: {reason}")
    lines.extend([
        "",
        "## 9. What Would Change NO-GO Into GO",
        "",
        "A future GO would require recovered datasets with primary experimental provenance, compatible SpCas9 on-target 30-mer-or-defensibly-preprocessable sequence evidence, explicit continuous activity labels with at least 20 observed values, acceptable independence, at least 1,000 unique compatible observations per accepted candidate, and at least 2,000 total newly accepted unique compatible observations.",
        "",
        "## 10. Scope Confirmation",
        "",
        "- No model training occurred.",
        "- No model evaluation occurred.",
        "- No Moreno evaluation or raw Moreno access occurred.",
        "- No dataset pooling occurred.",
        "- No label or sequence transformation occurred.",
        "- Phase 18 was not started.",
        "",
        "## 11. Canonical Integrity",
        "",
        f"canonical_integrity_unchanged = {payload['canonical_integrity']['unchanged']}",
        "",
    ])
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17g_data_sufficiency(timestamp=now.isoformat())
    payload["git"] = {
        "head": _git(["rev-parse", "HEAD"]),
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
    }
    result_path = Path("results") / f"phase17g_data_sufficiency_{stamp}.json"
    report_path = Path("docs") / "phase17g_data_sufficiency_report.md"
    payload["artifacts"] = {
        "data_sufficiency_json": str(result_path).replace("\\", "/"),
        "report_md": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17G data sufficiency audit complete: {result_path}")
    print(f"Phase 17G report: {report_path}")
    print(f"Final decision: {payload['final_decision']}")
    return payload


if __name__ == "__main__":
    main()
