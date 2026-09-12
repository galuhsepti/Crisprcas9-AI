#!/usr/bin/env python3
"""Run Phase 17G-R1 data sufficiency reassessment."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.data_sufficiency_reassessment import build_phase17g_r1, dump_json


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def _candidate(payload: dict, candidate_id: str) -> dict:
    return next(c for c in payload["candidates"] if c["candidate_id"] == candidate_id)


def render_report(payload: dict) -> str:
    xl = _candidate(payload, "crispron_xiang_luo")
    kim = _candidate(payload, "crispron_kim")
    corsi = _candidate(payload, "corsi_2022")
    lines = [
        "# Phase 17G-R1 - Data Sufficiency and Readiness Re-Evaluation",
        "",
        "## Scope",
        "",
        "- Original Phase 17G artifacts were not overwritten.",
        "- Phase 18 not started.",
        "- No model training, evaluation, calibration, or tuning.",
        "- Moreno remains locked and was not accessed.",
        "",
        "## Basis",
        "",
        f"- Original Phase 17G decision: {payload['basis']['original_phase17g_decision']}",
        f"- New evidence phases: {', '.join(payload['basis']['new_evidence_phases'])}",
        f"- CRISPRon SHA-256: {payload['basis']['crispron_sha256']}",
        f"- SHA-256 matches expected: {payload['basis']['crispron_sha256_matches_expected']}",
        "",
        "## CRISPRon Xiang/Luo Gates",
        "",
        f"- Provenance: {xl['provenance_gate']['status']}",
        f"- Task compatibility: {xl['task_compatibility_gate']['status']}",
        f"- Sequence usability: {xl['sequence_gate']['status']}",
        f"- Label classification: {xl['label_gate']['classification']}",
        f"- Label status: {xl['label_gate']['phase17g_continuous_label_status']}",
        f"- Independence: {xl['independence_gate']['status']}",
        f"- Quantity: {xl['quantity_gate']['status']}",
        f"- Guide diversity: {xl['guide_diversity_gate']['status']} ({xl['guide_diversity_gate']['guide_diversity_ratio']:.4f})",
        f"- Information value: {xl['information_value_gate']['status']}",
        f"- Final candidate status: {xl['final_candidate_status']}",
        "",
        "Evidence chain: primary publication -> official CRISPRon/RTH resource -> recovered workbook -> explicit Dataset component -> 30mer_gRNA -> HEK293T_indel_freq_avg_d8_d10.",
        "",
        "## Xiang/Luo Sequence And Label Audit",
        "",
        f"- raw_n: {xl['raw_n']}",
        f"- valid_sequence_n: {xl['sequence_gate']['valid_sequence_n']}",
        f"- unique_30mer_n: {xl['sequence_gate']['unique_30mer_n']}",
        f"- unique_23mer_n: {xl['sequence_gate']['unique_23mer_n']}",
        f"- unique_spacer20_n: {xl['sequence_gate']['unique_spacer20_n']}",
        f"- malformed_sequence_n: {xl['sequence_gate']['malformed_sequence_n']}",
        f"- ambiguous_base_n: {xl['sequence_gate']['ambiguous_base_n']}",
        f"- DeepSpCas9 exact 30-mer overlap: {xl['independence_gate']['overlap_30mer_n']}",
        f"- New unique 30-mers: {xl['independence_gate']['new_unique_30mer_n']}",
        f"- DeepSpCas9 spacer overlap: {xl['independence_gate']['overlap_spacer20_n']}",
        f"- New unique spacers: {xl['independence_gate']['new_unique_spacer20_n']}",
        f"- label n / finite / missing: {xl['label_gate']['n']} / {xl['label_gate']['finite_n']} / {xl['label_gate']['missing_n']}",
        f"- label min / median / max: {xl['label_gate']['min']} / {xl['label_gate']['median']} / {xl['label_gate']['max']}",
        f"- label mean / SD / IQR: {xl['label_gate']['mean']} / {xl['label_gate']['SD']} / {xl['label_gate']['IQR']}",
        f"- label unique values: {xl['label_gate']['unique_value_n']}",
        "",
        "## Corsi Handling",
        "",
        f"- Unique 30-mers: {corsi['sequence_gate']['unique_30mer_n']}",
        f"- Unique 20-mer spacers: {corsi['sequence_gate']['unique_spacer20_n']}",
        f"- Guide diversity: {corsi['guide_diversity_gate']['status']} ({corsi['guide_diversity_gate']['guide_diversity_ratio']:.4f})",
        f"- Final status: {corsi['final_candidate_status']}",
        f"- Exclusion reason: {corsi['exclusion_reason']}",
        "",
        "Corsi is not counted toward the accepted training-data total because its 30-mer count is dominated by PAM/context permutations over only 4 spacers and Dox-/Dox+ labels conflict for identical sequences.",
        "",
        "## CRISPRon Kim Handling",
        "",
        f"- Unique 30-mers: {kim['sequence_gate']['unique_30mer_n']}",
        f"- DeepSpCas9 exact overlap: {kim['independence_gate']['overlap_30mer_n']}",
        f"- Overlap fraction: {kim['independence_gate']['overlap_30mer_fraction']}",
        f"- Final status: {kim['final_candidate_status']}",
        f"- Exclusion reason: {kim['exclusion_reason']}",
        "",
        "## Accepted-Data Accounting",
        "",
        "| candidate | raw_n | unique_30mer_n | unique_spacer20_n | DeepSpCas9_overlap_n | new_unique_n | accepted_for_future_pipeline | accepted_new_unique_n | exclusion_reason |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in payload["accepted_data_accounting"]:
        lines.append(
            f"| {row['candidate']} | {row['raw_n']} | {row['unique_30mer_n']} | {row['unique_spacer20_n']} | "
            f"{row['DeepSpCas9_overlap_n']} | {row['new_unique_n']} | {row['accepted_for_future_pipeline']} | "
            f"{row['accepted_new_unique_n']} | {row['exclusion_reason']} |"
        )
    lines.extend([
        "",
        "## Final Decision",
        "",
        f"- accepted_candidate_count: {payload['aggregate']['accepted_candidate_count']}",
        f"- total_new_unique_compatible_observations: {payload['aggregate']['total_new_unique_compatible_observations']}",
        f"- minimum_required: {payload['aggregate']['minimum_required']}",
        f"- surplus_or_deficit: {payload['aggregate']['surplus_or_deficit']}",
        f"- final_decision: {payload['final_decision']}",
        "",
        "Decision reasons:",
    ])
    lines.extend([f"- {reason}" for reason in payload["decision_reasons"]])
    lines.extend([
        "",
        "## Canonical Integrity",
        "",
        f"- unchanged: {payload['canonical_integrity']['unchanged']}",
        "",
        "## Git",
        "",
        f"- HEAD: {payload['git']['head']}",
        "```",
        payload["git"]["diff_stat"],
        "```",
        "```",
        payload["git"]["status"],
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase17g_r1(timestamp=now.isoformat())
    payload["git"] = {
        "head": _git(["rev-parse", "HEAD"]),
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
    }
    result_path = Path("results") / f"phase17g_r1_data_sufficiency_reassessment_{stamp}.json"
    report_path = Path("docs") / "phase17g_r1_data_sufficiency_reassessment_report.md"
    payload["artifacts"] = {
        "json": str(result_path).replace("\\", "/"),
        "report": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17G-R1 reassessment complete: {result_path}")
    print(f"Phase 17G-R1 report: {report_path}")
    print(f"Final decision: {payload['final_decision']}")
    return payload


if __name__ == "__main__":
    main()
