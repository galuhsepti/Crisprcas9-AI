#!/usr/bin/env python3
"""Run Phase 17I biological-independence and CRISPRon recovery audit."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_recovery.biological_independence_crispron import build_phase17i, dump_json


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def render_report(payload: dict) -> str:
    corsi = payload["corsi"]
    crispron = payload["crispron"]
    conflict = corsi["condition_conflict"]
    overlap = crispron.get("overlap_with_deepspcas9", {}).get("all_recoverable", {})
    lines = [
        "# Phase 17I - Biological Independence and CRISPRon Recovery",
        "",
        "## Scope",
        "",
        "- Phase 18 not started.",
        "- No model training.",
        "- Moreno not accessed.",
        "- Phase 17G not rerun.",
        "- No label harmonization or condition pooling.",
        "",
        "## Corsi Biological-Independence Audit",
        "",
        f"- Raw observations: {corsi['corsi_raw_observation_n']}",
        f"- Unique 30-mers: {corsi['corsi_unique_30mer_n']}",
        f"- Unique 23-mer target+PAM sequences: {corsi['corsi_unique_23mer_n']}",
        f"- Unique 20-mer spacers: {corsi['corsi_unique_spacer20_n']}",
        f"- Unique spacer-condition pairs: {corsi['corsi_unique_spacer_condition_pairs_n']}",
        f"- Effective guide diversity ratio: {corsi['effective_guide_diversity_ratio']:.4f}",
        f"- Shared 20-mer spacers between Dox- and Dox+: {corsi['shared_20mer_spacers_between_conditions']}",
        f"- Identical 30-mers shared between Dox- and Dox+: {corsi['identical_30mers_shared_between_conditions']}",
        f"- Explicit gene/locus count from gRNA_ID: {corsi['joint']['unique_genes_loci_if_explicitly_available']}",
        "",
        "The approximately 1022 unique 30-mers are primarily PAM/context variants derived from a much smaller set of spacer sequences. They should not be treated as 1022 biologically independent sgRNA observations merely because the full 30-mer differs.",
        "",
        "## Corsi Condition Conflict Audit",
        "",
        f"- Shared sequence count: {conflict['shared_sequence_n']}",
        f"- Shared sequences with label differences: {conflict['shared_sequence_with_label_difference_n']}",
        f"- Median absolute label difference: {conflict['median_absolute_label_difference']:.6f}",
        f"- Mean absolute label difference: {conflict['mean_absolute_label_difference']:.6f}",
        f"- Pooling ambiguity for sequence-only model: {conflict['pooling_creates_target_ambiguity']}",
        f"- CORSI_MAIN_TRAINING_SUITABILITY: {corsi['training_suitability']}",
        "",
        corsi["training_suitability_reason"],
        "",
        "## CRISPRon Recovery",
        "",
        f"- Retrieval status: {crispron['retrieval_status']}",
        f"- Source URL: {crispron.get('source_url')}",
        f"- File size: {crispron.get('file_size_bytes')}",
        f"- SHA-256: {crispron.get('file_sha256')}",
        f"- HTTP/content type: {crispron.get('http_content_type')}",
        f"- Publication relationship: {crispron.get('publication_relationship')}",
        "",
    ]
    if crispron["retrieval_status"] == "RECOVERED":
        lines.extend(["## CRISPRon Workbook Inventory", ""])
        for sheet in crispron["workbook_inventory"]["sheets"]:
            lines.append(f"- `{sheet['sheet_name']}`: {sheet['row_count']} rows; columns: {', '.join(sheet['column_names'])}")
        lines.extend([
            "",
            "## CRISPRon Source Separation",
            "",
            f"- Source identity recoverable: {crispron['sources']['source_identity_recoverable']}",
            f"- Xiang/Luo raw rows: {crispron['xiang_luo'].get('raw_n')}",
            f"- Xiang/Luo unique 30-mers: {crispron['xiang_luo'].get('unique_30mer_n')}",
            f"- Xiang/Luo unique 20-mers: {crispron['xiang_luo'].get('unique_spacer20_n')}",
            f"- Kim raw rows: {crispron['kim'].get('raw_n')}",
            f"- Kim unique 30-mers: {crispron['kim'].get('unique_30mer_n')}",
            "",
            "## DeepSpCas9 Overlap",
            "",
            f"- All recoverable raw rows: {overlap.get('raw_n')}",
            f"- Unique 30-mers: {overlap.get('unique_30mer_n')}",
            f"- Exact DeepSpCas9 overlap: {overlap.get('exact_deepspcas9_overlap_n')}",
            f"- Exact overlap fraction: {overlap.get('exact_deepspcas9_overlap_fraction')}",
            f"- New unique 30-mers: {overlap.get('new_unique_30mer_n')}",
            f"- Unique 20-mer spacers: {overlap.get('unique_spacer20_n')}",
            f"- Spacer overlap with DeepSpCas9: {overlap.get('spacer20_overlap_with_deepspcas9_n')}",
            f"- New unique 20-mer spacers: {overlap.get('new_unique_spacer20_n')}",
            "",
            "## CRISPRon Label Audit",
            "",
        ])
        label = crispron["label_audit"]
        for key in ["label_name", "label_definition", "experimental_measurement", "scale", "minimum", "maximum", "mean", "median", "SD", "IQR", "number_of_unique_values", "missing_values", "classification", "derivation_assessment"]:
            lines.append(f"- {key}: {label.get(key)}")
    else:
        lines.extend([
            "A manual recovery manifest was created under `data/phase17i/manifests/` with the official URL and expected filename.",
            "",
        ])
    lines.extend([
        "",
        "## Preliminary CRISPRon Suitability",
        "",
        f"CRISPRON_PRELIMINARY_STATUS: {crispron['preliminary_status']}",
        "",
        "This is not acceptance. Acceptance can only occur in a later Phase 17G re-evaluation.",
        "",
        "## Canonical Integrity",
        "",
        f"- Canonical integrity unchanged: {payload['canonical_integrity']['unchanged']}",
        "- Canonical data not modified.",
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
    payload = build_phase17i(timestamp=now.isoformat())
    payload["git"] = {
        "head": _git(["rev-parse", "HEAD"]),
        "status": _git(["status", "--short"]),
        "diff_stat": _git(["diff", "--stat"]),
    }
    result_path = Path("results") / f"phase17i_biological_independence_crispron_recovery_{stamp}.json"
    report_path = Path("docs") / "phase17i_biological_independence_crispron_recovery_report.md"
    payload["artifacts"] = {
        "json": str(result_path).replace("\\", "/"),
        "report": str(report_path).replace("\\", "/"),
    }
    dump_json(result_path, payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Phase 17I audit complete: {result_path}")
    print(f"Phase 17I report: {report_path}")
    print(f"Corsi suitability: {payload['corsi']['training_suitability']}")
    print(f"CRISPRon status: {payload['crispron']['retrieval_status']}")
    return payload


if __name__ == "__main__":
    main()
